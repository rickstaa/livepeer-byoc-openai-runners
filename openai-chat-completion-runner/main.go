package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

type livepeerHeader struct {
	Request        string `json:"request"`
	Capability     string `json:"capability"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}

func main() {
	addr := env("RUNNER_ADDR", ":8080")
	upstream := env("UPSTREAM_URL", "")
	if upstream == "" {
		log.Fatal("UPSTREAM_URL is required, e.g. http://HOST:PORT/v1/chat/completions")
	}

	// Streaming-friendly transport
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     false, // SSE is usually HTTP/1.1
		MaxIdleConns:          200,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	client := &http.Client{Transport: transport}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Optional: honor Livepeer timeout_seconds if present
		ctx := r.Context()
		if lp, ok := decodeLivepeerHeader(r.Header.Get("Livepeer")); ok && lp.TimeoutSeconds > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, time.Duration(lp.TimeoutSeconds)*time.Second)
			defer cancel()
		}

		// Forward request to upstream (OpenAI-compatible endpoint)
		const maxBody = 5 << 20 // 5MB
		bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, maxBody))
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		_ = r.Body.Close()

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstream, bytes.NewReader(bodyBytes))
		if err != nil {
			http.Error(w, "failed to create upstream request", http.StatusBadGateway)
			return
		}
		req.ContentLength = int64(len(bodyBytes))

		// Pass-through essential headers
		copyHeader(req.Header, r.Header, []string{
			"Content-Type",
			"Accept",
		})

		// IMPORTANT: do NOT forward Livepeer header upstream
		req.Header.Del("Livepeer")
		// Also do not forward any user Authorization (Traefik can do internal auth if you want)
		req.Header.Del("Authorization")

		resp, err := client.Do(req)
		if err != nil {
			status := http.StatusBadGateway
			if errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "context deadline exceeded") {
				status = http.StatusGatewayTimeout
			}
			http.Error(w, "upstream request failed: "+err.Error(), status)
			return
		}
		defer resp.Body.Close()

		// Copy response headers/status and stream body (SSE passthrough)
		copyAllHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		streamResponse(w, resp.Body)
	})

	// Simple health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	log.Printf("BYOC runner listening on %s, upstream=%s", addr, upstream)
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}

func env(k, def string) string {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	return v
}

func decodeLivepeerHeader(v string) (livepeerHeader, bool) {
	var lp livepeerHeader
	if v == "" {
		return lp, false
	}
	raw, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return lp, false
	}
	if err := json.Unmarshal(raw, &lp); err != nil {
		return lp, false
	}
	return lp, true
}

func copyHeader(dst http.Header, src http.Header, keys []string) {
	for _, k := range keys {
		if v := src.Get(k); v != "" {
			dst.Set(k, v)
		}
	}
}

func copyAllHeaders(dst http.Header, src http.Header) {
	for k, vv := range src {
		// Avoid hop-by-hop headers
		if strings.EqualFold(k, "Connection") ||
			strings.EqualFold(k, "Keep-Alive") ||
			strings.EqualFold(k, "Proxy-Authenticate") ||
			strings.EqualFold(k, "Proxy-Authorization") ||
			strings.EqualFold(k, "TE") ||
			strings.EqualFold(k, "Trailer") ||
			strings.EqualFold(k, "Transfer-Encoding") ||
			strings.EqualFold(k, "Upgrade") {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func streamResponse(w http.ResponseWriter, body io.Reader) {
	// Ensure streaming flush
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			_, _ = w.Write(buf[:n])
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

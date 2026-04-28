package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestChatCompletionsHandler_AcceptsBodyUnderLimit verifies the size guard
// lets a sub-cap payload through to the upstream and forwards the response.
func TestChatCompletionsHandler_AcceptsBodyUnderLimit(t *testing.T) {
	var receivedBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		receivedBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	h := newChatCompletionsHandler(http.DefaultClient, upstream.URL, 1024)

	body := `{"model":"x","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%q", rec.Code, rec.Body.String())
	}
	if receivedBody != body {
		t.Fatalf("upstream got %q, want %q", receivedBody, body)
	}
	if got := rec.Body.String(); got != `{"ok":true}` {
		t.Fatalf("response body = %q, want %q", got, `{"ok":true}`)
	}
}

// TestChatCompletionsHandler_Returns413WhenOverLimit verifies the size guard
// rejects oversized payloads with a clean 413 *without* contacting the
// upstream. This is the behavior that distinguishes MaxBytesReader from
// io.LimitReader (which would silently truncate).
func TestChatCompletionsHandler_Returns413WhenOverLimit(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("upstream should not be called when body exceeds limit")
	}))
	defer upstream.Close()

	const cap = int64(1024)
	h := newChatCompletionsHandler(http.DefaultClient, upstream.URL, cap)

	body := strings.Repeat("a", int(cap)+1) // one byte over
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413; body=%q", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "request body too large") {
		t.Fatalf("response body = %q, want to contain %q", rec.Body.String(), "request body too large")
	}
}

// TestChatCompletionsHandler_RejectsNonPost guards a small extra invariant:
// non-POST methods get 405 before any body is read.
func TestChatCompletionsHandler_RejectsNonPost(t *testing.T) {
	h := newChatCompletionsHandler(http.DefaultClient, "http://unused", 1024)

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	h(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

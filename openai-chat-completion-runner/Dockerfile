FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
RUN go build -o /bin/runner .

FROM alpine:3.20
RUN adduser -D -H runner
USER runner
ENV RUNNER_ADDR=:8080
EXPOSE 8080
COPY --from=build /bin/runner /runner
ENTRYPOINT ["/runner"]


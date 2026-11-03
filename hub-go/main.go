// tvbl-mcp-hub (Go): single-process replacement for the Cloudflare
// Worker + Durable Object hub. One binary serves arbitrarily many pairs.
//
//	POST /pair/challenge        {bits}                 -> {nonce, bits, expires}
//	POST /pair                  {label,nonce,solution} -> pair credentials
//	POST /mcp/{pair}/{token}                           -> MCP JSON-RPC 2.0
//	GET  /run?pair=<id>                                -> runner WebSocket
//	GET  /health                                       -> liveness
//
// Abuse protections (parity with the Worker's PairGuard DO):
//   - SHA-256 proof-of-work for /pair (leading zero bits, one-time nonce spend)
//   - per-CIDR rate limit (/24 IPv4, /64 IPv6) on /pair
//   - unclaimed pairs expire after PAIR_TTL (sweeper goroutine, DO-alarm equivalent)
//   - jobs expire after 1h, idempotency keys after 10m, capped per pair
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

var (
	listenAddr     = flag.String("listen", "127.0.0.1:8787", "listen address")
	powBits        = flag.Int("pow-bits", 15, "proof-of-work leading zero bits for /pair (0 disables)")
	pairRateLimit  = flag.Int("pair-rate", 6, "pairs per hour per CIDR bucket (0 disables)")
	pairTTLSeconds = flag.Int("pair-ttl", 86400, "seconds before an unclaimed pair is deleted")
	runnerURL      = flag.String("runner-url", "", "runner page URL reported by site_info")
)

type config struct {
	powBits       int
	pairRate      int
	pairTTL       time.Duration
	runnerURL     string
	defaultWaitMS int64
	runWaitMS     int64
	maxWaitMS     int64
}

func loadConfig() config {
	cfg := config{
		powBits:       *powBits,
		pairRate:      *pairRateLimit,
		pairTTL:       time.Duration(*pairTTLSeconds) * time.Second,
		runnerURL:     *runnerURL,
		defaultWaitMS: 8_000,
		runWaitMS:     20_000,
		maxWaitMS:     120_000,
	}
	if cfg.powBits < 0 {
		cfg.powBits = 0
	}
	if cfg.powBits > 24 {
		cfg.powBits = 24
	}
	return cfg
}

func main() {
	flag.Parse()
	cfg := loadConfig()

	h := newHub(cfg)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/pair/challenge", h.handleChallenge)
	mux.HandleFunc("/pair", h.handlePair)
	mux.HandleFunc("/run", h.handleRun)
	mux.HandleFunc("/mcp/", h.handleMCP)
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	})

	srv := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go h.sweepLoop()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("tvbl-mcp-hub-go listening on %s (pow-bits=%d pair-rate=%d/h)", *listenAddr, cfg.powBits, cfg.pairRate)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	<-stop
	log.Println("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

// sweepLoop GCs expired nonces, rate windows, unclaimed pairs and old jobs
// (the DO alarm + prune equivalents, in one goroutine).
func (h *hub) sweepLoop() {
	t := time.NewTicker(time.Minute)
	defer t.Stop()
	for range t.C {
		now := time.Now()
		h.relay.g.sweep(now)
		h.relay.pairs.sweepUnclaimed(now, h.cfg.pairTTL)
		h.relay.jobs.prune(now)
	}
}

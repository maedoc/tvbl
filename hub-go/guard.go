package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/bits"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const nonceTTL = 5 * time.Minute

// ── guard: proof-of-work + rate limit (PairGuard DO equivalent) ─────────────

type nonceRecord struct {
	exp  time.Time
	bits int
	used bool
}

type rateRecord struct {
	count   int
	resetAt time.Time
}

type guard struct {
	mu     sync.Mutex
	nonces map[string]*nonceRecord
	rates  map[string]*rateRecord
}

func newGuard() *guard {
	return &guard{nonces: map[string]*nonceRecord{}, rates: map[string]*rateRecord{}}
}

func (g *guard) challenge(bits int) (string, time.Time) {
	nonce := randomHex(16)
	exp := time.Now().Add(nonceTTL)
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nonces[nonce] = &nonceRecord{exp: exp, bits: bits}
	return nonce, exp
}

// spend verifies a PoW solution and marks the nonce used exactly once.
func (g *guard) spend(nonce, solution string, required int) (bool, string) {
	g.mu.Lock()
	rec := g.nonces[nonce]
	g.mu.Unlock()
	if rec == nil {
		return false, "unknown_nonce"
	}
	if rec.used {
		return false, "nonce_reused"
	}
	if time.Now().After(rec.exp) {
		return false, "nonce_expired"
	}
	if leadingZeroBits(nonce+solution) < required {
		return false, "insufficient_work"
	}
	g.mu.Lock()
	rec.used = true
	g.mu.Unlock()
	return true, ""
}

func (g *guard) rate(bucket string, limit int, window time.Duration) (bool, int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	now := time.Now()
	rec := g.rates[bucket]
	if rec == nil || now.After(rec.resetAt) {
		g.rates[bucket] = &rateRecord{count: 1, resetAt: now.Add(window)}
		return true, limit - 1
	}
	if rec.count >= limit {
		return false, int(rec.resetAt.Sub(now) / time.Second)
	}
	rec.count++
	return true, limit - rec.count
}

func (g *guard) sweep(now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	for nonce, rec := range g.nonces {
		if now.After(rec.exp) && rec.used {
			delete(g.nonces, nonce)
		}
	}
	for bucket, rec := range g.rates {
		if now.After(rec.resetAt.Add(time.Hour)) {
			delete(g.rates, bucket)
		}
	}
}

func leadingZeroBits(text string) int {
	sum := sha256.Sum256([]byte(text))
	n := 0
	for _, b := range sum {
		if b == 0 {
			n += 8
			continue
		}
		n += bits.LeadingZeros8(b)
		break
	}
	return n
}

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// familyBucket buckets an IP to a /24 (IPv4) or /64 (IPv6), the same coarse
// grouping the Worker uses so one customer cannot mint unlimited pairs.
func familyBucket(ip string) string {
	parsed := net.ParseIP(strings.TrimSpace(ip))
	if parsed == nil {
		return "unknown"
	}
	if v4 := parsed.To4(); v4 != nil {
		return fmt.Sprintf("%s/24", v4.Mask(net.CIDRMask(24, 32)).String())
	}
	return fmt.Sprintf("%s/64", parsed.Mask(net.CIDRMask(64, 128)).String())
}

func clientIP(r *http.Request) string {
	if fwd := r.Header.Get("x-forwarded-for"); fwd != "" {
		return strings.TrimSpace(strings.Split(fwd, ",")[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func itoa(n int) string { return strconv.Itoa(n) }

package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ── pairing endpoints ───────────────────────────────────────────────────────

func (h *hub) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok": true, "now": time.Now().UnixMilli(), "service": "tvbl-mcp-hub-go",
	})
}

func (h *hub) handleChallenge(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Bits int `json:"bits"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	bits := body.Bits
	if bits <= 0 || bits > 24 {
		bits = h.cfg.powBits
	}
	if bits == 0 {
		writeJSON(w, http.StatusOK, map[string]any{"nonce": nil, "bits": 0})
		return
	}
	nonce, exp := h.relay.g.challenge(bits)
	writeJSON(w, http.StatusOK, map[string]any{
		"nonce": nonce, "bits": bits, "expires": exp.UnixMilli(),
	})
}

func (h *hub) handlePair(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Label    string `json:"label"`
		Nonce    string `json:"nonce"`
		Solution string `json:"solution"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "bad_request"})
		return
	}

	if h.cfg.powBits > 0 {
		if body.Nonce == "" || body.Solution == "" {
			writeJSON(w, http.StatusUnauthorized, map[string]any{"error": "proof_of_work_required"})
			return
		}
		if ok, reason := h.relay.g.spend(body.Nonce, body.Solution, h.cfg.powBits); !ok {
			writeJSON(w, http.StatusUnauthorized, map[string]any{
				"error": "proof_of_work_failed", "reason": reason,
			})
			return
		}
	} else if h.cfg.pairRate > 0 {
		bucket := familyBucket(clientIP(r))
		if ok, retryAfter := h.relay.g.rate(bucket, h.cfg.pairRate, time.Hour); !ok {
			w.Header().Set("retry-after", strconv.Itoa(retryAfter))
			writeJSON(w, http.StatusTooManyRequests, map[string]any{
				"error": "rate_limited", "retry_after": retryAfter,
			})
			return
		}
	}

	pairID := randomHex(8)
	token := randomHex(32)
	if !h.relay.pairs.init(pairID, sha256Hex(token), body.Label) {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "pair init failed"})
		return
	}

	origin := schemeOf(r) + "://" + r.Host
	writeJSON(w, http.StatusCreated, map[string]any{
		"pair_id":        pairID,
		"token":          token,
		"mcp_url":        origin + "/mcp/" + pairID + "/" + token,
		"mcp_url_bearer": origin + "/mcp/" + pairID,
		"runner_ws":      wsScheme(origin) + "/run?pair=" + pairID,
		"runner_url":     h.cfg.runnerURL,
	})
}

// ── small helpers ───────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("content-type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func schemeOf(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	if fwd := r.Header.Get("x-forwarded-proto"); fwd != "" {
		return fwd
	}
	return "http"
}

func wsScheme(origin string) string {
	if strings.HasPrefix(origin, "https") {
		return "wss" + strings.TrimPrefix(origin, "https")
	}
	return "ws" + strings.TrimPrefix(origin, "http")
}

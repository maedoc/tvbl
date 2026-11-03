package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ── relay: runner sockets + job dispatch (RunnerHub DO equivalent) ──────────

type attachment struct {
	authed   bool
	label    string
	lastSeen time.Time
}

// relay owns the runner sockets. One process, so a per-pair map guarded by a
// single mutex replaces the "one DO per pair" single-writer model.
type relay struct {
	mu      sync.Mutex
	runners map[string]map[*websocket.Conn]*attachment
	jobs    *jobStore
	pairs   *pairStore
	g       *guard
}

func newRelay() *relay {
	return &relay{
		runners: map[string]map[*websocket.Conn]*attachment{},
		jobs:    newJobStore(),
		pairs:   newPairStore(),
		g:       newGuard(),
	}
}

// authedRunners returns the pair's authenticated sockets, most recently seen
// first (same "most recent socket wins" policy as the Worker).
func (r *relay) authedRunners(pairID string) []*websocket.Conn {
	r.mu.Lock()
	defer r.mu.Unlock()
	conns := r.runners[pairID]
	if len(conns) == 0 {
		return nil
	}
	type entry struct {
		ws  *websocket.Conn
		att *attachment
	}
	list := make([]entry, 0, len(conns))
	for ws, att := range conns {
		if att.authed {
			list = append(list, entry{ws, att})
		}
	}
	// insertion order sort: keep it simple, newest lastSeen first
	for i := 1; i < len(list); i++ {
		for k := i; k > 0 && list[k].att.lastSeen.After(list[k-1].att.lastSeen); k-- {
			list[k], list[k-1] = list[k-1], list[k]
		}
	}
	out := make([]*websocket.Conn, len(list))
	for i, e := range list {
		out[i] = e.ws
	}
	return out
}

func (r *relay) runnerStatus(pairID string) (online bool, label string, lastSeen time.Time, count int) {
	conns := r.authedRunners(pairID)
	if len(conns) == 0 {
		return false, "", time.Time{}, 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	att := r.runners[pairID][conns[0]]
	return true, att.label, att.lastSeen, len(conns)
}

func (r *relay) send(pairID string, frame any) bool {
	data, err := json.Marshal(frame)
	if err != nil {
		return false
	}
	for _, ws := range r.authedRunners(pairID) {
		r.mu.Lock()
		att := r.runners[pairID][ws]
		att.lastSeen = time.Now()
		r.mu.Unlock()
		_ = ws.WriteMessage(websocket.TextMessage, data)
		return true
	}
	return false
}

type jobFrame struct {
	Type string         `json:"type"`
	ID   string         `json:"id"`
	Kind string         `json:"kind"`
	Args map[string]any `json:"args"`
}

type cancelFrame struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

// ── runner WebSocket endpoint ───────────────────────────────────────────────

const (
	waitCap          = 25 * time.Second
	closeAuthTimeout = 4001
	closeBadToken    = 4003
	authTimeout      = 10 * time.Second
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(*http.Request) bool { return true },
}

func (h *hub) handleRun(w http.ResponseWriter, r *http.Request) {
	pairID := r.URL.Query().Get("pair")
	if pairID == "" || !isID(pairID) {
		http.Error(w, "bad pair", http.StatusBadRequest)
		return
	}
	if !h.relay.pairs.exists(pairID) {
		http.Error(w, "unknown pair", http.StatusUnauthorized)
		return
	}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	h.serveRunner(pairID, conn)
}

func (h *hub) serveRunner(pairID string, conn *websocket.Conn) {
	defer func() {
		h.relay.mu.Lock()
		if conns := h.relay.runners[pairID]; conns != nil {
			delete(conns, conn)
			if len(conns) == 0 {
				delete(h.relay.runners, pairID)
			}
		}
		h.relay.mu.Unlock()
		_ = conn.Close()
	}()

	h.relay.mu.Lock()
	if h.relay.runners[pairID] == nil {
		h.relay.runners[pairID] = map[*websocket.Conn]*attachment{}
	}
	h.relay.runners[pairID][conn] = &attachment{lastSeen: time.Now()}
	h.relay.mu.Unlock()

	_ = conn.SetReadDeadline(time.Now().Add(authTimeout))
	var auth struct {
		Type  string `json:"type"`
		Token string `json:"token"`
		Label string `json:"label"`
	}
	if err := conn.ReadJSON(&auth); err != nil || auth.Type != "auth" {
		deadline := time.Now().Add(time.Second)
		_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(closeAuthTimeout, "auth timeout"), deadline)
		return
	}
	if !h.relay.pairs.verify(pairID, auth.Token) {
		_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(closeBadToken, "bad token"), time.Now().Add(time.Second))
		return
	}
	h.relay.mu.Lock()
	h.relay.runners[pairID][conn] = &attachment{authed: true, label: auth.Label, lastSeen: time.Now()}
	h.relay.mu.Unlock()

	_ = conn.SetReadDeadline(time.Time{})
	_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"hello","pair":true}`))

	for {
		var frame struct {
			Type       string         `json:"type"`
			ID         string         `json:"id"`
			OK         bool           `json:"ok"`
			Kind       string         `json:"kind"`
			Args       map[string]any `json:"args"`
			Content    []any          `json:"content,omitempty"`
			Structured map[string]any `json:"structured,omitempty"`
			Error      *jobError      `json:"error,omitempty"`
		}
		if err := conn.ReadJSON(&frame); err != nil {
			return
		}
		switch frame.Type {
		case "result":
			h.recordResult(frame.ID, frame.OK, frame.Content, frame.Structured, frame.Error)
		case "cancel":
			job := h.relay.jobs.get(frame.ID)
			if job != nil {
				job.finish(jobCancelled, nil, map[string]any{
					"ok": false,
					"error": jobError{
						Code:    "cancelled",
						Message: "the job was cancelled; the cell may still be computing in the browser",
					},
				}, nil)
			}
		case "progress":
			// not persisted in this process
		}
	}
}

func (h *hub) recordResult(id string, ok bool, content []any, structured map[string]any, err *jobError) {
	job := h.relay.jobs.get(id)
	if job == nil {
		return
	}
	if job.snapshot() == jobCancelled {
		return
	}
	status := jobFailed
	if ok {
		status = jobOK
	}
	job.finish(status, content, structured, err)
}

// ── call path: create a job, dispatch, optionally wait inline ───────────────

type callOutcome struct {
	status     string // "offline" | "running" | "ok" | "error" | "cancelled"
	jobID      string
	kind       string
	content    []any
	structured map[string]any
	err        *jobError
}

func (h *hub) call(pairID, kind string, args map[string]any, waitMS int64, idemKey string) callOutcome {
	if idemKey != "" {
		if prior := h.relay.jobs.lookupIdem("idem:"+kind+":"+idemKey, time.Now()); prior != nil {
			return h.awaitJob(prior, waitMS)
		}
	}
	if len(h.relay.authedRunners(pairID)) == 0 {
		return callOutcome{status: "offline"}
	}

	id := pairID + ":" + randomHex(12)
	job := h.relay.jobs.create(id, kind, args, idemKey)

	if !h.relay.send(pairID, jobFrame{Type: "job", ID: id, Kind: kind, Args: args}) {
		job.finish(jobFailed, nil, nil, &jobError{
			Code:    "runner_unreachable",
			Message: "the runner socket closed before the job was sent",
		})
		return callOutcome{status: "offline"}
	}

	if waitMS <= 0 {
		return callOutcome{status: "running", jobID: id}
	}
	select {
	case <-job.done:
	case <-time.After(time.Duration(waitMS) * time.Millisecond):
	}
	return h.outcomeOf(job)
}

func (h *hub) awaitJob(job *jobRecord, waitMS int64) callOutcome {
	if !job.finished() && waitMS > 0 {
		select {
		case <-job.done:
		case <-time.After(minDuration(time.Duration(waitMS)*time.Millisecond, waitCap)):
		}
	}
	return h.outcomeOf(job)
}

func (h *hub) outcomeOf(job *jobRecord) callOutcome {
	job.mu.Lock()
	defer job.mu.Unlock()
	out := callOutcome{
		status:     string(job.status),
		jobID:      job.id,
		kind:       job.kind,
		content:    job.content,
		structured: job.structured,
		err:        job.err,
	}
	if (job.status == jobRunning) && time.Since(job.createdAt) > jobStale {
		job.status = jobFailed
		job.err = &jobError{
			Code:    "job_stale",
			Message: "the runner never reported completion (tab closed or kernel wedged?)",
			Hint:    "call kernel_restart and re-run the cell",
		}
		job.updatedAt = time.Now()
		out.status = string(jobFailed)
		out.err = job.err
	}
	return out
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func isID(v string) bool {
	if v == "" {
		return false
	}
	if len(v) == 0 || len(v) > 64 {
		return false
	}
	for _, c := range v {
		ok := c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-'
		if !ok {
			return false
		}
	}
	return true
}

func init() {
	log.SetFlags(log.LstdFlags | log.LUTC)
}

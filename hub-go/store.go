package main

import (
	"sort"
	"strings"
	"sync"
	"time"
)

// ── pair store (RunnerHub DO "pair" record equivalent) ──────────────────────

type pairRecord struct {
	tokenHash string
	label     string
	createdAt time.Time
	claimedAt time.Time
}

type pairStore struct {
	mu    sync.Mutex
	pairs map[string]*pairRecord
}

func newPairStore() *pairStore { return &pairStore{pairs: map[string]*pairRecord{}} }

func (s *pairStore) init(pairID, tokenHash, label string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.pairs[pairID]; ok {
		return false
	}
	s.pairs[pairID] = &pairRecord{tokenHash: tokenHash, label: label, createdAt: time.Now()}
	return true
}

// verify constant-time-compares the token hash, marks the pair claimed on the
// first success and cancels the unclaimed-pair TTL (the DO alarm equivalent).
func (s *pairStore) verify(pairID, token string) bool {
	hash := sha256Hex(token)
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.pairs[pairID]
	if p == nil || p.tokenHash != hash {
		return false
	}
	if p.claimedAt.IsZero() {
		p.claimedAt = time.Now()
	}
	return true
}

func (s *pairStore) exists(pairID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pairs[pairID] != nil
}

// sweepUnclaimed deletes pairs that were minted but never claimed by a runner.
func (s *pairStore) sweepUnclaimed(now time.Time, ttl time.Duration) {
	if ttl <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, p := range s.pairs {
		if p.claimedAt.IsZero() && now.Sub(p.createdAt) > ttl {
			delete(s.pairs, id)
		}
	}
}

// ── job store (RunnerHub DO storage equivalent, in-process) ─────────────────

type jobStatus string

const (
	jobRunning   jobStatus = "running"
	jobOK        jobStatus = "ok"
	jobFailed    jobStatus = "error"
	jobCancelled jobStatus = "cancelled"
)

const (
	jobTTL      = time.Hour
	jobStale    = 15 * time.Minute
	idemTTL     = 10 * time.Minute
	maxKeptJobs = 200
)

type jobError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Hint    string `json:"hint,omitempty"`
}

type jobRecord struct {
	id         string
	kind       string
	args       map[string]any
	status     jobStatus
	createdAt  time.Time
	updatedAt  time.Time
	content    []any
	structured map[string]any
	err        *jobError
	done       chan struct{}
	mu         sync.Mutex
}

func (j *jobRecord) finished() bool {
	s := j.snapshot()
	return s == jobOK || s == jobFailed || s == jobCancelled
}

func (j *jobRecord) snapshot() jobStatus {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.status
}

func (j *jobRecord) finish(status jobStatus, content []any, structured map[string]any, err *jobError) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.status != jobRunning {
		return
	}
	j.status = status
	j.content = content
	j.structured = structured
	j.err = err
	j.updatedAt = time.Now()
	close(j.done)
}

// markStale flags a running job whose runner never reported completion.
func (j *jobRecord) markStale(now time.Time) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.status != jobRunning && j.status != "queued" {
		return false
	}
	if now.Sub(j.createdAt) <= jobStale {
		return false
	}
	j.status = jobFailed
	j.err = &jobError{
		Code:    "job_stale",
		Message: "the runner never reported completion (tab closed or kernel wedged?)",
		Hint:    "call kernel_restart and re-run the cell",
	}
	j.updatedAt = now
	return true
}

type idemRecord struct {
	jobID string
	at    time.Time
}

type jobStore struct {
	mu   sync.Mutex
	jobs map[string]*jobRecord
	idem map[string]*idemRecord
}

func newJobStore() *jobStore {
	return &jobStore{jobs: map[string]*jobRecord{}, idem: map[string]*idemRecord{}}
}

func (s *jobStore) create(id, kind string, args map[string]any, idemKey string) *jobRecord {
	now := time.Now()
	j := &jobRecord{
		id: id, kind: kind, args: args, status: jobRunning,
		createdAt: now, updatedAt: now,
		done: make(chan struct{}),
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[id] = j
	if idemKey != "" {
		s.idem[idemKey] = &idemRecord{jobID: id, at: now}
	}
	return j
}

func (s *jobStore) get(id string) *jobRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

func (s *jobStore) lookupIdem(idemKey string, now time.Time) *jobRecord {
	if idemKey == "" {
		return nil
	}
	s.mu.Lock()
	rec := s.idem[idemKey]
	s.mu.Unlock()
	if rec == nil || now.Sub(rec.at) > idemTTL {
		return nil
	}
	return s.get(rec.jobID)
}

// pending counts running jobs belonging to a pair (job ids are "<pair>:<hex>").
func (s *jobStore) pending(pairID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	prefix := pairID + ":"
	for id, j := range s.jobs {
		if strings.HasPrefix(id, prefix) && j.snapshot() == jobRunning {
			n++
		}
	}
	return n
}

func (s *jobStore) prune(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for key, rec := range s.idem {
		if now.Sub(rec.at) > idemTTL {
			delete(s.idem, key)
		}
	}
	type entry struct {
		id string
		j  *jobRecord
	}
	all := make([]entry, 0, len(s.jobs))
	for id, j := range s.jobs {
		if now.Sub(j.createdAt) > jobTTL {
			delete(s.jobs, id)
			continue
		}
		all = append(all, entry{id, j})
	}
	if len(all) <= maxKeptJobs {
		return
	}
	sort.Slice(all, func(i, k int) bool { return all[i].j.createdAt.After(all[k].j.createdAt) })
	for _, e := range all[maxKeptJobs:] {
		delete(s.jobs, e.id)
	}
}

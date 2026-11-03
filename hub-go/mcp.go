package main

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// ── MCP: stateless streamable-HTTP JSON-RPC 2.0 ─────────────────────────────
//
// Same contract as the Worker's McpServer + WebStandardStreamableHTTPServerTransport
// with sessionIdGenerator undefined: no session state, one fresh logical server
// per request. Supports initialize, notifications/initialized, ping, tools/list
// and tools/call.

const (
	serverName    = "tvbl-mcp-hub"
	serverVersion = "0.5.0-go"
	protocolVer   = "2025-06-18"

	textBudget   = 32_000
	imageMaxB64  = 1_200_000
	imageMaxN    = 4
	jobResultCap = 25_000
)

const instructions = "Drives a JupyterLite notebook in the user's browser. " +
	"Call site_info once at the start: it reports runner_online, the default kernel and the open notebook. " +
	"Calls are synchronous unless the result has status 'running'; then poll job_result with the job_id. " +
	"Before editing or running an existing cell, call notebook_read for the full cell list (stable cell ids; indices shift). " +
	"Images arrive as MCP image blocks; if you cannot see them, call notebook_figures for a text summary of the matplotlib axes. " +
	"Write calls accept an optional idempotency_key: retrying the same key within 10 minutes does not execute twice. " +
	"Large stdout is saved under _mcp/<notebook>/ and can be paged with file_read offset; file_stat reports .npy shape/dtype; " +
	"file_write supports overwrite/append/replace/lines."

const offlineText = "Runner offline: no browser tab is connected to this pair. " +
	"Ask the user to open the runner page and click Pair; nothing was executed."

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"params"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (h *hub) handleMCP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	pairID, token, ok := splitMCPToken(r)
	if !ok || !h.relay.pairs.verify(pairID, token) {
		w.Header().Set("www-authenticate", `Bearer realm="tvbl-mcp-hub"`)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		writeRPC(w, nil, nil, &rpcError{Code: -32700, Message: "parse error"})
		return
	}
	var req rpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeRPC(w, nil, nil, &rpcError{Code: -32700, Message: "parse error"})
		return
	}

	switch req.Method {
	case "initialize":
		writeRPC(w, req.ID, map[string]any{
			"protocolVersion": protocolVer,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": serverName, "version": serverVersion, "title": "TVBL JupyterLite notebook"},
			"instructions":    instructions,
		}, nil)
	case "notifications/initialized":
		w.WriteHeader(http.StatusAccepted)
	case "ping":
		writeRPC(w, req.ID, map[string]any{}, nil)
	case "tools/list":
		writeRPC(w, req.ID, map[string]any{"tools": toolsList()}, nil)
	case "tools/call":
		writeRPC(w, req.ID, h.toolsCall(pairID, req.Params.Name, req.Params.Arguments), nil)
	default:
		writeRPC(w, req.ID, nil, &rpcError{Code: -32601, Message: "method not found: " + req.Method})
	}
}

func splitMCPToken(r *http.Request) (pairID, token string, ok bool) {
	rest := strings.TrimPrefix(r.URL.Path, "/mcp/")
	parts := strings.Split(rest, "/")
	if len(parts) < 1 || parts[0] == "" || !isID(parts[0]) {
		return "", "", false
	}
	pairID = parts[0]
	if len(parts) >= 2 && parts[1] != "" {
		if !isToken(parts[1]) {
			return "", "", false
		}
		return pairID, parts[1], true
	}
	auth := r.Header.Get("authorization")
	if rest2, found := strings.CutPrefix(auth, "Bearer "); found {
		token = strings.TrimSpace(rest2)
		if isToken(token) {
			return pairID, token, true
		}
	}
	return "", "", false
}

func isToken(v string) bool {
	return len(v) >= 1 && len(v) <= 128 && isID(v)
}

func writeRPC(w http.ResponseWriter, id json.RawMessage, result any, err *rpcError) {
	if id == nil {
		id = json.RawMessage("null")
	}
	w.Header().Set("content-type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(rpcResponse{JSONRPC: "2.0", ID: id, Result: result, Error: err})
}

// ── tool registry (mirror of the Worker's SPECS) ────────────────────────────

type toolSpec struct {
	name        string
	title       string
	description string
	schema      map[string]any
	annotations map[string]any
	local       string // "", "status", "job", "cancel"
	kind        string // relay kind, defaults to name
	runWait     bool   // default wait is RUN_WAIT_MS instead of DEFAULT_WAIT_MS
}

func boolAnn(readOnly, destructive, idempotent bool) map[string]any {
	a := map[string]any{}
	if readOnly {
		a["readOnlyHint"] = true
	}
	if destructive {
		a["destructiveHint"] = true
	}
	if idempotent {
		a["idempotentHint"] = true
	}
	return a
}

func props(m ...any) map[string]any {
	out := map[string]any{}
	for i := 0; i+1 < len(m); i += 2 {
		out[m[i].(string)] = m[i+1]
	}
	return out
}

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }

func intProp(desc string, min, max int) map[string]any {
	return map[string]any{"type": "integer", "description": desc, "minimum": min, "maximum": max}
}

func idemProps() map[string]any {
	return props("idempotency_key", str("Optional. A retry with the same key within 10 minutes returns the original result instead of executing again."))
}

var commonProps = props("_confirmationReason", str("Optional. Some clients attach a confirmation reason to write tools; it is accepted uniformly and ignored."))

func withCommon(m map[string]any) map[string]any {
	out := map[string]any{"type": "object", "properties": map[string]any{}, "required": []string{}}
	for k, v := range m {
		out["properties"].(map[string]any)[k] = v
	}
	for k, v := range commonProps {
		out["properties"].(map[string]any)[k] = v
	}
	return out
}

var specs = []toolSpec{
	{
		name: "site_info", title: "Runner status", local: "status",
		description: "Use this first. Reports whether the browser runner is online, the default kernel and available kernels, the open notebook, and the hub limits.",
		annotations: boolAnn(true, false, true),
	},
	{
		name: "notebook_list", title: "List notebooks", kind: "notebook_list",
		description: "Use this to see what notebooks and files exist in the runner's browser workspace before opening or creating one.",
		annotations: boolAnn(true, false, true),
	},
	{
		name: "notebook_read", title: "Read notebook", kind: "notebook_read",
		description: "Use this before any surgical edit of an existing notebook, and whenever the user may have changed it. Returns every cell with a stable cell_id, type, source, execution state and (optionally) outputs. Long sources and outputs are truncated with an explicit marker; raise max_cell_chars / max_chars if needed.",
		annotations: boolAnn(true, false, true),
		schema: props(
			"path", str("Notebook path; defaults to the active notebook."),
			"include_outputs", map[string]any{"type": "boolean", "description": "Include cell outputs (default true). Set false for a cheap cell list."},
			"max_cell_chars", intProp("Per-cell source cap, default 6000.", 500, 60_000),
			"max_chars", intProp("Total output text cap, default 8000.", 500, 60_000),
			"max_images", intProp("How many stored figure outputs to attach (default 4).", 0, 8),
		),
	},
	{
		name: "notebook_create", title: "Create notebook", kind: "notebook_create",
		description: "Use this when the user wants a new notebook. Creates and opens it, starts the kernel, and returns the empty cell's cell_id. If the path already exists it opens that notebook instead of failing (set if_exists='error' to fail with code already_exists).",
		annotations: boolAnn(false, false, false),
		schema: merge(idemProps(),
			"path", str("e.g. demo.ipynb"),
			"kernel", str("Kernel name; defaults to site_info.default_kernel."),
			"if_exists", map[string]any{"type": "string", "enum": []string{"open", "error"}, "description": "Default 'open': reuse an existing notebook. 'error' returns code already_exists."},
		),
	},
	{
		name: "notebook_open", title: "Open notebook", kind: "notebook_open",
		description: "Use this to bring an existing notebook to the foreground so the user can see it. Opens (or reveals) the tab and starts its kernel if needed.",
		annotations: boolAnn(false, false, true),
		schema:      merge(idemProps(), "path", str("Notebook path, e.g. demo.ipynb")),
	},
	{
		name: "cell_insert", title: "Insert cell", kind: "cell_insert",
		description: "Use this to add a code or markdown cell. Returns the stable cell_id; use that id for every later edit or run. If the notebook's only cell is empty, this writes into it instead of leaving an orphan.",
		annotations: boolAnn(false, false, false),
		schema: merge(idemProps(),
			"source", str("Cell source"),
			"cell_type", map[string]any{"type": "string", "enum": []string{"code", "markdown"}},
			"index", intProp("Insert position; default appends.", 0, 1<<30),
			"notebook", str("Notebook path; defaults to the active notebook."),
		),
	},
	{
		name: "cell_edit", title: "Edit cell", kind: "cell_edit",
		description: "Use this to replace a cell's source after reading the notebook. Address the cell by cell_id, not by index.",
		annotations: boolAnn(false, false, true),
		schema:      merge(idemProps(), "cell_id", str("Stable cell id from notebook_read."), "source", str("Replacement cell source")),
	},
	{
		name: "cell_delete", title: "Delete cell", kind: "cell_delete",
		description: "Use this to remove a cell by cell_id.",
		annotations: boolAnn(false, true, false),
		schema:      merge(idemProps(), "cell_id", str("Stable cell id from notebook_read.")),
	},
	{
		name: "cell_move", title: "Move cell", kind: "cell_move",
		description: "Use this to reorder cells. Address the cell by cell_id.",
		annotations: boolAnn(false, false, false),
		schema:      merge(idemProps(), "cell_id", str("Stable cell id from notebook_read."), "index", intProp("New position.", 0, 1<<30)),
	},
	{
		name: "cell_run", title: "Run cell", kind: "cell_run", runWait: true,
		description: "Use this to execute one code cell and get its outputs (text and image blocks). Usually returns synchronously; a cell still running after wait_ms returns status 'running' with a job_id, then poll job_result. Do not re-run the cell while a job is running.",
		annotations: boolAnn(false, false, false),
		schema: merge(idemProps(),
			"cell_id", str("Stable cell id from notebook_read."),
			"wait_ms", intProp("Inline wait before returning a job_id; 0 returns immediately.", 0, 120_000),
		),
	},
	{
		name: "notebook_run_all", title: "Run all cells", kind: "notebook_run_all", runWait: true,
		description: "Use this only when the notebook is meant to be executed top to bottom; it is slower and stops at the first error. Returns a summary (and the errors) or a job_id.",
		annotations: boolAnn(false, false, false),
		schema: merge(idemProps(),
			"path", str("Notebook path; defaults to the active notebook."),
			"wait_ms", intProp("Inline wait before returning a job_id; 0 returns immediately.", 0, 120_000),
		),
	},
	{
		name: "notebook_figures", title: "Summarise figures", kind: "notebook_figures",
		description: "Use this when image outputs are not visible to you, or to verify a plot numerically. Returns a text summary of the matplotlib figures currently alive in the kernel: axes titles and labels, line labels, sample counts and min/max ranges.",
		annotations: boolAnn(true, false, true),
	},
	{
		name: "job_result", title: "Job result", local: "job",
		description: "Use this to fetch the result of a job that returned status 'running'. Returns status 'running' until the runner finishes.",
		annotations: boolAnn(true, false, true),
		schema: props(
			"job_id", str("Job id returned by a 'running' result."),
			"wait_ms", intProp("Long-poll for up to this long; default 15000.", 0, jobResultCap),
		),
	},
	{
		name: "job_cancel", title: "Cancel job", local: "cancel",
		description: "Use this to abandon a running job. The runner may keep computing until the cell finishes or the kernel is restarted.",
		annotations: boolAnn(false, false, false),
		schema:      props("job_id", str("Job id to cancel.")),
	},
	{
		name: "kernel_restart", title: "Restart kernel", kind: "kernel_restart",
		description: "Use this when the kernel is hung, out of memory, or a cell is looping. All Python state is lost; the notebook cells remain.",
		annotations: boolAnn(false, true, false),
		schema:      idemProps(),
	},
	{
		name: "file_list", title: "List files", kind: "file_list",
		description: "Use this to list files in the runner's workspace (not just notebooks).",
		annotations: boolAnn(true, false, true),
		schema:      props("path", str("Directory path; defaults to the workspace root.")),
	},
	{
		name: "file_read", title: "Read file", kind: "file_read",
		description: "Use this to read a text file from the runner's workspace. The body is returned as text and also as structuredContent.content, so clients that only read structured output still get it. Binary files are returned base64-encoded with encoding='base64'.",
		annotations: boolAnn(true, false, true),
		schema: props(
			"path", str("File path."),
			"max_chars", intProp("Default 20000.", 100, 200_000),
			"offset", intProp("Character offset for paging through large files; the result includes next_offset.", 0, 1<<40),
			"encoding", map[string]any{"type": "string", "enum": []string{"auto", "text", "base64"}, "description": "auto (default) picks base64 for known binary extensions; force it for other binaries."},
		),
	},
	{
		name: "file_write", title: "Write file", kind: "file_write",
		description: "Use this to create or edit a text file. mode='overwrite' (default) replaces it; 'append' adds to the end; 'replace' applies edits:[{old_text,new_text,replace_all?}]; 'lines' replaces start_line..end_line with content (end_line = start_line-1 inserts before start_line). No read-modify-write round trips.",
		annotations: boolAnn(false, true, true),
		schema: merge(idemProps(),
			"path", str("File path."),
			"content", str("New content; replacement text for replace/lines modes."),
			"mode", map[string]any{"type": "string", "enum": []string{"overwrite", "append", "replace", "lines"}, "description": "Default overwrite."},
			"edits", map[string]any{"type": "array", "description": "For mode='replace'.", "items": map[string]any{
				"type":       "object",
				"properties": props("old_text", str("Text to find."), "new_text", str("Replacement text."), "replace_all", map[string]any{"type": "boolean"}),
				"required":   []string{"old_text", "new_text"},
			}},
			"start_line", intProp("For mode='lines' (1-based).", 1, 1<<30),
			"end_line", intProp("For mode='lines', inclusive; end_line = start_line-1 inserts before start_line.", 0, 1<<30),
		),
	},
	{
		name: "file_stat", title: "File metadata", kind: "file_stat",
		description: "Use this before reading a data file: returns type, size, format, and for .npy the dtype/shape/fortran_order (parsed from the header), for .ipynb the cell count.",
		annotations: boolAnn(true, false, true),
		schema:      props("path", str("File path.")),
	},
	{
		name: "file_delete", title: "Delete file", kind: "file_delete",
		description: "Use this to remove a file or directory. Directories must be empty unless recursive=true.",
		annotations: boolAnn(false, true, false),
		schema: merge(idemProps(),
			"path", str("File or directory path."),
			"recursive", map[string]any{"type": "boolean", "description": "Delete directory contents too."},
		),
	},
	{
		name: "file_move", title: "Move or rename", kind: "file_move",
		description: "Use this to move or rename a file or directory.",
		annotations: boolAnn(false, true, true),
		schema:      merge(idemProps(), "from", str("Source path."), "to", str("Destination path.")),
	},
	{
		name: "file_mkdir", title: "Create directory", kind: "file_mkdir",
		description: "Use this to create a directory (idempotent, like mkdir -p).",
		annotations: boolAnn(false, false, true),
		schema:      merge(idemProps(), "path", str("Directory path.")),
	},
}

func merge(idem map[string]any, kv ...any) map[string]any {
	out := props(kv...)
	for k, v := range idem {
		out[k] = v
	}
	return out
}

func toolsList() []map[string]any {
	out := make([]map[string]any, 0, len(specs))
	for _, s := range specs {
		t := map[string]any{
			"name":        s.name,
			"title":       s.title,
			"description": s.description,
			"inputSchema": withCommon(s.schema),
		}
		if len(s.annotations) > 0 {
			t["annotations"] = s.annotations
		}
		out = append(out, t)
	}
	return out
}

// ── tools/call ──────────────────────────────────────────────────────────────

func (h *hub) toolsCall(pairID, name string, args map[string]any) map[string]any {
	switch name {
	case "site_info":
		return h.siteInfo(pairID)
	case "job_result":
		return present(h.jobResult(pairID, args))
	case "job_cancel":
		return h.jobCancel(pairID, args)
	}

	var spec *toolSpec
	for i := range specs {
		if specs[i].name == name {
			spec = &specs[i]
			break
		}
	}
	if spec == nil {
		return errorResult("invalid_argument", "unknown tool: "+name)
	}

	waitMS := h.cfg.defaultWaitMS
	if spec.runWait {
		waitMS = h.cfg.runWaitMS
	}
	if spec.kind == "cell_run" || spec.kind == "notebook_run_all" {
		waitMS = h.cfg.runWaitMS
	}
	if v, ok := args["wait_ms"].(float64); ok {
		waitMS = clampWait(v, float64(waitMS), float64(h.cfg.maxWaitMS))
	}
	idemKey, _ := args["idempotency_key"].(string)

	forward := map[string]any{}
	for k, v := range args {
		if k == "idempotency_key" || k == "wait_ms" || k == "_confirmationReason" {
			continue
		}
		forward[k] = v
	}

	kind := spec.kind
	if kind == "" {
		kind = spec.name
	}
	return present(h.relayCall(pairID, kind, forward, waitMS, idemKey))
}

func (h *hub) relayCall(pairID, kind string, args map[string]any, waitMS int64, idemKey string) callOutcome {
	return h.call(pairID, kind, args, waitMS, idemKey)
}

func (h *hub) siteInfo(pairID string) map[string]any {
	online, label, lastSeen, count := h.relay.runnerStatus(pairID)
	pending := h.relay.jobs.pending(pairID)
	structured := map[string]any{
		"ok":            true,
		"runner_online": online,
		"runner":        nilIfEmpty(label),
		"runner_page":   nilIfEmpty(h.cfg.runnerURL),
		"pair_id":       pairID,
		"pending_jobs":  pending,
	}

	lines := []string{"runner_online: " + boolText(online)}
	if label != "" {
		lines = append(lines, "runner: "+label)
	}
	if h.cfg.runnerURL != "" {
		lines = append(lines, "runner_page: "+h.cfg.runnerURL)
	}
	lines = append(lines, "pair_id: "+pairID)
	if !lastSeen.IsZero() {
		lines = append(lines, "runner_last_seen_ms_ago: "+itoa(int(time.Since(lastSeen).Milliseconds())))
	}
	lines = append(lines, "runners: "+itoa(count))

	if online {
		r := h.call(pairID, "site_info", map[string]any{}, 2_500, "")
		if r.status == "ok" && r.structured != nil {
			for _, k := range []string{"default_kernel", "kernels", "notebook", "cells", "kernel", "kernel_status"} {
				if v, ok := r.structured[k]; ok {
					structured[k] = v
				}
			}
			if dk, _ := r.structured["default_kernel"].(string); dk != "" {
				lines = append(lines, "default_kernel: "+dk)
			}
			if nb, _ := r.structured["notebook"].(string); nb != "" {
				lines = append(lines, "notebook: "+nb)
			} else {
				lines = append(lines, "notebook: (none open)")
			}
		}
	} else {
		lines = append(lines, "", offlineText)
	}

	return map[string]any{
		"content":           []any{map[string]any{"type": "text", "text": strings.Join(lines, "\n")}},
		"structuredContent": structured,
		"isError":           false,
	}
}

func (h *hub) jobResult(pairID string, args map[string]any) callOutcome {
	id, _ := args["job_id"].(string)
	if id == "" {
		return callOutcome{status: "error", err: &jobError{Code: "invalid_argument", Message: "missing job_id"}}
	}
	job := h.relay.jobs.get(id)
	if job == nil {
		return callOutcome{status: "error", err: &jobError{Code: "job_unknown", Message: "unknown job_id: " + id}}
	}
	waitMS := int64(15_000)
	if v, ok := args["wait_ms"].(float64); ok {
		waitMS = clampWait(v, 15_000, jobResultCap)
	}
	return h.awaitJob(job, waitMS)
}

func (h *hub) jobCancel(pairID string, args map[string]any) map[string]any {
	id, _ := args["job_id"].(string)
	if id == "" {
		return errorResult("invalid_argument", "missing job_id")
	}
	job := h.relay.jobs.get(id)
	if job == nil {
		return errorResult("job_unknown", "unknown job_id: "+id)
	}
	job.finish(jobCancelled, nil, map[string]any{"ok": false}, &jobError{
		Code:    "cancelled",
		Message: "the job was cancelled; the cell may still be computing in the browser",
	})
	h.relay.send(pairID, cancelFrame{Type: "cancel", ID: id})
	return map[string]any{
		"content":           []any{map[string]any{"type": "text", "text": "job " + id + " cancelled"}},
		"structuredContent": map[string]any{"ok": true, "job_id": id},
	}
}

// ── result shaping (present() equivalent) ───────────────────────────────────

func present(o callOutcome) map[string]any {
	switch o.status {
	case "offline":
		return errorResult("runner_offline", offlineText, "Open the runner page and click Pair.")
	case "running":
		return map[string]any{
			"content":           []any{map[string]any{"type": "text", "text": "Still running (job_id " + o.jobID + "). Call job_result with this id; do not re-run the cell."}},
			"structuredContent": map[string]any{"ok": true, "status": "running", "job_id": o.jobID},
		}
	case "error", "cancelled":
		err := o.err
		if err == nil {
			err = &jobError{Code: "error", Message: "error"}
		}
		content := normalizeContent(o.content)
		if !hasText(content) {
			text := err.Message
			if err.Hint != "" {
				text = err.Message + "\n" + err.Hint
			}
			content = append([]any{map[string]any{"type": "text", "text": text}}, content...)
		}
		return map[string]any{
			"content":           content,
			"structuredContent": map[string]any{"ok": false, "error": err},
			"isError":           true,
		}
	}

	structured := map[string]any{"ok": true, "job_id": o.jobID}
	for k, v := range o.structured {
		structured[k] = v
	}
	return map[string]any{
		"content":           normalizeContent(o.content),
		"structuredContent": structured,
	}
}

func hasText(content []any) bool {
	for _, item := range content {
		if m, ok := item.(map[string]any); ok {
			if t, _ := m["type"].(string); t == "text" {
				if s, _ := m["text"].(string); strings.TrimSpace(s) != "" {
					return true
				}
			}
		}
	}
	return false
}

func normalizeContent(raw []any) []any {
	if raw == nil {
		return []any{map[string]any{"type": "text", "text": "(no output)"}}
	}
	out := []any{}
	budget := textBudget
	images := 0
	for _, item := range raw {
		block, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			text, _ := block["text"].(string)
			if budget <= 0 {
				continue
			}
			text = truncate(text, budget)
			budget -= len(text)
			out = append(out, map[string]any{"type": "text", "text": text})
		case "image":
			data, _ := block["data"].(string)
			mime, _ := block["mimeType"].(string)
			if images >= imageMaxN {
				continue
			}
			if len(data) > imageMaxB64 {
				out = append(out, map[string]any{"type": "text", "text": "[image omitted: too large; see the notebook cell output]"})
				continue
			}
			images++
			out = append(out, map[string]any{"type": "image", "data": data, "mimeType": mime})
		case "resource_link":
			uri, _ := block["uri"].(string)
			name, _ := block["name"].(string)
			if name == "" {
				name = uri
			}
			entry := map[string]any{"type": "resource_link", "uri": uri, "name": name}
			if desc, ok := block["description"].(string); ok {
				entry["description"] = desc
			}
			out = append(out, entry)
		}
	}
	if len(out) == 0 {
		out = append(out, map[string]any{"type": "text", "text": "(no output)"})
	}
	return out
}

func truncate(text string, max int) string {
	if len(text) <= max {
		return text
	}
	cut := text[:max]
	return cut + "\n…[truncated " + itoa(len(text)-len(cut)) + " chars; full output is in the notebook]"
}

func errorResult(code, message string, hint ...string) map[string]any {
	text := message
	if len(hint) > 0 && hint[0] != "" {
		text = message + "\n" + hint[0]
	}
	err := map[string]any{"code": code, "message": message}
	if len(hint) > 0 && hint[0] != "" {
		err["hint"] = hint[0]
	}
	return map[string]any{
		"content":           []any{map[string]any{"type": "text", "text": text}},
		"structuredContent": map[string]any{"ok": false, "error": err},
		"isError":           true,
	}
}

func clampWait(v, fallback, maxMS float64) int64 {
	if v <= 0 || v != v { // <=0 or NaN
		return int64(fallback)
	}
	if v > maxMS {
		v = maxMS
	}
	return int64(v)
}

func boolText(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

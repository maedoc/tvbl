# tvbl-mcp-hub

Cloudflare Worker + Durable Object that relays MCP tool calls from a chat client
(ChatGPT, Claude, Gemini, Codex, …) to a JupyterLite notebook running in the
user's browser.

Deployed at <https://tvbl-mcp-hub.marmaduke-woodman.workers.dev>.

```
MCP client ──HTTP──> Worker /mcp/<pair>/<token> ──> DO(pair) ──WS──> browser runner
```

- One Durable Object per pairing (`HUB.getByName(pairId)`): its own queue, runner
  socket and job records.
- Jobs are persisted in DO storage, so `cell_run` can return `{status:"running",
  job_id}` after `wait_ms` and the result survives a reloaded tab or a dropped
  request.
- The runner authenticates with the pair token in its first WebSocket frame; the
  token is compared against a SHA-256 hash stored at pair creation.

## Deploy

```sh
cd hub
npm install
set -a; . ~/.cf-tvbl.env; set +a      # CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID
npx wrangler deploy
```

`npx wrangler dev --port 8787` runs the same thing locally (SQLite DO included).

## Create a pair

```sh
curl -sX POST https://tvbl-mcp-hub.marmaduke-woodman.workers.dev/pair \
  -H 'content-type: application/json' -d '{"label":"my tab"}'
```

Returns `pair_id`, `token`, `mcp_url` (token in the path), `mcp_url_bearer`
(token via `Authorization: Bearer`), and `runner_ws`. Paste `mcp_url` into the
client's connector settings.

## Test without a browser or an MCP client

```sh
node test/fake-runner.mjs                    # mints a pair, prints mcp_url
node test/fake-runner.mjs --pair <id> --token <t> --base http://127.0.0.1:8787

# then drive it
node /tmp/flow.mjs <mcp_url>
```

The fake runner implements every tool kind, including a 3 s cell (source
containing `slow`), an error cell (`boom`) and an image cell (`image`).

## Wire contract (hub ⇄ runner)

Runner → hub:

```jsonc
{ "type": "auth", "token": "…", "label": "JupyterLite tab" }
{ "type": "result", "id": "job-uuid", "ok": true,
  "content": [ { "type": "text", "text": "…" },
               { "type": "image", "data": "<base64>", "mimeType": "image/png" } ],
  "structured": { "cell_id": "…", "duration_ms": 12 } }
```

Hub → runner:

```jsonc
{ "type": "job", "id": "job-uuid", "kind": "cell_run", "args": { "cell_id": "…" } }
{ "type": "cancel", "id": "job-uuid" }
```

`content` is MCP content; the hub truncates text to 32 KB, images to 4 × 1.2 MB
base64, and passes `structured` through as `structuredContent`.

## Tool set

`site_info` (read), `notebook_list`/`notebook_read` (read), `notebook_create`,
`notebook_open`, `cell_insert`, `cell_edit`, `cell_delete`, `cell_move`,
`cell_run`, `notebook_run_all`, `job_result` (read), `job_cancel`,
`kernel_restart`, `file_list`/`file_read` (read), `file_write`.

`cell_run` / `notebook_run_all` accept `wait_ms` (default 20 s, max 120 s) and
return a `job_id` if the cell is still running.

## v0 limits

- `/pair` is unauthenticated (it only creates an empty shard). Rate-limit or
  front it with OAuth before exposing it widely.
- The MCP token sits in the connector URL and therefore in Cloudflare request
  logs; use the bearer form once the client supports headers.
- No `notifications/progress` yet: long calls hold the SSE stream open silently
  and `job_result` is the polling path.
- One runner tab per pair; the most recently active authenticated socket wins.

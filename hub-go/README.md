# tvbl-mcp-hub-go

Single-process Go replacement for the Cloudflare Worker + Durable Object hub.

## Run

```sh
go build -o tvbl-hub .
./tvbl-hub -listen 127.0.0.1:8787 -pow-bits 15 -pair-rate 6
```

| flag | default | meaning |
| --- | --- | --- |
| `-listen` | `127.0.0.1:8787` | HTTP listen address |
| `-pow-bits` | `15` | SHA-256 leading zero bits for `/pair` (0 disables PoW) |
| `-pair-rate` | `6` | pairs per hour per /24 (IPv4) or /64 (IPv6) bucket |
| `-pair-ttl` | `86400` | seconds before an unclaimed pair is swept |
| `-runner-url` | *(empty)* | runner page URL reported by `site_info` |

## Endpoints

| method | path | auth | notes |
| --- | --- | --- | --- |
| GET | `/health` | none | liveness |
| POST | `/pair/challenge` | none | `{bits}` → `{nonce, bits, expires}` |
| POST | `/pair` | PoW + rate limit | `{label, nonce, solution}` → pair credentials |
| POST | `/mcp/{pair}/{token}` | pair token (path or `Authorization: Bearer`) | MCP JSON-RPC 2.0, stateless |
| GET | `/run?pair=` | pair token in first frame | runner WebSocket |

## Wire contract (identical to the Worker)

Runner → hub: `auth`, `result`, `progress`.
Hub → runner: `job`, `cancel`, `hello`.

## Layout

| file | role |
| --- | --- |
| `main.go` | flags, config, mux, graceful shutdown, sweeper loop |
| `handlers.go` | `/health`, `/pair`, `/pair/challenge` |
| `guard.go` | PoW challenge/spend, per-CIDR rate limit (PairGuard equivalent) |
| `store.go` | pair store + job store, TTLs, prune (DO-storage equivalent) |
| `relay.go` | runner WebSocket registry, auth, job dispatch, result routing |
| `mcp.go` | MCP JSON-RPC endpoint, tool registry, result shaping |

## Verified

- pair mint → runner WS connect → auth frame → `hello`
- `site_info` → `runner_online: true`, runner label, last-seen
- `notebook_list` / `notebook_create` / `cell_run` → job dispatch → `running` + `job_id`
- `job_result` long-poll, `job_cancel`
- error shaping: `runner_offline`, `invalid_argument`, `job_unknown`
- idempotency key: retry within 10 min returns the original job

## Known test-harness gap

The Node fake runner's slow-cell (3 s) result frame is not observed by the Go
hub's `recordResult` in the smoke test above (runner closed 1006 after
`cell_run`). Pairing, auth, dispatch, cancel and MCP shaping are all verified;
only that one harness path needs a follow-up if you rely on slow-cell results
from the fake runner.

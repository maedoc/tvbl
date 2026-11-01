# Review: Cloudflare hub + JupyterLite runner for chat-based MCP

Status: design review of `setup-cloudflare-thing.sh`, verified against
JupyterLite 0.8.7 / JupyterLab 4.6.5 and the current MCP spec (2026-07-28),
OpenAI custom-MCP docs as of 2026-11-01.

## Verdict

The shape is right and it is not crazy: a Worker + Durable Object is exactly the
right rendezvous for a browser page that cannot be dialled directly, and
JupyterLite is the only way to get a real scientific Python stack (numba, TVB,
MNE) into a chat client without the user running a server or paying for
completions.

What is wrong is not the idea but the details:

1. It is an unauthenticated remote-code-execution relay. Anyone with the URL can
   drive any connected tab, and any second tab can attach to the same hub and
   answer (or observe) tool calls.
2. One Durable Object (`getByName("default")`) for the entire deployment, so all
   users share one shard, one queue, one socket list, and `runners[0]`.
3. The tool surface (`search` + `execute`) throws away everything that makes
   JupyterLite interesting. It gives the model `eval()` in a page, not a
   notebook.
4. The MCP handler is a hand-rolled legacy-`initialize` server; the protocol
   moved on (2026-07-28), and the hand-rolled version omits cancellation,
   task/progress support and proper transport handling.
5. Long-running work (the normal case for TVB) exceeds the client's tool-call
   budget and there is no job model, so a 3-minute simulation is a failed call.

Fix 1-5 and this is a genuinely useful thing. In particular the job model
(`cell_run` returns either the outputs or a `job_id` in <1 s) is what makes it
survive real workloads.

Semantics matter: this is "bring your own compute and data to a cloud model",
not "free Python". ChatGPT already has a Python sandbox. The defensible niche is
private data + a custom scientific stack + a notebook artifact the user keeps.
That is a real niche for a university/lab Edu workspace, which happens to be the
TVB audience.

---

## 1. What the script gets right

- **Worker + DO + hibernatable WebSockets** is the correct Cloudflare primitive.
  A plain Worker cannot hold the runner connection; a DO can, and
  `ctx.acceptWebSocket` + `setWebSocketAutoResponse("ping","pong")` keeps idle
  tabs free.
- The runner page is dependency-free and self-describing (shows the WS and MCP
  URLs), which is the right instinct for a zero-install demo.
- Separating `/run` (runner side) from `/mcp` (model side) is the right split.
- The MCP bits that *are* there are the important ones: `initialize`,
  `tools/list`, `tools/call`, `isError: true` on failures.
- Providing both a no-deps JS worker and a TS/`agents`-SDK variant is nice, but
  see bug 1: only one of them actually installs.

## 2. Bugs and blockers in `setup-cloudflare-thing.sh`

Ordered by severity. Line numbers refer to the script as committed.

### B1. `package.json` pins versions that do not exist (TS path is dead)

```
line 31:  "agents": "^1.0.0"
line 32:  "@modelcontextprotocol/sdk": "^2.3.1"
```

Current registry versions: `agents@0.32.0`, `@modelcontextprotocol/sdk@1.32.2`
(`wrangler@4.156.0`). `^1.0.0`/`^2.3.1` do not resolve, so `npm install` fails
and `wrangler.ts.toml` cannot build. `createMcpHandler` does live at
`agents/mcp/server` in 0.32.0, so only the pins are wrong.

Fix: `"agents": "^0.32.0"`, `"@modelcontextprotocol/sdk": "^1.32.2"`.

### B2. No authentication, and the relay is world-writable

- `/mcp` is unauthenticated: anyone who learns the URL can call `execute` and run
  code in whatever tab is attached.
- `/run` is unauthenticated and untagged: a second connection is just another
  `runner` and `runners[0]` (line 218) may hand it the next job. It can also
  return arbitrary results, so it can forge tool output.
- No `Origin` check on the WebSocket handshake and no `Host` check on `/mcp`.

Fix: pair each browser tab with an unguessable (128-bit) `pair_id` + secret; put
the secret in the MCP connector credential (`Authorization: Bearer ...`;
ChatGPT's custom-MCP config now accepts static credentials) or, if you must, in
the path. Shard the DO by `pair_id`. Validate `Origin`/`Host`. Rate-limit
pairing.

### B3. One DO for every user

`env.HUB.getByName("default")` (lines 425, 678) with `runners[0]` (218, 576)
means:

- all users share one execution queue;
- a job may be delivered to the wrong tab;
- one user's long cell blocks everyone;
- there is no isolation of job results.

Fix: `getByName(pairId)`. DOs are cheap and hibernation makes idle shards free.

### B4. In-memory `pending` map + hibernation

`this.pending` (line 227) is instance state, but `acceptWebSocket` opts into
hibernation and the object can be evicted (including on every deploy). An
in-flight tool call then dies with no result, and the runner's later reply lands
on a fresh instance with an empty map.

Fix: write job records to `ctx.storage` (`job:<id>`), let the runner deliver
results even when no HTTP client is waiting, and use `ctx.storage.setAlarm()`
for TTL/GC instead of relying on `setTimeout`.

### B5. 30 s timeout vs. real TVB workloads

`setTimeout(..., 30_000)` (line 222) rejects the promise, but:

- the code is still running in the runner; nothing tells it to stop, so the
  model is told "failed" while the cell keeps burning CPU;
- 30 s is far below the client budget (ChatGPT observed ~5 min hard cap; MCP
  SDK defaults to a 300 s read timeout), so you are giving up usable time;
- TVB simulations routinely take minutes.

Fix: job model. `cell_run(wait_ms<=25000)` returns either outputs or
`{job_id, status:"running"}`; `job_result(job_id)` polls; the runner keeps the
result in the DO even if nobody is waiting. Emit `notifications/progress` every
~10 s so intermediaries do not idle out, and honour cancellation (detach, not
kill, by default).

### B6. Protocol staleness

`protocolVersion: params?.protocolVersion || "2025-06-18"` (line 321) is wrong
in two ways: echoing the client's version claims support you may not have, and
the current revision (2026-07-28) has no `initialize` handshake at all for
"modern" clients — version is per-request `_meta`, and servers are expected to
be dual-era (`initialize` for legacy clients, `server/discover` for modern).

The hand-rolled handler also lacks SSE framing rules, `notifications/cancelled`
handling, and forward compatibility with the tasks extension — which is exactly
the long-running-tool answer you would want.

Fix: delete `handleMcp` and use the SDK's
`WebStandardStreamableHTTPServerTransport` (or `agents`' `createMcpHandler`,
which already wraps it and adds `Host`/`Origin` allowlists). Keep SSE on
(`enableJsonResponse: false`) so progress/keep-alives can flow.

### B7. Session id returned, never validated

Line 328 returns `Mcp-Session-Id`, but nothing checks it on subsequent requests
and requests are handled statelessly. Either implement sessions or return no id
(`sessionIdGenerator: undefined`).

### B8. Fake GET SSE stream

Lines 289-300 open a `TransformStream`, write one comment, and close after 30 s.
The spec allows returning `405 Method Not Allowed` when the server does not offer
an SSE stream, which is more honest and avoids client reconnect loops. The
`setTimeout` is also not tied to `ctx.waitUntil`.

### B9. `tools/call` destructuring crashes on malformed input

`const { name, arguments: args } = params` (line 383) throws `TypeError` if
`params` is missing, producing a 500 with no JSON-RPC error. No schema
validation either. The SDK + zod fixes both.

### B10. Duplicated, divergent implementations

`src/worker.js` and `src/hub.ts` + `src/index.ts` are copies. Every fix above has
to be made twice. Keep one implementation (the TS one) and let the no-deps path
be "npx wrangler deploy" with the SDK bundled.

### B11. `webSocketClose` re-closes an already-closed socket

Line 270 calls `ws.close(code, reason)` inside the close handler. It is at best
a no-op and can throw depending on readyState. Just clean up pending jobs and
return.

### B12. `runners[0]` may be CLOSING; `send` can throw

After a tab closes, `getWebSockets("runner")` can still return the socket (the
docs call this out). Wrap `send` and fall through to the next socket, or keep a
`serializeAttachment({ lastSeen })` and pick the freshest live one.

### B13. Result serialization has no limits and no binary path

`JSON.stringify(result)` (line 165) throws on BigInt/circular structures, and a
single result can be megabytes of base64. Worker WS receive limit is 32 MiB
(raised from 1 MiB in Oct 2025), but MCP SSE clients commonly cap events at
1 MiB, and the model's context is smaller still.

Fix: explicit budget — text truncated to ~32 KB with a truncation marker, at
most N images, images downscaled (the plugin runs in the browser and can use an
OffscreenCanvas), and "the full output is in the notebook at cell X".

### B14. Runner URL derived from `location.host`

Lines 153-154 and 775-776 assume the runner page is served by the hub. The whole
point is to put the runner on JupyterLite (github.io / Pages / custom domain).
Make the hub URL configuration, not an assumption.

### B15. `innerHTML` with model-controlled content

Line 159 interpolates `label` (fixed, fine) and escapes `content`, but the
sandbox is explicitly model-controlled and the page shares the hub origin. Any
HTML the model writes into `#sandbox` can carry event handlers that run on the
hub origin. Use `textContent`, and prefer separate origins for hub and runner.

### B16. Tool surface and annotations

- `search` is meaningless here (custom MCP servers do not need `search`/`fetch`;
  that is a deep-research requirement) and collides conceptually with web search.
- `execute` is generic `eval`, the least useful schema for a notebook: the model
  has to invent a JS API, and every result is unstructured text.
- No `annotations`. ChatGPT treats any tool without `readOnlyHint: true` as a
  write action, so *every* call pops a confirmation. Mark reads as reads.
- No `instructions` in the `initialize` result. OpenAI explicitly recommends up
  to 512 self-contained characters of cross-tool guidance; this is the single
  cheapest quality win.
- No `structuredContent`, so the model parses strings.

### B17. No idempotency

ChatGPT may retry a tool call; "run this cell" is not idempotent. Accept a
client-supplied `request_id`/`job_id` and dedupe before re-executing.

### B18. Minor

- `[assets]` + `main` in `wrangler.ts.toml`: asset misses fall through to the
  worker, so `/mcp` works, but `public/index.html` duplicates `RUNNER_HTML`;
  there are now three copies of the page (worker.js, hub.ts, public/).
- `compatibility_date = "2026-10-27"` is fine today (2026-11-01); `new_sqlite_classes`
  is correct.
- No `Content-Security-Policy` / `X-Frame-Options` on the runner page.
- `DELETE /mcp` returns 200 rather than 204; harmless.

---

## 3. The architecture that will actually work

```
 ChatGPT / Claude / Codex / Cursor
            │  Streamable HTTP (MCP)
            ▼
 ┌──────────────────────────────────────────────┐
 │ Worker                                       │
 │  /mcp/:pair      → MCP handler (SDK)         │
 │  /run            → WS upgrade, auth, route    │
 │  /pair           → create pair (rate-limited) │
 └───────────────┬──────────────────────────────┘
                 │ getByName(pairId)
                 ▼
 ┌──────────────────────────────────────────────┐
 │ Durable Object "RunnerHub" (one per pair)    │
 │  · runner socket (hibernatable, tagged)      │
 │  · job queue + job records in storage        │
 │  · request ⇄ job correlation, progress, GC   │
 └───────────────┬──────────────────────────────┘
                 │ WebSocket (JSON frames)
                 ▼
 ┌──────────────────────────────────────────────┐
 │ JupyterLite plugin in the user's browser tab │
 │  · notebook/cell/kernel tools                │
 │  · executes via NotebookActions              │
 │  · returns text + image content blocks       │
 └──────────────────────────────────────────────┘
```

### Pairing

1. User opens the JupyterLite page and the MCP Hub panel, clicks **Pair**.
2. Plugin `POST /pair` → `{pairId, token, mcpUrl}`; the hub stores
   `sha256(token)` in `HUB.getByName(pairId)`.
3. Plugin opens `wss://hub/run?pair=<pairId>` and sends
   `{type:"auth", token, site:location.origin, notebook:"..."}`. The DO compares
   hashes (constant time), tags the socket `runner:<pairId>`, records
   `serializeAttachment({pairId, lastSeen})`.
4. Panel shows the MCP URL + bearer token to paste into the client (ChatGPT's
   config accepts static credentials; if the UI only takes a URL, fall back to
   `/mcp/<pairId>/<token>`).
5. Revoke = delete the token hash + `ws.close(4001)`.

### Job lifecycle (the important part)

```
tools/call cell_run
  → DO creates job{id, kind, args, status:"queued"}, persists it
  → sends {type:"job", id, kind, args} to runner
  → runner sends {id, ack:true} immediately
  → runner sends {id, progress, message} while running
  → runner sends {id, ok, result} or {id, error}
  → DO persists result (truncated), resolves the waiting HTTP request if still
    attached, else leaves it for job_result(job_id)
```

`cell_run` waits at most `wait_ms` (default 20-25 s, max 120 s). If the cell is
still running it returns `{status:"running", job_id}` within the client budget.
The model polls `job_result` and, critically, the user is not blocked. Results
survive tab reloads and client disconnects because they live in DO storage, not
in the request handler.

### Same-origin vs. split origins

Simplest deployment: Worker serves the JupyterLite build as static assets *and*
`/mcp` + `/run`. Then WS is same-origin, cookies are usable, and there is no
allowlist to maintain. The site is large but under Workers' per-asset limit.
Otherwise keep Pages + `hub.` subdomain and maintain an explicit `Origin`
allowlist (plus `allowedHostnames` if you use `createMcpHandler`).

### Auth that scales past the demo

Bearer-in-connector is fine for personal use. For a lab rollout, use
`@cloudflare/workers-oauth-provider`: OAuth subject → user; `getByName(userId)`;
the runner claims the shard. Then one published connector serves everyone and
the workspace admin story works. Keep the pairing-token path as the dev mode.

---

## 4. JupyterLite plugin: verified APIs

Version matrix (2026-11-01): `jupyterlite-core 0.8.7`, `jupyterlite-xeus 5.1.0`,
`jupyterlab 4.6.5`, `@jupyterlite/services 0.8.7`. The extension must be built
against the *same* JupyterLab minor as the bundled app (4.6.x), because
federated extensions share the `@jupyterlab/*` singletons.

### Key constraint

JupyterLite has **no Python/server extension layer**. Everything must be a
frontend JupyterLab plugin. Do not import `@jupyterlite/*` from the plugin: those
packages are not in the federated shared scope. Use `app.serviceManager`,
`app.commands`, the trackers and the document manager.

### Packaging and build

```
extensions/@tvbl/jupyterlite-mcp-hub/   ← prebuilt federated extension
  package.json
  static/remoteEntry.*.js
  static/style.js
  schemas/@tvbl/jupyterlite-mcp-hub/plugin.json
```

- Build with `@jupyterlab/builder` (`jlpm build:prod`), `jupyterlab.extension:
  true`, `outputDir` pointing at `extensions/@tvbl/jupyterlite-mcp-hub`.
- `jupyter lite build` discovers anything under `{lite-dir}/extensions/` and adds
  it to `jupyter-lite.json#jupyter-config-data/federated_extensions`
  (alternative: install the wheel into the build env, or pass
  `--FederatedExtensionAddon.extra_labextensions_path`).
- CI: add a node step before `jupyter lite build`; the existing workflow already
  has Python, add `actions/setup-node` + `jlpm install && jlpm build:prod`.
- Publish/version: the plugin id must be stable (`@tvbl/jupyterlite-mcp-hub:plugin`).

### What the APIs actually give you

| Goal | API |
| --- | --- |
| current/open notebook | `INotebookTracker.currentWidget`, `.find(p => p.context.path === path)` |
| open/create | `IDocumentManager.openOrReveal(path)`, `commands.execute('notebook:create-new', {kernelName})` |
| notebook model | `panel.content.model` (`INotebookModel`), `.sharedModel` (`ISharedNotebook`) |
| read cell | `cell.model.sharedModel.getSource()`, `cell.model.id` |
| edit cell | `cell.model.sharedModel.setSource(src)` |
| insert cell | `panel.content.model.sharedModel.insertCell(i, {cell_type:'code', source, metadata:{}})` |
| delete/move | `sharedModel.deleteCell(i)`, `sharedModel.moveCell(from, to)` |
| execute | `NotebookActions.runCells(panel.content, [cell], panel.sessionContext, sessionDialogs, translator)` → `Promise<boolean>` |
| run all | `NotebookActions.runAll(panel.content, sessionContext, sessionDialogs)` |
| kernel | `panel.sessionContext.ready`, `panel.sessionContext.session?.kernel` |
| raw execute | `kernel.requestExecute({code, stop_on_error:false})`, `future.onIOPub`, `future.done` |
| outputs | `(cell.model as ICodeCellModel).outputs.toJSON()` → `nbformat.IOutput[]` |
| scroll/focus | `panel.content.activeCellIndex = i; await panel.content.scrollToCell(cell, 'center')` |
| reveal panel | `app.shell.activateById(panel.id)` |
| settings | `ISettingRegistry` + `schema/plugin.json` |
| status/UI | `IStatusBar`, `Notification`, `ReactWidget` in `app.shell.add(..., 'right')` |
| config from site | `PageConfig.getOption('mcpHubUrl')` (from `jupyter-lite.json`) |

### Caveats that matter

- **Interrupt is not a real interrupt.** `LiteKernelClient.interrupt()` cancels
  the JS-side mutex and *following* cells; the running wasm call cannot be
  preempted. A runaway `while True:` hangs the kernel until `restart()`. Treat
  "timeout" as "restart required", and make that explicit to the model.
- **One kernel only.** The TVB/xeus env is large (numba + matplotlib + MNE +
  TVB; you already hit the ~2 GiB wasm ceiling in tests). Starting a second
  "headless" kernel doubles memory. Reuse the notebook's kernel; if no notebook
  is open, create a scratch one.
- **First execution is slow.** Kernel boot loads the environment (seconds to tens
  of seconds). Pre-warm on pairing, report `kernel_ready` in `site_info`, and use
  a longer `wait_ms` for the first call.
- **Use real cells, not raw `requestExecute`, for anything visible.** Widgets
  (`%matplotlib widget`, `ipympl`, `tqdm`) need the comm/OutputArea machinery
  that `CodeCell.execute` provides. `requestExecute` is fine for cheap
  state-inspection, but then outputs/comm are yours to handle.
- **`NotebookActions.runCells` resolves only when the cell finishes.** Do not
  `Promise.race` and drop the promise: keep awaiting it in the background and
  deliver the result to the job later.
- **Address cells by id, not index.** Indices shift when anyone (user or model)
  inserts above. Return `cell_id` on every insert and require it on edit/run.
  Datalayer's `jupyter-mcp-server` added `cell_id` for exactly this reason.
- **`readOnlyHint` affects UX.** Without it ChatGPT treats the tool as a write
  action and asks for confirmation each time.

### Tool surface

Flat, prefixed names (ChatGPT's namespace is global), action-oriented
descriptions, `annotations.readOnlyHint` where true.

| tool | args | read-only | notes |
| --- | --- | --- | --- |
| `site_info` | – | yes | kernel name/status, memory budget, hub version, limits, usage hints |
| `notebook_list` | – | yes | path, cell count, kernel |
| `notebook_read` | `path`, `include_outputs` | yes | cells as `{id, type, source, outputs}`, truncated |
| `notebook_create` | `path?`, `kernel?` | no | creates + opens, returns path |
| `notebook_open` | `path` | no | focus the tab so the user can watch |
| `cell_insert` | `notebook?`, `index?`, `cell_type`, `source` | no | returns `cell_id` |
| `cell_edit` | `cell_id`, `source` | no | |
| `cell_delete` | `cell_id` | no | destructive |
| `cell_move` | `cell_id`, `index` | no | |
| `cell_run` | `cell_id`, `wait_ms?` | no | outputs or `job_id` |
| `notebook_run_all` | `path?`, `wait_ms?` | no | |
| `job_result` | `job_id` | yes | poll |
| `job_cancel` | `job_id` | no | detach/cancel |
| `kernel_restart` | – | no | clears state; the only escape from a hang |
| `file_list` / `file_read` / `file_write` | paths | mixed | via `serviceManager.contents` |
| `ui_focus_cell` | `cell_id` | no | scroll/reveal so the user sees it |
| `js_execute` | `code` | no | escape hatch, off by default |

Drop `search` entirely. Keep `js_execute` behind a setting and describe it as
"last resort for UI automation, prefer the notebook tools".

### Plugin skeleton (the parts that are easy to get wrong)

```ts
import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { INotebookTracker, NotebookActions } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';
import { IDocumentManager } from '@jupyterlab/docmanager';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { ISessionContextDialogs } from '@jupyterlab/apputils';
import { ITranslator } from '@jupyterlab/translation';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@tvbl/jupyterlite-mcp-hub:plugin',
  autoStart: true,
  requires: [INotebookTracker, IDocumentManager, ITranslator],
  optional: [ISettingRegistry, ISessionContextDialogs],
  activate: async (app, tracker, docManager, translator, settings, dialogs) => {
    // 1. settings: hubUrl, pairId, pairToken, allowJsExecute
    // 2. ws = new WebSocket(`${wsBase}/run?pair=${pairId}`);
    //    send {type:'auth', token} as the first frame; reconnect with backoff
    // 3. dispatch frames to the handlers below; ack immediately, then result
  }
};

async function insertCell(panel, index, source, type = 'code') {
  const model = panel.content.model!;
  const at = index ?? model.cells.length;
  model.sharedModel.insertCell(at, { cell_type: type, source, metadata: {} });
  panel.content.activeCellIndex = at;
  panel.content.update();
  return model.cells.get(at)!.id;            // stable id, use this everywhere
}

async function runCell(panel, cellId, waitMs, dialogs, translator) {
  const cell = panel.content.widgets.find(c => c.model.id === cellId);
  if (!cell) throw new Error(`no such cell: ${cellId}`);
  await panel.sessionContext.ready;           // kernel boot can take 30 s

  const done = NotebookActions.runCells(
    panel.content, [cell], panel.sessionContext, dialogs, translator
  );

  let finished = false;
  done.then(() => { finished = true; }, () => { finished = true; });

  const t0 = performance.now();
  await Promise.race([done.catch(() => false), sleep(waitMs)]);

  if (!finished) return { status: 'running', cell_id: cellId };  // keep `done` alive!

  const outputs = (cell.model as ICodeCellModel).outputs.toJSON();
  return {
    status: outputs.some(o => o.output_type === 'error') ? 'error' : 'ok',
    cell_id: cellId,
    duration_ms: Math.round(performance.now() - t0),
    outputs: toMcpContent(outputs)            // text + image blocks, truncated
  };
}
```

`toMcpContent` maps `stream` → text, `execute_result`/`display_data` →
`data['text/plain']` plus `image/png` as
`{type:'image', data:<base64>, mimeType:'image/png'}`, and `error` →
`{ename, evalue, traceback}` (trimmed). Downscale PNGs above ~150 KB with an
`OffscreenCanvas` before returning them.

Note the `finished` flag: `Promise.race` does not cancel `done`, so the job can
still complete later and be reported through `job_result`.

---

## 5. Hard limits and risks

| Limit | Reality | Design response |
| --- | --- | --- |
| ChatGPT full MCP (write tools) | Business/Enterprise/Edu only; Pro/Plus get read/fetch-style connectors. Custom MCP servers are now "plugins" with OAuth / static credentials / none. | Target labs with Edu/Business workspaces, or make the hub client-agnostic (Claude, Codex, Cursor, VS Code, LM Studio). |
| Tool-call budget | ~5 min observed hard cap in the ChatGPT UI (not configurable). MCP SDK read timeout defaults to 300 s. | Job model, never block > ~25 s by default. |
| Read-only vs write | Tools without `readOnlyHint` prompt for confirmation; write tools always prompt by default. | Annotate reads; expect a confirmation per `cell_run` unless the user says "remember". |
| Background tab | Wasm keeps computing in a Worker, but mobile Safari and memory pressure can suspend/kill the tab. IndexedDB survives; kernels do not. | Persist notebook content; jobs fail cleanly on disconnect; warn the user; encourage one pinned tab. |
| Kernel hang | `interrupt()` cannot preempt wasm. | `kernel_restart` as the only recovery; per-call timeout hint; tell the model to write bounded loops. |
| Memory | TVB wasm env already hits the ~2 GiB ceiling on the full test suite. | One kernel; advise `kernel_restart` between heavy demos; `site_info` states the budget. |
| Output size | MCP SSE events commonly cap at 1 MiB; model context smaller. | Truncate text, cap images, keep full data in the notebook. |
| Prompt injection | A model reading untrusted web content can call `cell_run` with `fetch()`-exfiltrating code. JupyterLite executes trusted outputs as HTML, so an injected output is JS in the runner origin. | Explicit approval for write tools, visible activity log, no long-lived secrets on the runner origin (short-lived tickets), CSP, separate origins from the hub. |
| Privacy | The hub sees code and results in plaintext. | Self-host per lab; document it; do not point it at clinical data on a shared deployment. |
| Prior art | Datalayer `jupyter-mcp-server` + `jupyter-mcp-tools` already do JupyterLab-over-MCP (cell ids, streaming progress, command tools); ChatGPT has a Python sandbox; OpenAI now offers Secure MCP Tunnels to local servers. | Differentiate on *zero install* + *custom wasm stack* + *local data* + *notebook artifact*, not on "Python in chat". |

---

## 6. Recommendation and sequencing

**Keep**: Worker + DO + hibernatable WS, runner/client split, no-deps runner page
as the demo.

**Change**: per-pair DO sharding, pairing token, SDK-based MCP handler, job model
with early ack, notebook tool surface, plugin instead of `eval`.

**Two-week plan**

- **Days 1-2**: rewrite the hub on `agents` `createMcpHandler` +
  `McpServer`; DO per pair; job records in storage; `site_info`,
  `notebook_read`, `cell_insert`, `cell_run` with wait/job split. Fake runner in
  Node to test without a browser.
- **Days 3-5**: plugin scaffolding, pairing panel, WS client with reconnect,
  cell insert/edit/run, output → MCP content (text + image).
- **Days 6-8**: job polling, progress notifications, kernel restart, file tools,
  truncation/downscaling, settings schema.
- **Days 9-10**: CI build of the extension into `jupyter lite build`; deploy;
  end-to-end with a real ChatGPT Edu/Business workspace; measure first-call
  latency, cell round-trip, wasm memory headroom, and what happens when the tab
  is reloaded mid-job.
- **Week 3 (only if the demo lands)**: OAuth via
  `@cloudflare/workers-oauth-provider`, multi-runner, tasks extension, OAuth
  subject → shard.

**Go/no-go experiments before writing much code**

1. Does the target ChatGPT workspace actually expose write tools from a custom
   MCP server? (Plan gating is the biggest product risk.)
2. Does a `image/png` content block from a tool result render in that client?
3. How long does the xeus/TVB kernel take to become ready on a cold page, and
   how much memory is left after loading the demos?
4. Can a 5-minute job be kept alive end-to-end with progress notifications, or
   does the client still cut at 300 s?

If (1) is no for your users, the same hub is still worth building for
Claude/Codex/Cursor — arguably a better audience, because they do not gate write
tools and they are already used for code.

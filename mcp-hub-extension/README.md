# @tvbl/jupyterlite-mcp-hub

JupyterLite side of the MCP hub: a federated JupyterLab extension that registers
this browser tab as the compute peer for a chat client.

```
ChatGPT / Claude / Gemini ──HTTPS MCP──> Worker + DO ──WebSocket──> this extension
                                                                     └─ NotebookActions
```

The hub is in [`../hub`](../hub). The frame contract and tool list are documented
there.

## What it does

- **Pair**: solves the hub's proof-of-work challenge (a few hundred ms, skipped
  against old hubs), `POST /pair` mints a pair id + token, saves them in the
  JupyterLab settings registry, opens `wss://hub/run?pair=…` and authenticates
  with the token as the first frame.
- **Serve**: every `job` frame is executed against real JupyterLab APIs —
  `NotebookActions.runCells` for execution (so widget comms and output areas
  behave exactly as if the user pressed Shift+Enter), the shared notebook model
  for edits, and the contents API for files.
- **Report**: outputs become MCP content blocks (text, `image/png` downscaled
  above the configured budget) plus `structuredContent` with cell ids and
  timings. A long cell just keeps running; the hub returns `job_id` and the
  client polls `job_result`.
- **Address cells by id**, never by index (`cell_insert` returns `cell_id`).

## Build

```sh
# the site's JupyterLab must provide jlpm; any env with `jupyterlab` works
jlpm install --immutable
jlpm build          # tsc + jupyter labextension build
```

Output goes to `../extensions/@tvbl/jupyterlite-mcp-hub` (gitignored), and
`jupyter_lite_config.json` points `jupyter lite build` at `extensions/`:

```json
{ "FederatedExtensionAddon": { "extra_labextensions_path": ["extensions"] } }
```

(The `{lite-dir}/extensions` auto-discovery is not registered as a build phase
in jupyterlite-core 0.8.7, hence the explicit path.)

CI does this before `jupyter lite build`; see `.github/workflows/deploy.yml`.

## Settings

Settings → MCP Hub:

| setting | default | |
| --- | --- | --- |
| `hubUrl` | `https://tvbl-mcp-hub.marmaduke-woodman.workers.dev` | Worker base URL (a `jupyter-config-data/mcpHubUrl` value is used when the setting has no user override, e.g. for local `wrangler dev`) |
| `pairId` / `pairToken` | — | written by the Pair button |
| `autoConnect` | `true` | reconnect when the page loads |
| `maxImageBytes` | `200000` | PNGs above this are downscaled via OffscreenCanvas |

The MCP URL to paste into the client is shown in the panel; the token is part of
the path (`/mcp/<pair>/<token>`). Pairing again replaces the credentials; the old
Durable Object is left empty.

## Test checklist

1. Open the site (`https://maedoc.github.io/tvbl/lab/index.html`).
2. The **MCP Hub** panel opens on the right; click **Pair this tab**.
3. Copy the URL, add it as a custom MCP connector with no authentication.
4. Ask the model to `site_info`, then create a notebook and run a cell.
5. Close the tab and call any tool: the hub reports `runner_offline` and does not
   queue work.

Known limits: `interrupt` cannot preempt wasm, so a runaway cell needs
`kernel_restart`; a closed tab means no compute until it is reopened; the
`@jupyter-widgets` manager can log a `null (reading 'kernel')` page error while a
kernel starts (harmless); Cloudflare's bot protection blocks the
`Python-urllib/x.y` user agent, so use curl/httpx for scripted checks.

#!/usr/bin/env node
/**
 * Fake runner: stands in for the JupyterLite plugin so the hub can be tested
 * from a terminal (no browser, no MCP client).
 *
 *   node test/fake-runner.mjs                       # mint a pair, print mcp_url
 *   node test/fake-runner.mjs --pair <id> --token <t>
 *
 * Env: HUB_BASE (default http://127.0.0.1:8787), PAIR_ID, PAIR_TOKEN
 */

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const base = (flag("base", process.env.HUB_BASE ?? "http://127.0.0.1:8787")).replace(/\/$/, "");
let pairId = flag("pair", process.env.PAIR_ID);
let token = flag("token", process.env.PAIR_TOKEN);

if (!pairId || !token) {
  const res = await fetch(`${base}/pair`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ label: "fake-runner" }),
  });
  if (!res.ok) {
    console.error("pair failed", res.status, await res.text());
    process.exit(1);
  }
  const pair = await res.json();
  pairId = pair.pair_id;
  token = pair.token;
  console.log(`# pair_id   ${pair.pair_id}`);
  console.log(`# mcp_url   ${pair.mcp_url}`);
  console.log(`# runner_ws ${pair.runner_ws}`);
}

const state = {
  notebook: "fake.ipynb",
  kernel: "fake-python",
  cells: [],
  files: new Map([["data/readme.txt", "hello from the fake runner"]]),
};
let seq = 0;

const wsUrl = `${base.replace(/^http/, "ws")}/run?pair=${pairId}`;
const ws = new WebSocket(wsUrl);

ws.addEventListener("open", () => {
  console.log(`# connected ${wsUrl}`);
  ws.send(JSON.stringify({ type: "auth", token, label: "fake-runner" }));
});

ws.addEventListener("close", ev => {
  console.log(`# closed ${ev.code} ${ev.reason}`);
  process.exit(0);
});

ws.addEventListener("error", ev => console.error("# ws error", ev.message ?? ev));

ws.addEventListener("message", event => {
  let frame;
  try {
    frame = JSON.parse(String(event.data));
  } catch {
    return;
  }
  if (frame.type === "hello") {
    console.log("# authenticated");
    return;
  }
  if (frame.type === "job") {
    console.log(`> ${frame.kind}`);
    handle(frame).catch(err => {
      send(frame.id, false, err.message);
    });
  }
});

async function handle(frame) {
  const a = frame.args ?? {};
  switch (frame.kind) {
    case "site_info":
      return send(frame.id, true, {
        summary: `kernel ${state.kernel}, notebook ${state.notebook}, ${state.cells.length} cells`,
        structured: { kernel: state.kernel, notebook: state.notebook, cells: state.cells.length },
      });

    case "notebook_list":
      return send(frame.id, true, {
        summary: `1 notebook: ${state.notebook}`,
        structured: { notebooks: [{ path: state.notebook, cells: state.cells.length }] },
      });

    case "notebook_read":
      return send(frame.id, true, {
        summary: `${state.cells.length} cells in ${state.notebook}`,
        structured: {
          path: state.notebook,
          cells: state.cells.map(c => ({ cell_id: c.id, type: c.type, source: c.source, outputs: c.outputs ?? [] })),
        },
      });

    case "notebook_create":
      state.notebook = a.path ?? `notebook-${Date.now()}.ipynb`;
      state.cells = [];
      return send(frame.id, true, { summary: `created ${state.notebook}`, structured: { path: state.notebook } });

    case "notebook_open":
      if (a.path) state.notebook = a.path;
      return send(frame.id, true, { summary: `opened ${state.notebook}` });

    case "cell_insert": {
      const cell = {
        id: `cell-${++seq}`,
        type: a.cell_type ?? "code",
        source: a.source ?? "",
        outputs: [],
      };
      const at = typeof a.index === "number" ? Math.min(a.index, state.cells.length) : state.cells.length;
      state.cells.splice(at, 0, cell);
      return send(frame.id, true, {
        summary: `inserted ${cell.id} at ${at}`,
        structured: { cell_id: cell.id, index: at },
      });
    }

    case "cell_edit": {
      const cell = find(a.cell_id);
      cell.source = a.source ?? "";
      return send(frame.id, true, { summary: `edited ${cell.id}`, structured: { cell_id: cell.id } });
    }

    case "cell_delete": {
      const i = state.cells.findIndex(c => c.id === a.cell_id);
      if (i < 0) throw new Error(`no such cell: ${a.cell_id}`);
      state.cells.splice(i, 1);
      return send(frame.id, true, { summary: `deleted ${a.cell_id}` });
    }

    case "cell_move": {
      const cell = find(a.cell_id);
      const i = state.cells.indexOf(cell);
      state.cells.splice(i, 1);
      state.cells.splice(Math.min(a.index ?? 0, state.cells.length), 0, cell);
      return send(frame.id, true, { summary: `moved ${cell.id} to ${a.index}` });
    }

    case "cell_run": {
      const cell = find(a.cell_id);
      if (cell.source.includes("slow")) await sleep(3_000);
      if (cell.source.includes("boom")) {
        cell.outputs = [{ output_type: "error", ename: "ValueError", evalue: "boom" }];
        return send(frame.id, false, "ValueError: boom", {
          outputs: cell.outputs,
          structured: { cell_id: cell.id, duration_ms: 1, outputs: cell.outputs },
        });
      }
      const outputs = [{ output_type: "stream", name: "stdout", text: `ran ${cell.id}: ${cell.source}` }];
      if (cell.source.includes("image")) {
        outputs.push({
          output_type: "display_data",
          data: {
            "image/png":
              "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            "text/plain": "<Figure size 1x1>",
          },
        });
      }
      cell.outputs = outputs;
      const content = outputs.map(o =>
        o.output_type === "stream"
          ? { type: "text", text: o.text }
          : o.data?.["image/png"]
            ? { type: "image", data: o.data["image/png"], mimeType: "image/png" }
            : { type: "text", text: o.data?.["text/plain"] ?? "" }
      );
      return send(frame.id, true, {
        summary: outputs.map(o => o.text ?? `${o.data?.["text/plain"] ?? "figure"}`).join("\n"),
        content,
        structured: { cell_id: cell.id, duration_ms: 1, outputs },
      });
    }

    case "notebook_run_all":
      await sleep(500);
      return send(frame.id, true, { summary: `ran ${state.cells.length} cells` });

    case "kernel_restart":
      state.cells.forEach(c => (c.outputs = []));
      return send(frame.id, true, { summary: "kernel restarted; all state lost" });

    case "file_list":
      return send(frame.id, true, {
        summary: [...state.files.keys()].join(", "),
        structured: { files: [...state.files.keys()] },
      });

    case "file_read": {
      const text = state.files.get(a.path);
      if (text == null) throw new Error(`no such file: ${a.path}`);
      return send(frame.id, true, { summary: text, structured: { path: a.path, content: text } });
    }

    case "file_write":
      state.files.set(a.path, a.content ?? "");
      return send(frame.id, true, { summary: `wrote ${a.path}` });

    default:
      throw new Error(`unknown kind: ${frame.kind}`);
  }
}

function find(id) {
  const cell = state.cells.find(c => c.id === id);
  if (!cell) throw new Error(`no such cell: ${id}`);
  return cell;
}

function send(id, ok, a, b) {
  const content = [];
  if (typeof a === "string") content.push({ type: "text", text: a });
  if (a && typeof a === "object" && Array.isArray(a.content)) content.push(...a.content);
  if (a && typeof a === "object" && "summary" in a && !content.some(c => c.type === "text")) {
    content.unshift({ type: "text", text: String(a.summary) });
  }
  if (!content.length) content.push({ type: "text", text: JSON.stringify(a ?? "") });
  const structured = a && typeof a === "object" ? (a.structured ?? b?.structured) : b?.structured;
  const error = ok ? undefined : typeof a === "string" ? a : undefined;
  ws.send(JSON.stringify({ type: "result", id, ok, content, structured, error }));
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

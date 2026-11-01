import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { CfWorkerJsonSchemaValidator } from "@modelcontextprotocol/sdk/validation/cfworker";
import type { CallToolResult, ContentBlock } from "@modelcontextprotocol/sdk/types.js";
import { z, type ZodRawShape } from "zod";
import { RunnerHub, sha256hex, type Env } from "./hub";

export { RunnerHub };

// ── limits ─────────────────────────────────────────────────────────

const TEXT_BUDGET = 32_000;
const IMAGE_MAX_B64 = 1_200_000;
const IMAGE_MAX_COUNT = 4;
const DEFAULT_WAIT_MS = 8_000;
const RUN_WAIT_MS = 20_000;
const MAX_WAIT_MS = 120_000;

const INSTRUCTIONS = [
  "Drives a JupyterLite notebook running in the user's browser.",
  "Call site_info first. If runner_online is false, ask the user to open the runner page and click Pair; do not retry in a loop.",
  "Cells are addressed by cell_id returned from cell_insert/cell_edit, never by index.",
  "cell_run may return status 'running' with a job_id: call job_result with that id until it finishes.",
  "Keep cells small and bounded. If the kernel hangs, kernel_restart is the only recovery.",
].join(" ");

const OFFLINE_TEXT = [
  "Runner offline: no browser tab is connected to this pair.",
  "Ask the user to open the JupyterLite runner page, click 'Pair', and keep the tab open.",
  "Nothing was executed.",
].join(" ");

// ── worker entry ───────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return json({ ok: true, now: Date.now(), service: "tvbl-mcp-hub" });
    }

    if (url.pathname === "/pair" && request.method === "OPTIONS") {
      return withCors(new Response(null, { status: 204 }));
    }

    if (url.pathname === "/pair" && request.method === "POST") {
      return withCors(await createPair(request, env));
    }

    if (url.pathname === "/run") {
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("expected websocket", { status: 426 });
      }
      const pairId = url.searchParams.get("pair") ?? "";
      if (!isId(pairId)) return new Response("bad pair", { status: 400 });
      return env.HUB.getByName(pairId).fetch(request);
    }

    const match = url.pathname.match(/^\/mcp\/([A-Za-z0-9_-]{1,64})(?:\/([A-Za-z0-9_-]{1,128}))?$/);
    if (match) {
      const [, pairId, pathToken] = match;
      const token = pathToken ?? bearer(request);
      if (!token || !(await authorized(env, pairId, token))) {
        return new Response("unauthorized", {
          status: 401,
          headers: { "www-authenticate": 'Bearer realm="tvbl-mcp-hub"' },
        });
      }
      return handleMcp(request, env, pairId);
    }

    return new Response("not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;

// ── MCP ────────────────────────────────────────────────────────────

async function handleMcp(request: Request, env: Env, pairId: string): Promise<Response> {
  const server = new McpServer(
    { name: "tvbl-mcp-hub", version: "0.1.0", title: "TVBL JupyterLite notebook" },
    {
      instructions: INSTRUCTIONS,
      jsonSchemaValidator: new CfWorkerJsonSchemaValidator(),
    },
  );
  registerTools(server, env, pairId);

  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  });
  await server.connect(transport);
  return transport.handleRequest(request);
}

type RelayResponse =
  | { status: "offline" }
  | { status: "running"; jobId: string }
  | {
      status: "ok" | "error" | "cancelled";
      jobId: string;
      content?: unknown;
      structured?: unknown;
      error?: string;
    };

interface ToolSpec {
  name: string;
  title: string;
  description: string;
  shape: ZodRawShape;
  annotations: {
    readOnlyHint?: boolean;
    destructiveHint?: boolean;
    idempotentHint?: boolean;
    openWorldHint?: boolean;
  };
  kind?: string;
  waitMs?: number;
}

const SPECS: ToolSpec[] = [
  {
    name: "site_info",
    title: "Runner status",
    description:
      "Use this first, and whenever you are unsure about the environment. Reports whether the browser runner is online, the kernel, the notebook, and the hub limits.",
    shape: {},
    annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  },
  {
    name: "notebook_list",
    title: "List notebooks",
    description:
      "Use this to see what notebooks exist in the runner's browser workspace before opening or creating one.",
    shape: {},
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "notebook_list",
  },
  {
    name: "notebook_read",
    title: "Read notebook",
    description:
      "Use this to read the current state of a notebook (cells with ids, sources, execution state and optionally outputs). Re-read after the user may have edited it.",
    shape: {
      path: z.string().optional().describe("Notebook path; defaults to the active notebook."),
      include_outputs: z.boolean().optional().describe("Include cell outputs (default true)."),
      max_chars: z.number().int().min(500).max(60_000).optional(),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "notebook_read",
  },
  {
    name: "notebook_create",
    title: "Create notebook",
    description:
      "Use this when the user wants a new notebook, or when no notebook is open. Creates and opens it so the user can watch.",
    shape: {
      path: z.string().optional(),
      kernel: z.string().optional().describe("Kernel name, e.g. xeus-python."),
    },
    annotations: { destructiveHint: false },
    kind: "notebook_create",
  },
  {
    name: "notebook_open",
    title: "Open notebook",
    description:
      "Use this to bring a notebook to the foreground so the user can see the cells you are about to add or run.",
    shape: { path: z.string() },
    annotations: {},
    kind: "notebook_open",
  },
  {
    name: "cell_insert",
    title: "Insert cell",
    description:
      "Use this to add a code or markdown cell. Returns the stable cell_id; use that id for every later edit or run.",
    shape: {
      source: z.string(),
      cell_type: z.enum(["code", "markdown"]).optional(),
      index: z.number().int().min(0).optional(),
      notebook: z.string().optional(),
    },
    annotations: {},
    kind: "cell_insert",
  },
  {
    name: "cell_edit",
    title: "Edit cell",
    description:
      "Use this to replace a cell's source after reading the notebook. Address the cell by cell_id, not by index.",
    shape: { cell_id: z.string(), source: z.string() },
    annotations: { idempotentHint: true },
    kind: "cell_edit",
  },
  {
    name: "cell_delete",
    title: "Delete cell",
    description: "Use this to remove a cell by cell_id.",
    shape: { cell_id: z.string() },
    annotations: { destructiveHint: true },
    kind: "cell_delete",
  },
  {
    name: "cell_move",
    title: "Move cell",
    description: "Use this to reorder cells. Address the cell by cell_id.",
    shape: { cell_id: z.string(), index: z.number().int().min(0) },
    annotations: {},
    kind: "cell_move",
  },
  {
    name: "cell_run",
    title: "Run cell",
    description:
      "Use this to execute one code cell and get its outputs (text and images). May return status 'running' with a job_id for long cells; then poll job_result.",
    shape: {
      cell_id: z.string(),
      wait_ms: z.number().int().min(0).max(MAX_WAIT_MS).optional(),
    },
    annotations: { openWorldHint: true },
    kind: "cell_run",
    waitMs: RUN_WAIT_MS,
  },
  {
    name: "notebook_run_all",
    title: "Run all cells",
    description:
      "Use this only when the notebook is meant to be executed top to bottom; it is slower and fails at the first error. Returns the run summary or a job_id.",
    shape: {
      path: z.string().optional(),
      wait_ms: z.number().int().min(0).max(MAX_WAIT_MS).optional(),
    },
    annotations: { openWorldHint: true },
    kind: "notebook_run_all",
    waitMs: RUN_WAIT_MS,
  },
  {
    name: "job_result",
    title: "Job result",
    description:
      "Use this to fetch the result of a job returned as status 'running' by cell_run or notebook_run_all.",
    shape: { job_id: z.string() },
    annotations: { readOnlyHint: true, idempotentHint: true },
  },
  {
    name: "job_cancel",
    title: "Cancel job",
    description:
      "Use this to abandon a running job. The runner may keep computing until the cell finishes or the kernel is restarted.",
    shape: { job_id: z.string() },
    annotations: {},
  },
  {
    name: "kernel_restart",
    title: "Restart kernel",
    description:
      "Use this when the kernel is hung or memory is exhausted. All Python state is lost; the notebook cells remain.",
    shape: {},
    annotations: { destructiveHint: true },
    kind: "kernel_restart",
  },
  {
    name: "file_list",
    title: "List files",
    description: "Use this to list files in the runner's workspace (not just notebooks).",
    shape: { path: z.string().optional() },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "file_list",
  },
  {
    name: "file_read",
    title: "Read file",
    description: "Use this to read a text file from the runner's workspace.",
    shape: { path: z.string(), max_chars: z.number().int().min(100).max(60_000).optional() },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "file_read",
  },
  {
    name: "file_write",
    title: "Write file",
    description: "Use this to create or overwrite a text file in the runner's workspace.",
    shape: { path: z.string(), content: z.string() },
    annotations: { destructiveHint: true },
    kind: "file_write",
  },
];

function registerTools(server: McpServer, env: Env, pairId: string): void {
  for (const spec of SPECS) {
    server.registerTool(
      spec.name,
      {
        title: spec.title,
        description: spec.description,
        inputSchema: spec.shape,
        annotations: spec.annotations,
      },
      async (args: Record<string, unknown>): Promise<CallToolResult> => {
        if (spec.name === "site_info") return siteInfo(env, pairId);
        if (spec.name === "job_result") return jobResult(env, pairId, String(args.job_id ?? ""));
        if (spec.name === "job_cancel") return jobCancel(env, pairId, String(args.job_id ?? ""));
        const wait = clampWait(args.wait_ms, spec.waitMs ?? DEFAULT_WAIT_MS);
        const { wait_ms: _ignored, ...forward } = args;
        return present(await relay(env, pairId, spec.kind ?? spec.name, forward, wait));
      },
    );
  }
}

// ── tool implementations ───────────────────────────────────────────

async function siteInfo(env: Env, pairId: string): Promise<CallToolResult> {
  const status = (await doJson(env, pairId, "/status")) as {
    online?: boolean;
    runner?: string | null;
    lastSeen?: number | null;
    pending?: number;
    pairedAt?: number | null;
  };

  const lines = [
    `runner_online: ${status.online ? "true" : "false"}`,
    status.runner ? `runner: ${status.runner}` : null,
    status.pending ? `pending_jobs: ${status.pending}` : null,
    `runner_page: ${env.RUNNER_URL ?? "(unknown)"}`,
    `pair_id: ${pairId}`,
    `cell_addressing: use cell_id`,
    `default_wait_ms: ${RUN_WAIT_MS} (long jobs return a job_id)`,
  ].filter(Boolean) as string[];

  if (!status.online) {
    lines.push("", OFFLINE_TEXT);
  } else {
    const r = await relay(env, pairId, "site_info", {}, 2_500);
    if (r.status === "ok") {
      for (const block of normalizeContent(r.content)) {
        if (block.type === "text" && block.text) lines.push("", block.text);
      }
    }
  }

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    structuredContent: {
      runner_online: Boolean(status.online),
      runner: status.runner ?? null,
      runner_page: env.RUNNER_URL ?? null,
      pair_id: pairId,
      pending_jobs: status.pending ?? 0,
    },
    isError: false,
  };
}

async function jobResult(env: Env, pairId: string, id: string): Promise<CallToolResult> {
  if (!id) return errorResult("missing job_id");
  const r = (await doJson(env, pairId, `/job?id=${encodeURIComponent(id)}`)) as RelayResponse;
  return present(r);
}

async function jobCancel(env: Env, pairId: string, id: string): Promise<CallToolResult> {
  if (!id) return errorResult("missing job_id");
  const r = (await doJson(env, pairId, "/cancel", { method: "POST", body: JSON.stringify({ id }) })) as {
    ok?: boolean;
    error?: string;
  };
  if (!r.ok) return errorResult(r.error ?? "cancel failed");
  return { content: [{ type: "text", text: `job ${id} cancelled` }] };
}

// ── relay plumbing ─────────────────────────────────────────────────

async function authorized(env: Env, pairId: string, token: string): Promise<boolean> {
  const res = await env.HUB.getByName(pairId).fetch("https://do/verify", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
  return res.ok;
}

async function relay(
  env: Env,
  pairId: string,
  kind: string,
  args: unknown,
  waitMs: number,
): Promise<RelayResponse> {
  return (await doJson(env, pairId, "/call", {
    method: "POST",
    body: JSON.stringify({ kind, args, waitMs }),
  })) as RelayResponse;
}

async function doJson(env: Env, pairId: string, path: string, init?: RequestInit): Promise<unknown> {
  const res = await env.HUB.getByName(pairId).fetch(`https://do${path}`, init);
  if (!res.ok) {
    return { status: "error", error: `hub ${path} -> ${res.status}` };
  }
  return res.json();
}

function present(r: RelayResponse): CallToolResult {
  if (r.status === "offline") {
    return {
      content: [{ type: "text", text: OFFLINE_TEXT }],
      structuredContent: { runner_online: false },
      isError: true,
    };
  }
  if (r.status === "running") {
    return {
      content: [
        {
          type: "text",
          text: `Still running (job_id ${r.jobId}). Call job_result with this id; do not re-run the cell.`,
        },
      ],
      structuredContent: { status: "running", job_id: r.jobId },
    };
  }

  const content = normalizeContent(r.content);
  if (r.status !== "ok" && !content.some(b => b.type === "text" && b.text.trim())) {
    content.unshift({ type: "text", text: r.error ?? "runner error" });
  }
  const structured =
    r.structured && typeof r.structured === "object" && !Array.isArray(r.structured)
      ? (r.structured as Record<string, unknown>)
      : {};
  return {
    content,
    structuredContent: { ...structured, status: r.status, job_id: r.jobId },
    isError: r.status === "error" || r.status === "cancelled",
  };
}

function normalizeContent(input: unknown): ContentBlock[] {
  const raw = Array.isArray(input)
    ? input
    : input == null
      ? []
      : [{ type: "text", text: typeof input === "string" ? input : JSON.stringify(input) }];

  const out: ContentBlock[] = [];
  let budget = TEXT_BUDGET;
  let images = 0;

  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const block = item as Record<string, unknown>;
    if (block.type === "text" && typeof block.text === "string") {
      if (budget <= 0) continue;
      const text = truncate(block.text, budget);
      budget -= text.length;
      out.push({ type: "text", text });
    } else if (
      block.type === "image" &&
      typeof block.data === "string" &&
      typeof block.mimeType === "string" &&
      images < IMAGE_MAX_COUNT
    ) {
      if (block.data.length > IMAGE_MAX_B64) {
        out.push({ type: "text", text: "[image omitted: too large; see the notebook cell output]" });
        continue;
      }
      images++;
      out.push({ type: "image", data: block.data, mimeType: block.mimeType });
    } else if (block.type === "resource_link" && typeof block.uri === "string") {
      out.push({
        type: "resource_link",
        uri: String(block.uri),
        name: String(block.name ?? block.uri),
        description: typeof block.description === "string" ? block.description : undefined,
      });
    }
  }

  if (out.length === 0) out.push({ type: "text", text: "(no output)" });
  return out;
}

function errorResult(text: string): CallToolResult {
  return { content: [{ type: "text", text }], isError: true };
}

function clampWait(value: unknown, fallback: number): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.max(0, Math.min(MAX_WAIT_MS, Math.floor(n)));
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, Math.max(0, max));
  return `${cut}\n…[truncated ${text.length - cut.length} chars; full output is in the notebook]`;
}

// ── pairing ────────────────────────────────────────────────────────

async function createPair(request: Request, env: Env): Promise<Response> {
  const body = (await request.json().catch(() => ({}))) as { label?: string };
  const pairId = randomHex(8);
  const token = randomHex(32);
  const stub = env.HUB.getByName(pairId);
  const res = await stub.fetch("https://do/init", {
    method: "POST",
    body: JSON.stringify({ tokenHash: await sha256hex(token), label: body.label }),
  });
  if (!res.ok) return json({ error: "pair init failed" }, 500);

  const origin = new URL(request.url).origin;
  return json(
    {
      pair_id: pairId,
      token,
      mcp_url: `${origin}/mcp/${pairId}/${token}`,
      mcp_url_bearer: `${origin}/mcp/${pairId}`,
      runner_ws: `${origin.replace(/^http/, "ws")}/run?pair=${pairId}`,
      runner_url: env.RUNNER_URL ?? null,
    },
    201,
  );
}

function bearer(request: Request): string {
  const h = request.headers.get("authorization") ?? "";
  const m = h.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : "";
}

function isId(v: string): boolean {
  return /^[A-Za-z0-9_-]{1,64}$/.test(v);
}

function randomHex(bytes: number): string {
  const b = new Uint8Array(bytes);
  crypto.getRandomValues(b);
  return [...b].map(x => x.toString(16).padStart(2, "0")).join("");
}

function withCors(response: Response): Response {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "POST, OPTIONS");
  headers.set("access-control-allow-headers", "content-type");
  headers.set("access-control-max-age", "86400");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

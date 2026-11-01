import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { CfWorkerJsonSchemaValidator } from "@modelcontextprotocol/sdk/validation/cfworker";
import type { CallToolResult, ContentBlock } from "@modelcontextprotocol/sdk/types.js";
import { z, type ZodRawShape } from "zod";
import { RunnerHub, normalizeJobError, sha256hex, type Env, type JobError } from "./hub";
import { PairGuard } from "./guard";

export { RunnerHub, PairGuard };

// ── limits ─────────────────────────────────────────────────────────

const TEXT_BUDGET = 32_000;
const IMAGE_MAX_B64 = 1_200_000;
const IMAGE_MAX_COUNT = 4;
const DEFAULT_WAIT_MS = 8_000;
const RUN_WAIT_MS = 20_000;
const MAX_WAIT_MS = 120_000;

const INSTRUCTIONS = [
  "Drives a JupyterLite notebook in the user's browser.",
  "Call site_info once at the start: it reports runner_online, the default kernel and the open notebook.",
  "Calls are synchronous unless the result has status 'running'; then poll job_result with the job_id.",
  "Before editing or running an existing cell, call notebook_read for the full cell list (stable cell ids; indices shift).",
  "Images arrive as MCP image blocks; if you cannot see them, call notebook_figures for a text summary of the matplotlib axes.",
  "Write calls accept an optional idempotency_key: retrying the same key within 10 minutes does not execute twice.",
  "Large stdout is saved under _mcp/<notebook>/ and can be paged with file_read offset; file_stat reports .npy shape/dtype; file_write supports overwrite/append/replace/lines.",
].join(" ");

const OFFLINE_TEXT =
  "Runner offline: no browser tab is connected to this pair. Ask the user to open the runner page and click Pair; nothing was executed.";

// ── worker entry ───────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return json({ ok: true, now: Date.now(), service: "tvbl-mcp-hub" });
    }

    if (
      (url.pathname === "/pair" || url.pathname === "/pair/challenge") &&
      request.method === "OPTIONS"
    ) {
      return withCors(new Response(null, { status: 204 }));
    }

    if (url.pathname === "/pair/challenge" && request.method === "POST") {
      return withCors(await createChallenge(request, env));
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
  const inline = Number(env.INLINE_WAIT_MS ?? "");
  const runWait = Number.isFinite(inline) && inline >= 0 ? Math.floor(inline) : RUN_WAIT_MS;
  const server = new McpServer(
    { name: "tvbl-mcp-hub", version: "0.4.0", title: "TVBL JupyterLite notebook" },
    {
      instructions: INSTRUCTIONS,
      jsonSchemaValidator: new CfWorkerJsonSchemaValidator(),
    },
  );
  registerTools(server, env, pairId, runWait);

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
      error?: JobError | string;
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
  idempotent?: boolean;
  local?: "status" | "job" | "cancel";
}

const COMMON = {
  _confirmationReason: z
    .string()
    .max(500)
    .optional()
    .describe(
      "Optional. Some clients attach a confirmation reason to write tools; it is accepted uniformly and ignored.",
    ),
};

const IDEM = {
  idempotency_key: z
    .string()
    .min(1)
    .max(200)
    .optional()
    .describe(
      "Optional. A retry with the same key within 10 minutes returns the original result instead of executing again.",
    ),
};

const SPECS: ToolSpec[] = [
  {
    name: "site_info",
    title: "Runner status",
    description:
      "Use this first. Reports whether the browser runner is online, the default kernel and available kernels, the open notebook, and the hub limits.",
    shape: {},
    annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
    local: "status",
  },
  {
    name: "notebook_list",
    title: "List notebooks",
    description:
      "Use this to see what notebooks and files exist in the runner's browser workspace before opening or creating one.",
    shape: {},
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "notebook_list",
  },
  {
    name: "notebook_read",
    title: "Read notebook",
    description:
      "Use this before any surgical edit of an existing notebook, and whenever the user may have changed it. Returns every cell with a stable cell_id, type, source, execution state and (optionally) outputs. Long sources and outputs are truncated with an explicit marker; raise max_cell_chars / max_chars if needed.",
    shape: {
      path: z.string().optional().describe("Notebook path; defaults to the active notebook."),
      include_outputs: z
        .boolean()
        .optional()
        .describe("Include cell outputs (default true). Set false for a cheap cell list."),
      max_cell_chars: z
        .number()
        .int()
        .min(500)
        .max(60_000)
        .optional()
        .describe("Per-cell source cap, default 6000."),
      max_chars: z
        .number()
        .int()
        .min(500)
        .max(60_000)
        .optional()
        .describe("Total output text cap, default 8000."),
      max_images: z
        .number()
        .int()
        .min(0)
        .max(8)
        .optional()
        .describe("How many stored figure outputs to attach (default 4)."),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "notebook_read",
  },
  {
    name: "notebook_create",
    title: "Create notebook",
    description:
      "Use this when the user wants a new notebook. Creates and opens it, starts the kernel, and returns the empty cell's cell_id. If the path already exists it opens that notebook instead of failing (set if_exists='error' to fail with code already_exists).",
    shape: {
      path: z.string().optional().describe("e.g. demo.ipynb"),
      kernel: z
        .string()
        .optional()
        .describe("Kernel name; defaults to site_info.default_kernel."),
      if_exists: z
        .enum(["open", "error"])
        .optional()
        .describe("Default 'open': reuse an existing notebook. 'error' returns code already_exists."),
      ...IDEM,
    },
    annotations: { destructiveHint: false },
    kind: "notebook_create",
    idempotent: true,
  },
  {
    name: "notebook_open",
    title: "Open notebook",
    description:
      "Use this to bring an existing notebook to the foreground so the user can see it. Opens (or reveals) the tab and starts its kernel if needed.",
    shape: { path: z.string().describe("Notebook path, e.g. demo.ipynb"), ...IDEM },
    annotations: { idempotentHint: true },
    kind: "notebook_open",
    idempotent: true,
  },
  {
    name: "cell_insert",
    title: "Insert cell",
    description:
      "Use this to add a code or markdown cell. Returns the stable cell_id; use that id for every later edit or run. If the notebook's only cell is empty, this writes into it instead of leaving an orphan.",
    shape: {
      source: z.string().describe("Cell source"),
      cell_type: z.enum(["code", "markdown"]).optional(),
      index: z.number().int().min(0).optional().describe("Insert position; default appends."),
      notebook: z.string().optional(),
      ...IDEM,
    },
    annotations: {},
    kind: "cell_insert",
    idempotent: true,
  },
  {
    name: "cell_edit",
    title: "Edit cell",
    description:
      "Use this to replace a cell's source after reading the notebook. Address the cell by cell_id, not by index.",
    shape: { cell_id: z.string(), source: z.string(), ...IDEM },
    annotations: { idempotentHint: true },
    kind: "cell_edit",
    idempotent: true,
  },
  {
    name: "cell_delete",
    title: "Delete cell",
    description: "Use this to remove a cell by cell_id.",
    shape: { cell_id: z.string(), ...IDEM },
    annotations: { destructiveHint: true },
    kind: "cell_delete",
    idempotent: true,
  },
  {
    name: "cell_move",
    title: "Move cell",
    description: "Use this to reorder cells. Address the cell by cell_id.",
    shape: { cell_id: z.string(), index: z.number().int().min(0), ...IDEM },
    annotations: {},
    kind: "cell_move",
    idempotent: true,
  },
  {
    name: "cell_run",
    title: "Run cell",
    description:
      "Use this to execute one code cell and get its outputs (text and image blocks). Usually returns synchronously; a cell still running after wait_ms returns status 'running' with a job_id, then poll job_result. Do not re-run the cell while a job is running.",
    shape: {
      cell_id: z.string(),
      wait_ms: z.number().int().min(0).max(MAX_WAIT_MS).optional(),
      ...IDEM,
    },
    annotations: { openWorldHint: true },
    kind: "cell_run",
    waitMs: RUN_WAIT_MS,
    idempotent: true,
  },
  {
    name: "notebook_run_all",
    title: "Run all cells",
    description:
      "Use this only when the notebook is meant to be executed top to bottom; it is slower and stops at the first error. Returns a summary (and the errors) or a job_id.",
    shape: {
      path: z.string().optional(),
      wait_ms: z.number().int().min(0).max(MAX_WAIT_MS).optional(),
      ...IDEM,
    },
    annotations: { openWorldHint: true },
    kind: "notebook_run_all",
    waitMs: RUN_WAIT_MS,
    idempotent: true,
  },
  {
    name: "notebook_figures",
    title: "Summarise figures",
    description:
      "Use this when image outputs are not visible to you, or to verify a plot numerically. Returns a text summary of the matplotlib figures currently alive in the kernel: axes titles and labels, line labels, sample counts and min/max ranges.",
    shape: {},
    annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
    kind: "notebook_figures",
  },
  {
    name: "job_result",
    title: "Job result",
    description:
      "Use this to fetch the result of a job that returned status 'running'. Returns status 'running' until the runner finishes.",
    shape: {
      job_id: z.string(),
      wait_ms: z
        .number()
        .int()
        .min(0)
        .max(25_000)
        .optional()
        .describe("Long-poll for up to this long; default 15000."),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
    local: "job",
  },
  {
    name: "job_cancel",
    title: "Cancel job",
    description:
      "Use this to abandon a running job. The runner may keep computing until the cell finishes or the kernel is restarted.",
    shape: { job_id: z.string() },
    annotations: {},
    local: "cancel",
  },
  {
    name: "kernel_restart",
    title: "Restart kernel",
    description:
      "Use this when the kernel is hung, out of memory, or a cell is looping. All Python state is lost; the notebook cells remain.",
    shape: { ...IDEM },
    annotations: { destructiveHint: true },
    kind: "kernel_restart",
    idempotent: true,
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
    description:
      "Use this to read a text file from the runner's workspace. The body is returned as text and also as structuredContent.content, so clients that only read structured output still get it. Binary files are returned base64-encoded with encoding='base64'.",
    shape: {
      path: z.string(),
      max_chars: z.number().int().min(100).max(200_000).optional().describe("Default 20000."),
      offset: z
        .number()
        .int()
        .min(0)
        .optional()
        .describe("Character offset for paging through large files; the result includes next_offset."),
      encoding: z
        .enum(["auto", "text", "base64"])
        .optional()
        .describe("auto (default) picks base64 for known binary extensions; force it for other binaries."),
    },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "file_read",
  },
  {
    name: "file_write",
    title: "Write file",
    description:
      "Use this to create or edit a text file. mode='overwrite' (default) replaces it; 'append' adds to the end; 'replace' applies edits:[{old_text,new_text,replace_all?}]; 'lines' replaces start_line..end_line with content (end_line = start_line-1 inserts before start_line). No read-modify-write round trips.",
    shape: {
      path: z.string(),
      content: z
        .string()
        .optional()
        .describe("New content; replacement text for replace/lines modes."),
      mode: z
        .enum(["overwrite", "append", "replace", "lines"])
        .optional()
        .describe("Default overwrite."),
      edits: z
        .array(
          z.object({
            old_text: z.string(),
            new_text: z.string(),
            replace_all: z.boolean().optional(),
          }),
        )
        .optional()
        .describe("For mode='replace'."),
      start_line: z.number().int().min(1).optional().describe("For mode='lines' (1-based)."),
      end_line: z
        .number()
        .int()
        .min(0)
        .optional()
        .describe("For mode='lines', inclusive; end_line = start_line-1 inserts before start_line."),
      ...IDEM,
    },
    annotations: { destructiveHint: true, idempotentHint: true },
    kind: "file_write",
    idempotent: true,
  },
  {
    name: "file_stat",
    title: "File metadata",
    description:
      "Use this before reading a data file: returns type, size, format, and for .npy the dtype/shape/fortran_order (parsed from the header), for .ipynb the cell count.",
    shape: { path: z.string() },
    annotations: { readOnlyHint: true, idempotentHint: true },
    kind: "file_stat",
  },
  {
    name: "file_delete",
    title: "Delete file",
    description:
      "Use this to remove a file or directory. Directories must be empty unless recursive=true.",
    shape: {
      path: z.string(),
      recursive: z.boolean().optional().describe("Delete directory contents too."),
      ...IDEM,
    },
    annotations: { destructiveHint: true },
    kind: "file_delete",
    idempotent: true,
  },
  {
    name: "file_move",
    title: "Move or rename",
    description: "Use this to move or rename a file or directory.",
    shape: { from: z.string(), to: z.string(), ...IDEM },
    annotations: { destructiveHint: true, idempotentHint: true },
    kind: "file_move",
    idempotent: true,
  },
  {
    name: "file_mkdir",
    title: "Create directory",
    description: "Use this to create a directory (idempotent, like mkdir -p).",
    shape: { path: z.string(), ...IDEM },
    annotations: { idempotentHint: true },
    kind: "file_mkdir",
    idempotent: true,
  },
];

function registerTools(server: McpServer, env: Env, pairId: string, runWait: number): void {
  for (const spec of SPECS) {
    server.registerTool(
      spec.name,
      {
        title: spec.title,
        description: spec.description,
        inputSchema: { ...COMMON, ...spec.shape },
        annotations: spec.annotations,
      },
      async (args: Record<string, unknown>): Promise<CallToolResult> => {
        if (spec.local === "status") return siteInfo(env, pairId);
        if (spec.local === "job") {
          return jobResult(
            env,
            pairId,
            String(args.job_id ?? ""),
            Math.min(25_000, clampWait(args.wait_ms, 15_000)),
          );
        }
        if (spec.local === "cancel") return jobCancel(env, pairId, String(args.job_id ?? ""));
        const { idempotency_key, wait_ms, _confirmationReason, ...forward } = args;
        void _confirmationReason;
        const fallback =
          spec.kind === "cell_run" || spec.kind === "notebook_run_all"
            ? runWait
            : (spec.waitMs ?? DEFAULT_WAIT_MS);
        const wait = clampWait(wait_ms, fallback);
        const key = typeof idempotency_key === "string" ? idempotency_key : undefined;
        return present(
          await relay(env, pairId, spec.kind ?? spec.name, forward, wait, key),
        );
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

  let runner: Record<string, unknown> = {};
  if (status.online) {
    const r = await relay(env, pairId, "site_info", {}, 2_500);
    if (r.status === "ok" && r.structured && typeof r.structured === "object") {
      runner = r.structured as Record<string, unknown>;
    }
  }

  const defaultKernel = (runner.default_kernel as string | undefined) ?? null;
  const kernels = Array.isArray(runner.kernels) ? (runner.kernels as string[]) : [];
  const notebook = (runner.notebook as string | null | undefined) ?? null;
  const cells = typeof runner.cells === "number" ? runner.cells : null;
  const kernel = (runner.kernel as string | null | undefined) ?? null;

  const lines = [
    `runner_online: ${status.online ? "true" : "false"}`,
    status.runner ? `runner: ${status.runner}` : null,
    `runner_page: ${env.RUNNER_URL ?? "(unknown)"}`,
    `pair_id: ${pairId}`,
    defaultKernel ? `default_kernel: ${defaultKernel}` : null,
    kernels.length ? `kernels: ${kernels.join(", ")}` : null,
    notebook ? `notebook: ${notebook}${cells !== null ? ` (${cells} cells)` : ""}` : "notebook: (none open)",
    kernel ? `kernel: ${kernel} (${String(runner.kernel_status ?? "unknown")})` : null,
    "cells: address by cell_id; call notebook_read for the full list",
  ].filter(Boolean) as string[];

  if (!status.online) {
    lines.push("", OFFLINE_TEXT);
  }

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    structuredContent: {
      ok: true,
      runner_online: Boolean(status.online),
      runner: status.runner ?? null,
      runner_page: env.RUNNER_URL ?? null,
      pair_id: pairId,
      pending_jobs: status.pending ?? 0,
      default_kernel: defaultKernel,
      kernels,
      notebook,
      cells,
      kernel,
    },
    isError: false,
  };
}

async function jobResult(env: Env, pairId: string, id: string, waitMs: number): Promise<CallToolResult> {
  if (!id) return errorResult("invalid_argument", "missing job_id");
  const r = (await doJson(
    env,
    pairId,
    `/job?id=${encodeURIComponent(id)}&wait=${Math.max(0, waitMs)}`,
  )) as RelayResponse;
  return present(r);
}

async function jobCancel(env: Env, pairId: string, id: string): Promise<CallToolResult> {
  if (!id) return errorResult("invalid_argument", "missing job_id");
  const r = (await doJson(env, pairId, "/cancel", {
    method: "POST",
    body: JSON.stringify({ id }),
  })) as { ok?: boolean; error?: unknown };
  if (!r.ok) {
    const err = normalizeJobError(r.error);
    return errorResult(err.code, err.message, err.hint);
  }
  return { content: [{ type: "text", text: `job ${id} cancelled` }], structuredContent: { ok: true, job_id: id } };
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
  idempotencyKey?: string,
): Promise<RelayResponse> {
  return (await doJson(env, pairId, "/call", {
    method: "POST",
    body: JSON.stringify({ kind, args, waitMs, idempotencyKey }),
  })) as RelayResponse;
}

async function doJson(env: Env, pairId: string, path: string, init?: RequestInit): Promise<unknown> {
  const res = await env.HUB.getByName(pairId).fetch(`https://do${path}`, init);
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: unknown } | null;
    return {
      status: "error",
      error: normalizeJobError(body?.error ?? `hub ${path} -> ${res.status}`),
    };
  }
  return res.json();
}

function present(r: RelayResponse): CallToolResult {
  if (r.status === "offline") {
    return errorResult("runner_offline", OFFLINE_TEXT, "Open the runner page and click Pair.");
  }

  if (r.status === "running") {
    return {
      content: [
        {
          type: "text",
          text: `Still running (job_id ${r.jobId}). Call job_result with this id; do not re-run the cell.`,
        },
      ],
      structuredContent: { ok: true, status: "running", job_id: r.jobId },
    };
  }

  if (r.status === "error" || r.status === "cancelled") {
    const error = normalizeJobError(r.error);
    const content = normalizeContent(r.content);
    if (!content.some(block => block.type === "text" && block.text.trim())) {
      content.unshift({
        type: "text",
        text: error.hint ? `${error.message}\n${error.hint}` : error.message,
      });
    }
    return {
      content,
      structuredContent: { ok: false, error },
      isError: true,
    };
  }

  const structured =
    r.structured && typeof r.structured === "object" && !Array.isArray(r.structured)
      ? (r.structured as Record<string, unknown>)
      : {};
  return {
    content: normalizeContent(r.content),
    structuredContent: { ok: true, job_id: r.jobId, ...structured },
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
      out.push({
        type: "image",
        data: block.data,
        mimeType: block.mimeType,
      });
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

function errorResult(code: string, message: string, hint?: string): CallToolResult {
  return {
    content: [{ type: "text", text: hint ? `${message}\n${hint}` : message }],
    structuredContent: { ok: false, error: { code, message, hint } },
    isError: true,
  };
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

async function createChallenge(request: Request, env: Env): Promise<Response> {
  const { bits, rate } = powConfig(env);
  const limited = await rateLimit(env, request, rate);
  if (limited) return limited;
  if (bits === 0) return json({ nonce: null, bits: 0 });
  const res = await env.GUARD.getByName("global").fetch("https://guard/challenge", {
    method: "POST",
    body: JSON.stringify({ bits }),
  });
  if (!res.ok) return json({ error: "challenge_failed" }, 500);
  return new Response(res.body, {
    status: 200,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

async function createPair(request: Request, env: Env): Promise<Response> {
  const body = (await request.json().catch(() => ({}))) as {
    label?: string;
    nonce?: string;
    solution?: string;
  };
  const { bits, rate } = powConfig(env);

  if (bits > 0) {
    if (!body.nonce || typeof body.solution !== "string") {
      return json({ error: "proof_of_work_required" }, 401);
    }
    const spend = await env.GUARD.getByName("global").fetch("https://guard/spend", {
      method: "POST",
      body: JSON.stringify({ nonce: body.nonce, solution: body.solution, bits }),
    });
    const spent = (await spend.json().catch(() => ({}))) as { ok?: boolean; reason?: string };
    if (!spent.ok) {
      return json({ error: "proof_of_work_failed", reason: spent.reason }, 401);
    }
  } else {
    const limited = await rateLimit(env, request, rate);
    if (limited) return limited;
  }

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

function powConfig(env: Env): { bits: number; rate: number } {
  const bits = Number(env.POW_BITS ?? "15");
  const rate = Number(env.PAIR_RATE_PER_HOUR ?? "6");
  return {
    bits: Number.isFinite(bits) && bits > 0 ? Math.min(24, Math.floor(bits)) : 0,
    rate: Number.isFinite(rate) && rate > 0 ? Math.floor(rate) : 0,
  };
}

async function rateLimit(env: Env, request: Request, limit: number): Promise<Response | null> {
  if (limit <= 0) return null;
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const bucket = await sha256hex(familyBucket(ip));
  const res = await env.GUARD.getByName("global").fetch("https://guard/rate", {
    method: "POST",
    body: JSON.stringify({ bucket, limit, windowMs: 3_600_000 }),
  });
  if (res.ok) return null;
  const body = (await res.json().catch(() => ({}))) as { retryAfter?: number };
  return json({ error: "rate_limited", retry_after: body.retryAfter ?? 3600 }, 429);
}

/** Coarse bucket: /24 for IPv4, /64 for IPv6 (a customer normally owns a /64). */
function familyBucket(ip: string): string {
  if (ip.includes(".")) {
    return `${ip.split(".").slice(0, 3).join(".")}.0/24`;
  }
  return `${ip.split(":").filter(Boolean).slice(0, 4).join(":")}::/64`;
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

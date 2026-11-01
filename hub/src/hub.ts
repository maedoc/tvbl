import { DurableObject } from "cloudflare:workers";

/**
 * One RunnerHub per pairing. Holds:
 *  - the pair record (token hash),
 *  - the runner WebSocket(s) (hibernatable),
 *  - job records in SQLite-backed storage, so results survive a dead request,
 *    a reloaded tab, or a DO eviction.
 */

export interface Env {
  HUB: DurableObjectNamespace<RunnerHub>;
  RUNNER_URL?: string;
}

export const JOB_TTL_MS = 60 * 60 * 1000;
export const JOB_STALE_MS = 15 * 60 * 1000;
const MAX_KEPT_JOBS = 200;

export type JobStatus = "queued" | "running" | "ok" | "error" | "cancelled";

export interface JobRecord {
  id: string;
  kind: string;
  args: unknown;
  status: JobStatus;
  createdAt: number;
  updatedAt: number;
  content?: unknown;
  structured?: unknown;
  error?: string;
}

interface PairRecord {
  tokenHash: string;
  createdAt: number;
  label?: string;
}

type RunnerFrame =
  | { type: "auth"; token: string; label?: string }
  | { type: "result"; id: string; ok: boolean; content?: unknown; structured?: unknown; error?: string }
  | { type: "progress"; id: string; message?: string };

interface Attachment {
  authed: boolean;
  label?: string;
  lastSeen: number;
}

export async function sha256hex(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, "0")).join("");
}

export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export class RunnerHub extends DurableObject<Env> {
  /** In-flight long-poll waiters. Not durable by design: a reload re-polls job_result. */
  private waiters = new Map<string, (job: JobRecord | null) => void>();
  private authTimers = new Map<WebSocket, number>();

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/run") {
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("expected websocket", { status: 426 });
      }
      return this.attach();
    }

    switch (`${request.method} ${url.pathname}`) {
      case "POST /init":
        return this.init(request);
      case "POST /verify":
        return this.verify(request);
      case "GET /status":
        return this.status();
      case "POST /call":
        return this.call(request);
      case "GET /job":
        return this.job(url.searchParams.get("id") ?? "");
      case "POST /cancel":
        return this.cancel(request);
      default:
        return new Response("not found", { status: 404 });
    }
  }

  // ── pairing ──────────────────────────────────────────────────────

  private async init(request: Request): Promise<Response> {
    const body = (await request.json()) as { tokenHash: string; label?: string };
    const existing = await this.ctx.storage.get<PairRecord>("pair");
    if (existing) return new Response("pair already initialised", { status: 409 });
    await this.ctx.storage.put<PairRecord>("pair", {
      tokenHash: body.tokenHash,
      createdAt: Date.now(),
      label: body.label,
    });
    return json({ ok: true });
  }

  private async verify(request: Request): Promise<Response> {
    const { token } = (await request.json()) as { token?: string };
    const pair = await this.ctx.storage.get<PairRecord>("pair");
    if (!pair || !token) return json({ ok: false }, 401);
    const hash = await sha256hex(token);
    return safeEqual(hash, pair.tokenHash) ? json({ ok: true }) : json({ ok: false }, 401);
  }

  // ── runner socket ────────────────────────────────────────────────

  private attach(): Response {
    const { 0: client, 1: server } = new WebSocketPair();
    this.ctx.acceptWebSocket(server, ["runner"]);
    server.serializeAttachment({ authed: false, lastSeen: Date.now() } satisfies Attachment);

    const timer = setTimeout(() => {
      this.authTimers.delete(server);
      const att = server.deserializeAttachment() as Attachment | null;
      if (!att?.authed) server.close(4001, "auth timeout");
    }, 10_000);
    this.authTimers.set(server, timer);

    return new Response(null, { status: 101, webSocket: client });
  }

  private runners(): Array<{ ws: WebSocket; att: Attachment }> {
    const out: Array<{ ws: WebSocket; att: Attachment }> = [];
    for (const ws of this.ctx.getWebSockets()) {
      const att = ws.deserializeAttachment() as Attachment | null;
      if (att?.authed) out.push({ ws, att });
    }
    return out.sort((a, b) => b.att.lastSeen - a.att.lastSeen);
  }

  private async status(): Promise<Response> {
    const pair = await this.ctx.storage.get<PairRecord>("pair");
    const runners = this.runners();
    const jobs = await this.ctx.storage.list<JobRecord>({ prefix: "job:" });
    let pending = 0;
    for (const job of jobs.values()) {
      if (job.status === "queued" || job.status === "running") pending++;
    }
    return json({
      online: runners.length > 0,
      runner: runners[0]?.att.label ?? null,
      lastSeen: runners[0]?.att.lastSeen ?? null,
      runners: runners.length,
      pending,
      pairedAt: pair?.createdAt ?? null,
    });
  }

  // ── jobs ─────────────────────────────────────────────────────────

  private async call(request: Request): Promise<Response> {
    const { kind, args, waitMs } = (await request.json()) as {
      kind: string;
      args: unknown;
      waitMs?: number;
    };

    const runners = this.runners();
    if (runners.length === 0) return json({ status: "offline" });

    const id = crypto.randomUUID();
    const now = Date.now();
    const job: JobRecord = {
      id,
      kind,
      args,
      status: "running",
      createdAt: now,
      updatedAt: now,
    };
    await this.ctx.storage.put(`job:${id}`, job);

    const wait = waitMs && waitMs > 0 ? this.waitFor(id, waitMs) : Promise.resolve(null);
    const sent = this.send(runners, { type: "job", id, kind, args });
    if (!sent) {
      await this.ctx.storage.put(`job:${id}`, { ...job, status: "error", error: "runner unreachable" });
      return json({ status: "offline" });
    }

    const done = await wait;
    if (!done) return json({ status: "running", jobId: id });
    return json(toRelay(done));
  }

  private send(runners: Array<{ ws: WebSocket }>, frame: unknown): boolean {
    const text = JSON.stringify(frame);
    for (const { ws } of runners) {
      try {
        ws.send(text);
        return true;
      } catch {
        // try the next socket
      }
    }
    return false;
  }

  private waitFor(id: string, ms: number): Promise<JobRecord | null> {
    return new Promise(resolve => {
      const timer = setTimeout(() => {
        this.waiters.delete(id);
        resolve(null);
      }, ms);
      this.waiters.set(id, job => {
        clearTimeout(timer);
        resolve(job);
      });
    });
  }

  private async job(id: string): Promise<Response> {
    if (!id) return json({ status: "error", error: "missing job id" }, 400);
    const job = await this.ctx.storage.get<JobRecord>(`job:${id}`);
    if (!job) return json({ status: "error", error: "unknown job id" }, 404);
    if (
      (job.status === "running" || job.status === "queued") &&
      Date.now() - job.createdAt > JOB_STALE_MS
    ) {
      job.status = "error";
      job.error = "runner never reported completion (tab closed or kernel wedged?)";
      job.updatedAt = Date.now();
      await this.ctx.storage.put(`job:${id}`, job);
    }
    return json(toRelay(job));
  }

  private async cancel(request: Request): Promise<Response> {
    const { id } = (await request.json()) as { id: string };
    const key = `job:${id}`;
    const job = await this.ctx.storage.get<JobRecord>(key);
    if (!job) return json({ ok: false, error: "unknown job id" }, 404);
    job.status = "cancelled";
    job.updatedAt = Date.now();
    await this.ctx.storage.put(key, job);
    this.send(this.runners(), { type: "cancel", id });
    this.waiters.get(id)?.(job);
    this.waiters.delete(id);
    return json({ ok: true });
  }

  // ── socket handlers (hibernation API) ────────────────────────────

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): Promise<void> {
    const text = typeof message === "string" ? message : new TextDecoder().decode(message);
    let frame: RunnerFrame;
    try {
      frame = JSON.parse(text) as RunnerFrame;
    } catch {
      return;
    }

    if (frame.type === "auth") {
      const pair = await this.ctx.storage.get<PairRecord>("pair");
      const hash = await sha256hex(frame.token ?? "");
      if (!pair || !safeEqual(hash, pair.tokenHash)) {
        ws.close(4003, "bad token");
        return;
      }
      const timer = this.authTimers.get(ws);
      if (timer !== undefined) {
        clearTimeout(timer);
        this.authTimers.delete(ws);
      }
      const att: Attachment = { authed: true, label: frame.label, lastSeen: Date.now() };
      ws.serializeAttachment(att);
      ws.send(JSON.stringify({ type: "hello", pair: true }));
      return;
    }

    const att = ws.deserializeAttachment() as Attachment | null;
    if (!att?.authed) {
      ws.close(4003, "not authenticated");
      return;
    }
    ws.serializeAttachment({ ...att, lastSeen: Date.now() });

    if (frame.type === "result") await this.recordResult(frame);
  }

  private async recordResult(frame: Extract<RunnerFrame, { type: "result" }>): Promise<void> {
    const key = `job:${frame.id}`;
    const job = await this.ctx.storage.get<JobRecord>(key);
    if (!job || job.status === "cancelled") return;
    job.status = frame.ok ? "ok" : "error";
    job.content = frame.content;
    job.structured = frame.structured;
    job.error = frame.ok ? undefined : (frame.error ?? "runner error");
    job.updatedAt = Date.now();
    await this.ctx.storage.put(key, job);

    const waiter = this.waiters.get(frame.id);
    if (waiter) {
      this.waiters.delete(frame.id);
      waiter(job);
    }
    await this.prune();
  }

  async webSocketClose(ws: WebSocket): Promise<void> {
    const timer = this.authTimers.get(ws);
    if (timer !== undefined) {
      clearTimeout(timer);
      this.authTimers.delete(ws);
    }
    // In-flight jobs keep running in the (now gone) tab; job_result reports
    // stale after JOB_STALE_MS. Do not close the socket: it is already closed.
  }

  async webSocketError(ws: WebSocket, error: unknown): Promise<void> {
    console.error("runner socket error", error);
    const timer = this.authTimers.get(ws);
    if (timer !== undefined) {
      clearTimeout(timer);
      this.authTimers.delete(ws);
    }
  }

  private async prune(): Promise<void> {
    const all = await this.ctx.storage.list<JobRecord>({ prefix: "job:" });
    const entries = [...all.entries()].sort((a, b) => b[1].createdAt - a[1].createdAt);
    for (const [i, [key, job]] of entries.entries()) {
      const tooOld = Date.now() - job.createdAt > JOB_TTL_MS;
      if (tooOld || i >= MAX_KEPT_JOBS) await this.ctx.storage.delete(key);
    }
  }
}

export function toRelay(job: JobRecord): Record<string, unknown> {
  return {
    status: job.status === "ok" ? "ok" : job.status === "cancelled" ? "cancelled" : job.status === "error" ? "error" : "running",
    jobId: job.id,
    kind: job.kind,
    content: job.content,
    structured: job.structured,
    error: job.error,
  };
}

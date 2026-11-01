import type { ConnState, HubSettings, InboundFrame, JobFrame, JobResult } from './types';

export interface HubClientOptions {
  onState: (state: ConnState, detail?: string) => void;
  onLog: (kind: string, text: string) => void;
  onJob: (job: JobFrame) => Promise<JobResult>;
  onCancel: (id: string) => void;
}

/**
 * WebSocket client for the hub's `/run` endpoint.
 *
 * The browser cannot set handshake headers, so the token is sent as the first
 * frame; the Durable Object closes the socket if it does not match the pair.
 */
export class HubClient {
  private ws: WebSocket | null = null;
  private retry = 1000;
  private timer: number | null = null;
  private disposed = false;
  private cfg: HubSettings;

  constructor(
    cfg: HubSettings,
    private readonly opts: HubClientOptions,
  ) {
    this.cfg = cfg;
  }

  configure(cfg: HubSettings): void {
    this.cfg = cfg;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect(): void {
    this.teardown();
    if (!this.cfg.hubUrl || !this.cfg.pairId || !this.cfg.pairToken) {
      this.opts.onState('offline', 'not paired');
      return;
    }

    const base = this.cfg.hubUrl.replace(/\/+$/, '').replace(/^http/, 'ws');
    const url = `${base}/run?pair=${encodeURIComponent(this.cfg.pairId)}`;
    this.opts.onState('connecting', url);

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      this.opts.onLog('error', `websocket: ${messageOf(err)}`);
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: 'auth',
          token: this.cfg.pairToken,
          label: `JupyterLite · ${navigator.userAgent.includes('Firefox') ? 'Firefox' : 'browser'}`,
        }),
      );
    };

    ws.onmessage = event => {
      void this.onMessage(event);
    };

    ws.onclose = event => {
      if (this.ws === ws) {
        this.ws = null;
      }
      if (this.disposed) {
        return;
      }
      this.opts.onState('offline', `disconnected (${event.code})`);
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose always follows; avoid duplicate logs.
    };
  }

  disconnect(): void {
    this.teardown();
    this.opts.onState('offline', 'disconnected by user');
  }

  private teardown(): void {
    this.disposed = true;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
      try {
        ws.close(1000, 'client disconnect');
      } catch {
        // already closed
      }
    }
    this.disposed = false;
  }

  private scheduleReconnect(): void {
    if (this.disposed || this.timer !== null) {
      return;
    }
    const delay = this.retry;
    this.retry = Math.min(this.retry * 2, 30_000);
    this.opts.onLog('info', `reconnecting in ${Math.round(delay / 1000)}s`);
    this.timer = window.setTimeout(() => {
      this.timer = null;
      this.connect();
    }, delay);
  }

  private async onMessage(event: MessageEvent): Promise<void> {
    let frame: InboundFrame;
    try {
      frame = JSON.parse(String(event.data)) as InboundFrame;
    } catch {
      return;
    }

    if (frame.type === 'hello') {
      this.retry = 1000;
      this.opts.onState('connected', 'this tab is the runner');
      this.opts.onLog('info', 'authenticated with hub');
      return;
    }

    if (frame.type === 'cancel') {
      this.opts.onCancel(frame.id);
      return;
    }

    if (frame.type !== 'job') {
      return;
    }

    const started = performance.now();
    this.opts.onLog('job', `${frame.kind} (${frame.id.slice(0, 8)})`);
    try {
      const result = await this.opts.onJob(frame);
      this.send({
        type: 'result',
        id: frame.id,
        ok: result.ok,
        content: result.content,
        structured: result.structured,
        error: result.error,
      });
      const ms = Math.round(performance.now() - started);
      this.opts.onLog(result.ok ? 'ok' : 'error', `${frame.kind} ${ms}ms`);
    } catch (err) {
      const message = messageOf(err);
      this.send({
        type: 'result',
        id: frame.id,
        ok: false,
        content: [{ type: 'text', text: message }],
        error: message,
      });
      this.opts.onLog('error', `${frame.kind}: ${message}`);
    }
  }

  private send(frame: unknown): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.opts.onLog('error', 'result dropped: socket not open');
      return;
    }
    ws.send(JSON.stringify(frame));
  }
}

export function messageOf(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

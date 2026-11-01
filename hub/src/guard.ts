import { DurableObject } from "cloudflare:workers";

/**
 * Small global guard for `/pair`: proof-of-work nonces and per-CIDR rate
 * limiting. One instance (`getByName("global")`) is enough at this scale and
 * keeps the counters atomic.
 */

const NONCE_TTL_MS = 5 * 60 * 1000;

interface NonceRecord {
  exp: number;
  bits: number;
  used: boolean;
}

interface RateRecord {
  count: number;
  resetAt: number;
}

export class PairGuard extends DurableObject {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    switch (url.pathname) {
      case "/challenge":
        return this.challenge(request);
      case "/spend":
        return this.spend(request);
      case "/rate":
        return this.rate(request);
      default:
        return new Response("not found", { status: 404 });
    }
  }

  private async challenge(request: Request): Promise<Response> {
    const { bits } = (await request.json()) as { bits: number };
    const nonce = randomHex(16);
    const now = Date.now();
    await this.ctx.storage.put<NonceRecord>(`nonce:${nonce}`, {
      exp: now + NONCE_TTL_MS,
      bits,
      used: false,
    });
    await this.prune(now);
    return json({ nonce, bits, expires: now + NONCE_TTL_MS });
  }

  private async spend(request: Request): Promise<Response> {
    const { nonce, solution, bits } = (await request.json()) as {
      nonce?: string;
      solution?: string;
      bits?: number;
    };
    if (!nonce || typeof solution !== "string") {
      return json({ ok: false, reason: "missing_nonce" });
    }
    const key = `nonce:${nonce}`;
    const rec = await this.ctx.storage.get<NonceRecord>(key);
    if (!rec) return json({ ok: false, reason: "unknown_nonce" });
    if (rec.used) return json({ ok: false, reason: "nonce_reused" });
    if (Date.now() > rec.exp) return json({ ok: false, reason: "nonce_expired" });

    const required = Math.max(0, Number(bits ?? rec.bits) || 0);
    const have = await leadingZeroBits(nonce + solution);
    if (have < required) {
      return json({ ok: false, reason: "insufficient_work", have, required });
    }

    rec.used = true;
    await this.ctx.storage.put(key, rec);
    return json({ ok: true });
  }

  private async rate(request: Request): Promise<Response> {
    const { bucket, limit, windowMs } = (await request.json()) as {
      bucket: string;
      limit: number;
      windowMs: number;
    };
    const key = `rl:${bucket}`;
    const now = Date.now();
    const rec = await this.ctx.storage.get<RateRecord>(key);

    if (!rec || now > rec.resetAt) {
      await this.ctx.storage.put<RateRecord>(key, {
        count: 1,
        resetAt: now + Math.max(1_000, windowMs),
      });
      return json({ ok: true, remaining: Math.max(0, limit - 1) });
    }

    if (rec.count >= limit) {
      return json({ ok: false, retryAfter: Math.ceil((rec.resetAt - now) / 1000) }, 429);
    }

    rec.count += 1;
    await this.ctx.storage.put(key, rec);
    await this.prune(now);
    return json({ ok: true, remaining: Math.max(0, limit - rec.count) });
  }

  private async prune(now: number): Promise<void> {
    const nonces = await this.ctx.storage.list<NonceRecord>({ prefix: "nonce:" });
    for (const [key, rec] of nonces) {
      if (now > rec.exp && rec.used) await this.ctx.storage.delete(key);
    }
    const rates = await this.ctx.storage.list<RateRecord>({ prefix: "rl:" });
    for (const [key, rec] of rates) {
      if (now > rec.resetAt + 60 * 60 * 1000) await this.ctx.storage.delete(key);
    }
  }
}

export async function leadingZeroBits(text: string): Promise<number> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
  );
  let bits = 0;
  for (const byte of digest) {
    if (byte === 0) {
      bits += 8;
      continue;
    }
    bits += Math.clz32(byte) - 24;
    break;
  }
  return bits;
}

export function randomHex(bytes: number): string {
  const b = new Uint8Array(bytes);
  crypto.getRandomValues(b);
  return [...b].map(x => x.toString(16).padStart(2, "0")).join("");
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

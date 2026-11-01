import type { Cell, ICodeCellModel } from '@jupyterlab/cells';
import type { Content } from './types';

const ANSI = /\u001b\[[0-9;]*[A-Za-z]/g;

export interface CollectedOutputs {
  content: Content[];
  raw: unknown[];
  hasError: boolean;
  errorText?: string;
}

/**
 * Turn a code cell's nbformat outputs into MCP content blocks: text for
 * stream/results, `image` blocks for PNG/JPEG (downscaled if they exceed the
 * budget), and a readable error block.
 */
export async function collectOutputs(
  cell: Cell,
  maxImageBytes: number,
  maxText = 12_000,
): Promise<CollectedOutputs> {
  if (cell.model.type !== 'code') {
    return { content: [], raw: [], hasError: false };
  }

  const model = cell.model as ICodeCellModel;
  const raw = model.outputs.toJSON();
  const textParts: string[] = [];
  const images: Content[] = [];
  const errors: string[] = [];
  const maxB64 = Math.floor(maxImageBytes * 1.37);

  for (const out of raw) {
    if (out.output_type === 'stream') {
      textParts.push(stripAnsi(String(out.text ?? '')));
      continue;
    }
    if (out.output_type === 'error') {
      const traceback = Array.isArray(out.traceback)
        ? out.traceback.map(entry => stripAnsi(String(entry))).filter(Boolean)
        : [];
      errors.push(`${String(out.ename)}: ${String(out.evalue)}`);
      if (traceback.length) {
        textParts.push(traceback.slice(-8).join('\n'));
      }
      continue;
    }
    if (
      out.output_type === 'execute_result' ||
      out.output_type === 'display_data' ||
      out.output_type === 'update_display_data'
    ) {
      const data = (out.data ?? {}) as Record<string, unknown>;
      const png = typeof data['image/png'] === 'string' ? data['image/png'] : null;
      const jpeg = typeof data['image/jpeg'] === 'string' ? data['image/jpeg'] : null;
      if (png || jpeg) {
        const mimeType = png ? 'image/png' : 'image/jpeg';
        const shrunk = await shrinkImage(png ?? jpeg!, maxB64);
        if (shrunk) {
          images.push({ type: 'image', data: shrunk, mimeType });
        } else {
          textParts.push(`[image omitted: exceeds ${maxImageBytes} bytes]`);
        }
      }
      const plain = data['text/plain'];
      if (typeof plain === 'string' && plain.trim()) {
        textParts.push(plain);
      }
    }
  }

  const text = truncate(textParts.join('\n').trim(), maxText);
  const content: Content[] = [];
  if (text) {
    content.push({ type: 'text', text });
  }
  content.push(...images);

  const hasError = errors.length > 0;
  if (hasError) {
    const errText = truncate(errors.join('\n'), 4_000);
    if (!content.some(c => c.type === 'text')) {
      content.unshift({ type: 'text', text: errText });
    }
  }
  if (content.length === 0) {
    content.push({ type: 'text', text: '(no output)' });
  }

  return {
    content,
    raw: sanitizeOutputs(raw),
    hasError,
    errorText: hasError ? errors.join('; ') : undefined,
  };
}

/** Strip base64 payloads so `structuredContent` stays readable. */
export function sanitizeOutputs(raw: unknown[]): unknown[] {
  const walk = (value: unknown): unknown => {
    if (typeof value === 'string') {
      return value.length > 512 ? `<${value.length} chars>` : value;
    }
    if (Array.isArray(value)) {
      return value.map(walk);
    }
    if (value && typeof value === 'object') {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        out[k] = /^image\//.test(k) ? `<image ${typeof v === 'string' ? v.length : '?'} chars>` : walk(v);
      }
      return out;
    }
    return value;
  };
  return raw.map(walk);
}

function stripAnsi(text: string): string {
  return text.replace(ANSI, '');
}

function truncate(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`;
}

async function shrinkImage(base64: string, maxB64: number): Promise<string | null> {
  if (base64.length <= maxB64) {
    return base64;
  }
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') {
    return null;
  }
  try {
    const blob = await (await fetch(`data:image/png;base64,${base64}`)).blob();
    const bitmap = await createImageBitmap(blob);
    let scale = Math.min(1, Math.sqrt(maxB64 / base64.length));
    for (let attempt = 0; attempt < 4; attempt++) {
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return null;
      }
      ctx.drawImage(bitmap, 0, 0, width, height);
      const out = await canvas.convertToBlob({ type: 'image/png' });
      const data = await blobToBase64(out);
      if (data.length <= maxB64) {
        return data;
      }
      scale *= 0.7;
    }
    return null;
  } catch {
    return null;
  }
}

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

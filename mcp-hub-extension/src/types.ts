export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageContent {
  type: 'image';
  data: string;
  mimeType: string;
}

export type Content = TextContent | ImageContent;

export interface JobError {
  code: string;
  message: string;
  hint?: string;
}

export interface JobResult {
  ok: boolean;
  content: Content[];
  structured?: Record<string, unknown>;
  error?: JobError;
}

export interface JobFrame {
  type: 'job';
  id: string;
  kind: string;
  args: Record<string, unknown>;
}

export interface CancelFrame {
  type: 'cancel';
  id: string;
}

export type InboundFrame = JobFrame | CancelFrame | { type: 'hello' };

export interface HubSettings {
  hubUrl: string;
  pairId: string;
  pairToken: string;
  autoConnect: boolean;
  logLimit: number;
  maxImageBytes: number;
}

export type ConnState = 'offline' | 'connecting' | 'connected';

export interface PairInfo {
  pair_id: string;
  token: string;
  mcp_url: string;
  mcp_url_bearer?: string;
  runner_ws?: string;
}

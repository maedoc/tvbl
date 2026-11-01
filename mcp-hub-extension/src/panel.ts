import { Widget } from '@lumino/widgets';
import type { ConnState, HubSettings } from './types';

export interface HubPanelOptions {
  onPair: () => void;
  onDisconnect: () => void;
  logLimit: number;
}

/**
 * Right-side panel: connection state, the MCP URL to paste into a chat client,
 * pairing controls and an activity log.
 */
export class HubPanel extends Widget {
  private readonly _dot: HTMLElement;
  private readonly _state: HTMLElement;
  private readonly _detail: HTMLElement;
  private readonly _mcp: HTMLInputElement;
  private readonly _log: HTMLElement;
  private readonly _hub: HTMLElement;
  private readonly _pairButton: HTMLButtonElement;
  private readonly _disconnectButton: HTMLButtonElement;
  private readonly _copyButton: HTMLButtonElement;

  constructor(private readonly opts: HubPanelOptions) {
    super();
    this.id = 'tvbl-mcp-hub';
    this.title.label = 'MCP Hub';
    this.title.closable = true;
    this.addClass('tvbl-mcp-panel');

    this._dot = document.createElement('span');
    this._dot.className = 'tvbl-mcp-dot tvbl-mcp-offline';

    this._state = document.createElement('span');
    this._state.className = 'tvbl-mcp-state';
    this._state.textContent = 'offline';

    const status = document.createElement('div');
    status.className = 'tvbl-mcp-row tvbl-mcp-status-row';
    status.append(this._dot, this._state);

    this._detail = document.createElement('div');
    this._detail.className = 'tvbl-mcp-detail';
    this._detail.textContent = 'Not paired yet.';

    this._hub = document.createElement('code');
    this._hub.className = 'tvbl-mcp-huburl';

    this._pairButton = document.createElement('button');
    this._pairButton.className = 'jp-mod-accept tvbl-mcp-button';
    this._pairButton.textContent = 'Pair this tab';
    this._pairButton.onclick = () => this.opts.onPair();

    this._disconnectButton = document.createElement('button');
    this._disconnectButton.className = 'jp-mod-reject tvbl-mcp-button';
    this._disconnectButton.textContent = 'Disconnect';
    this._disconnectButton.onclick = () => this.opts.onDisconnect();

    this._copyButton = document.createElement('button');
    this._copyButton.className = 'tvbl-mcp-button';
    this._copyButton.textContent = 'Copy URL';
    this._copyButton.onclick = () => {
      void navigator.clipboard?.writeText(this._mcp.value);
      this.log('info', 'MCP URL copied');
    };

    const buttons = document.createElement('div');
    buttons.className = 'tvbl-mcp-row';
    buttons.append(this._pairButton, this._disconnectButton);

    const label = document.createElement('div');
    label.className = 'tvbl-mcp-label';
    label.textContent = 'MCP server URL';

    this._mcp = document.createElement('input');
    this._mcp.className = 'tvbl-mcp-mcpurl';
    this._mcp.readOnly = true;
    this._mcp.placeholder = 'Pair this tab, then paste this into ChatGPT / Claude / Gemini';
    this._mcp.onclick = () => this._mcp.select();

    const urlRow = document.createElement('div');
    urlRow.className = 'tvbl-mcp-row';
    urlRow.append(this._mcp, this._copyButton);

    const hint = document.createElement('div');
    hint.className = 'tvbl-mcp-hint';
    hint.innerHTML =
      'Add this URL as a custom MCP connector. Keep this tab open: it is the compute. ' +
      'Pairing creates a fresh, unguessable token; <em>Disconnect</em> stops the runner.';

    const logLabel = document.createElement('div');
    logLabel.className = 'tvbl-mcp-label';
    logLabel.textContent = 'Activity';

    this._log = document.createElement('div');
    this._log.className = 'tvbl-mcp-log';

    const node = document.createElement('div');
    node.className = 'tvbl-mcp-body';
    node.append(status, this._detail, buttons, label, urlRow, hint, logLabel, this._log);
    this.node.appendChild(node);
  }

  setState(state: ConnState, detail?: string): void {
    this._dot.className = `tvbl-mcp-dot tvbl-mcp-${state}`;
    this._state.textContent = state;
    if (detail) {
      this._detail.textContent = detail;
    }
  }

  setConfig(cfg: HubSettings): void {
    this._hub.textContent = cfg.hubUrl || '(no hub URL)';
    if (cfg.pairId) {
      this._detail.textContent = `pair ${cfg.pairId}`;
    }
  }

  setMcpUrl(url: string): void {
    this._mcp.value = url;
    if (url) {
      this._detail.textContent = 'Ready. Paste the MCP URL into your chat client.';
    }
  }

  log(kind: string, text: string): void {
    const entry = document.createElement('div');
    entry.className = `tvbl-mcp-entry tvbl-mcp-${kind}`;
    const time = new Date().toLocaleTimeString();
    entry.textContent = `${time}  ${text}`;
    this._log.prepend(entry);
    while (this._log.childElementCount > this.opts.logLimit) {
      this._log.lastElementChild?.remove();
    }
  }

  setBusy(busy: boolean): void {
    this._pairButton.disabled = busy;
    this._disconnectButton.disabled = busy;
    this._pairButton.textContent = busy ? 'Pairing…' : 'Pair this tab';
  }
}

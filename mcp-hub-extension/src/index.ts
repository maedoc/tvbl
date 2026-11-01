import type { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, ISessionContextDialogs } from '@jupyterlab/apputils';
import { PageConfig } from '@jupyterlab/coreutils';
import { IDocumentManager } from '@jupyterlab/docmanager';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { IStatusBar } from '@jupyterlab/statusbar';
import { ITranslator } from '@jupyterlab/translation';
import { Widget } from '@lumino/widgets';

import { HubClient, messageOf } from './client';
import { HubPanel } from './panel';
import { ToolDispatcher } from './tools';
import type { HubSettings, JobFrame, JobResult, PairInfo } from './types';

const PLUGIN_ID = '@tvbl/jupyterlite-mcp-hub:plugin';
const DEFAULT_HUB = 'https://tvbl-mcp-hub.marmaduke-woodman.workers.dev';

const COMMANDS = [
  'mcp-hub:open',
  'mcp-hub:pair',
  'mcp-hub:connect',
  'mcp-hub:disconnect',
];

/**
 * JupyterLite runner: registers this tab with the hub, then executes
 * notebook/cell jobs sent by an MCP client.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: PLUGIN_ID,
  description: 'Expose this JupyterLite tab to an MCP hub so chat clients can drive the notebook.',
  autoStart: true,
  requires: [INotebookTracker, IDocumentManager, ITranslator],
  optional: [ISettingRegistry, ICommandPalette, IStatusBar, ISessionContextDialogs],
  activate: async (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    docManager: IDocumentManager,
    translator: ITranslator,
    settingRegistry: ISettingRegistry | null,
    palette: ICommandPalette | null,
    statusBar: IStatusBar | null,
    sessionDialogs: ISessionContextDialogs | null,
  ): Promise<void> => {
    if (!settingRegistry) {
      console.warn('mcp-hub: settings registry unavailable; plugin disabled');
      return;
    }
    const settings = await settingRegistry.load(PLUGIN_ID);
    let cfg = readSettings(settings);

    const makeDispatcher = (): ToolDispatcher =>
      new ToolDispatcher(app, tracker, docManager, sessionDialogs ?? null, translator, cfg.maxImageBytes);
    let dispatcher = makeDispatcher();

    const client = new HubClient(cfg, {
      onState: (state, detail) => {
        panel.setState(state, detail);
        if (statusWidget) {
          statusWidget.node.textContent = `MCP ${state}`;
          statusWidget.node.className = `tvbl-mcp-statusbar tvbl-mcp-${state}`;
        }
      },
      onLog: (kind, text) => panel.log(kind, text),
      onJob: (job: JobFrame): Promise<JobResult> => dispatcher.dispatch(job),
      onCancel: () => undefined,
    });

    let statusWidget: Widget | null = null;
    if (statusBar) {
      statusWidget = new Widget();
      statusWidget.node.className = 'tvbl-mcp-statusbar tvbl-mcp-offline';
      statusWidget.node.textContent = 'MCP offline';
      statusBar.registerStatusItem(PLUGIN_ID, { item: statusWidget, align: 'right', rank: 1 });
    }

    const panel = new HubPanel({
      onPair: () => {
        void pair();
      },
      onDisconnect: () => client.disconnect(),
      logLimit: cfg.logLimit,
    });
    app.shell.add(panel, 'right', { rank: 900 });
    panel.setConfig(cfg);

    let pairing = false;
    let lastKey = keyOf(cfg);

    const apply = (next: HubSettings): void => {
      cfg = next;
      dispatcher = makeDispatcher();
      client.configure(cfg);
      panel.setConfig(cfg);
    };

    settings.changed.connect(() => {
      if (pairing) {
        return;
      }
      const next = readSettings(settings);
      apply(next);
      const key = keyOf(next);
      if (key !== lastKey) {
        lastKey = key;
        if (next.autoConnect) {
          client.connect();
        } else {
          client.disconnect();
        }
      }
    });

    async function pair(): Promise<void> {
      panel.setBusy(true);
      pairing = true;
      try {
        const base = cfg.hubUrl.replace(/\/+$/, '');
        panel.log('info', `pairing with ${base} …`);

        // Optional proof-of-work: the hub sets POW_BITS=0 to disable it, and an
        // older hub answers 404, in which case we pair without it.
        let pow: { nonce: string; solution: string } | undefined;
        const challengeRes = await fetch(`${base}/pair/challenge`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: '{}',
        });
        if (challengeRes.ok) {
          const challenge = (await challengeRes.json()) as { nonce: string | null; bits: number };
          if (challenge.bits > 0 && challenge.nonce) {
            panel.log('info', `solving proof-of-work (${challenge.bits} bits)…`);
            const startedAt = performance.now();
            const solution = await solveProofOfWork(challenge.nonce, challenge.bits);
            pow = { nonce: challenge.nonce, solution };
            panel.log('info', `proof-of-work solved in ${Math.round(performance.now() - startedAt)}ms`);
          }
        } else if (challengeRes.status === 429) {
          const body = (await challengeRes.json().catch(() => ({}))) as { retry_after?: number };
          throw new Error(`too many pairings; retry in ${body.retry_after ?? 'a while'}s`);
        } else if (challengeRes.status !== 404) {
          throw new Error(`pair challenge failed: HTTP ${challengeRes.status}`);
        }

        const res = await fetch(`${base}/pair`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ label: `JupyterLite ${location.host}`, ...(pow ?? {}) }),
        });
        if (!res.ok) {
          throw new Error(`pair failed: HTTP ${res.status} ${await res.text()}`);
        }
        const info = (await res.json()) as PairInfo;
        await settings.set('pairId', info.pair_id);
        await settings.set('pairToken', info.token);
        cfg = readSettings(settings);
        lastKey = keyOf(cfg);
        apply(cfg);
        panel.setMcpUrl(info.mcp_url);
        panel.log('info', `paired as ${info.pair_id}`);
        client.connect();
      } catch (err) {
        panel.log('error', messageOf(err));
      } finally {
        pairing = false;
        panel.setBusy(false);
      }
    }

    app.commands.addCommand('mcp-hub:open', {
      label: 'MCP Hub: Open panel',
      execute: () => {
        app.shell.activateById(panel.id);
      },
    });
    app.commands.addCommand('mcp-hub:pair', {
      label: 'MCP Hub: Pair this tab',
      execute: () => pair(),
    });
    app.commands.addCommand('mcp-hub:connect', {
      label: 'MCP Hub: Reconnect',
      execute: () => client.connect(),
    });
    app.commands.addCommand('mcp-hub:disconnect', {
      label: 'MCP Hub: Disconnect',
      execute: () => client.disconnect(),
    });

    if (palette) {
      for (const command of COMMANDS) {
        palette.addItem({ command, category: 'MCP Hub' });
      }
    }

    if (cfg.pairId && cfg.pairToken) {
      panel.setMcpUrl(`${cfg.hubUrl.replace(/\/+$/, '')}/mcp/${cfg.pairId}`);
      if (cfg.autoConnect) {
        client.connect();
      }
    } else {
      panel.setState('offline', 'not paired');
    }

    void app.restored.then(() => {
      app.shell.activateById(panel.id);
    });
  },
};

function readSettings(settings: ISettingRegistry.ISettings): HubSettings {
  const get = (key: string): unknown => settings.get(key)?.composite;
  const str = (key: string, fallback = ''): string => {
    const value = get(key);
    return typeof value === 'string' ? value : fallback;
  };
  const num = (key: string, fallback: number): number => {
    const value = get(key);
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  };
  const bool = (key: string, fallback: boolean): boolean => {
    const value = get(key);
    return typeof value === 'boolean' ? value : fallback;
  };
  // A settings override wins; otherwise allow the site to point the plugin at a
  // local hub with jupyter-config-data/mcpHubUrl (the schema default would
  // otherwise mask it).
  const hubUser = settings.get('hubUrl')?.user;
  const hubUrl =
    typeof hubUser === 'string' && hubUser
      ? hubUser
      : PageConfig.getOption('mcpHubUrl') || DEFAULT_HUB;

  return {
    hubUrl: hubUrl.replace(/\/+$/, ''),
    pairId: str('pairId'),
    pairToken: str('pairToken'),
    autoConnect: bool('autoConnect', true),
    logLimit: Math.max(20, Math.min(2000, num('logLimit', 200))),
    maxImageBytes: Math.max(10_000, Math.min(2_000_000, num('maxImageBytes', 200_000))),
  };
}

async function solveProofOfWork(nonce: string, bits: number): Promise<string> {
  const encoder = new TextEncoder();
  for (let i = 0; ; i++) {
    const solution = i.toString(36);
    const digest = new Uint8Array(
      await crypto.subtle.digest('SHA-256', encoder.encode(nonce + solution)),
    );
    let have = 0;
    for (const byte of digest) {
      if (byte === 0) {
        have += 8;
        continue;
      }
      have += Math.clz32(byte) - 24;
      break;
    }
    if (have >= bits) {
      return solution;
    }
    if ((i & 0x3ff) === 0 && i > 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}

function keyOf(cfg: HubSettings): string {
  return `${cfg.hubUrl}|${cfg.pairId}|${cfg.pairToken}`;
}

export default plugin;

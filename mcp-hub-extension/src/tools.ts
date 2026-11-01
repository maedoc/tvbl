import type { JupyterFrontEnd } from '@jupyterlab/application';
import type { ISessionContextDialogs } from '@jupyterlab/apputils';
import type { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { PageConfig } from '@jupyterlab/coreutils';
import type { IDocumentManager } from '@jupyterlab/docmanager';
import { NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import type { INotebookTracker } from '@jupyterlab/notebook';
import type { Contents } from '@jupyterlab/services';
import type { ITranslator } from '@jupyterlab/translation';

import { collectOutputs, sanitizeOutputs } from './output';
import type { JobFrame, JobResult } from './types';

const SOURCE_CLIP = 6_000;

/**
 * Executes hub jobs against real JupyterLab APIs: the notebook model for
 * editing, `NotebookActions` for execution (so widget comms and output areas
 * behave exactly as if the user pressed Shift+Enter), and the contents API for
 * files.
 */
export class ToolDispatcher {
  constructor(
    private readonly app: JupyterFrontEnd,
    private readonly tracker: INotebookTracker,
    private readonly docManager: IDocumentManager,
    private readonly dialogs: ISessionContextDialogs | null,
    private readonly translator: ITranslator,
    private readonly maxImageBytes: number,
  ) {}

  async dispatch(frame: JobFrame): Promise<JobResult> {
    const args = (frame.args ?? {}) as Record<string, unknown>;
    switch (frame.kind) {
      case 'site_info':
        return this.siteInfo();
      case 'notebook_list':
        return this.notebookList();
      case 'notebook_read':
        return this.notebookRead(args);
      case 'notebook_create':
        return this.notebookCreate(args);
      case 'notebook_open':
        return this.notebookOpen(args);
      case 'cell_insert':
        return this.cellInsert(args);
      case 'cell_edit':
        return this.cellEdit(args);
      case 'cell_delete':
        return this.cellDelete(args);
      case 'cell_move':
        return this.cellMove(args);
      case 'cell_run':
        return this.cellRun(args);
      case 'notebook_run_all':
        return this.notebookRunAll(args);
      case 'kernel_restart':
        return this.kernelRestart();
      case 'file_list':
        return this.fileList(args);
      case 'file_read':
        return this.fileRead(args);
      case 'file_write':
        return this.fileWrite(args);
      default:
        return fail(`unknown job kind: ${frame.kind}`);
    }
  }

  // ── info / navigation ───────────────────────────────────────────

  private siteInfo(): JobResult {
    const panel = this.tracker.currentWidget;
    const session = panel?.sessionContext;
    const specs = this.app.serviceManager.kernelspecs.specs;
    const info = {
      notebook: panel?.context.path ?? null,
      cells: panel?.content.model?.cells.length ?? 0,
      kernel: session?.session?.kernel?.name ?? null,
      kernel_status: session?.kernelDisplayStatus ?? 'none',
      default_kernel: specs?.default ?? null,
      app: PageConfig.getOption('appName') || 'JupyterLite',
      app_version: PageConfig.getOption('appVersion') || null,
    };
    const lines = [
      `notebook: ${info.notebook ?? '(none open)'}`,
      `cells: ${info.cells}`,
      `kernel: ${info.kernel ?? '(not started)'} (${info.kernel_status})`,
      `default kernel: ${info.default_kernel ?? '(unknown)'}`,
      `${info.app} ${info.app_version ?? ''}`.trim(),
    ];
    return ok(lines.join('\n'), info);
  }

  private async notebookList(): Promise<JobResult> {
    const model = await this.contents.get('', { content: true });
    const entries = ((model.content ?? []) as Contents.IModel[]).map(item => ({
      name: item.name,
      path: item.path,
      type: item.type,
    }));
    const text = entries.length
      ? entries.map(e => `${e.type === 'directory' ? 'dir ' : 'file'} ${e.path}`).join('\n')
      : '(workspace is empty)';
    return ok(text, { entries });
  }

  private async notebookRead(args: Record<string, unknown>): Promise<JobResult> {
    const panel = this.panel(args.path);
    if (!panel) {
      return fail(
        typeof args.path === 'string'
          ? `notebook not open: ${args.path}`
          : 'no notebook open; call notebook_create first',
      );
    }
    const includeOutputs = args.include_outputs !== false;
    const model = panel.content.model;
    if (!model) {
      return fail('notebook has no model');
    }

    const cells = [];
    for (let i = 0; i < model.cells.length; i++) {
      const cell = model.cells.get(i);
      const code = cell.type === 'code' ? (cell as ICodeCellModel) : null;
      const outputs = includeOutputs && code ? sanitizeOutputs(code.outputs.toJSON()) : null;
      cells.push({
        cell_id: cell.id,
        index: i,
        type: cell.type,
        source: clip(cell.sharedModel.getSource(), SOURCE_CLIP),
        execution_count: code?.executionCount ?? null,
        execution_state: code?.executionState ?? null,
        ...(outputs ? { outputs } : {}),
      });
    }

    const summary = [`${panel.context.path}: ${cells.length} cells`]
      .concat(
        cells.map(c => {
          const first = c.source.split('\n')[0]?.slice(0, 70) ?? '';
          const state = c.type === 'code' ? ` [${c.execution_state ?? 'idle'}]` : '';
          return `${c.index}. ${c.cell_id} ${c.type}${state} ${first}`;
        }),
      )
      .join('\n');

    return ok(summary, {
      path: panel.context.path,
      kernel: panel.sessionContext.session?.kernel?.name ?? null,
      cells,
    });
  }

  private async notebookCreate(args: Record<string, unknown>): Promise<JobResult> {
    const requested = typeof args.path === 'string' ? args.path : null;
    const spec = this.kernelSpec(typeof args.kernel === 'string' ? args.kernel : undefined);

    if (requested) {
      const exists = await this.exists(requested);
      if (exists) {
        return fail(`notebook already exists: ${requested}; use notebook_open`);
      }
      const notebook = {
        cells: [],
        metadata: {
          kernelspec: spec,
          language_info: { name: spec.language ?? 'python' },
        },
        nbformat: 4,
        nbformat_minor: 5,
      };
      await this.ensureParent(requested);
      await this.contents.save(requested, { type: 'notebook', format: 'json', content: notebook });
      const widget = this.docManager.open(requested, undefined, { name: spec.name });
      if (!(widget instanceof NotebookPanel)) {
        return fail(`opened ${requested} but it is not a notebook`);
      }
      this.app.shell.activateById(widget.id);
      await widget.context.ready;
      await this.ensureKernel(widget, spec.name);
      const first = widget.content.model?.cells.get(0);
      return ok(
        `created and opened ${requested} with one empty cell ${first?.id ?? '(none)'}; ` +
          `write into it with cell_edit or add more with cell_insert`,
        {
          path: requested,
          kernel: widget.sessionContext.session?.kernel?.name ?? spec.name,
          cell_id: first?.id ?? null,
        },
      );
    }

    await this.app.commands.execute(
      'notebook:create-new',
      typeof args.kernel === 'string' ? { kernelName: args.kernel } : {},
    );
    const panel = this.tracker.currentWidget;
    if (!panel) {
      return fail('notebook:create-new did not create a notebook');
    }
    await panel.context.ready;
    await this.ensureKernel(panel);
    return ok(`created and opened ${panel.context.path}`, {
      path: panel.context.path,
      kernel: panel.sessionContext.session?.kernel?.name ?? null,
    });
  }

  private async notebookOpen(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('missing path');
    }
    const widget = this.docManager.openOrReveal(path);
    if (!widget) {
      return fail(`could not open ${path}`);
    }
    this.app.shell.activateById(widget.id);
    await widget.context.ready;
    if (widget instanceof NotebookPanel) {
      await this.ensureKernel(widget);
    }
    return ok(`opened ${path}`);
  }

  // ── cell editing ────────────────────────────────────────────────

  private cellInsert(args: Record<string, unknown>): JobResult {
    const panel = this.requirePanel(args.notebook);
    const model = panel.content.model;
    if (!model) {
      return fail('notebook has no model');
    }
    const type = args.cell_type === 'markdown' ? 'markdown' : 'code';
    const source = String(args.source ?? '');

    // JupyterLab puts one empty cell in every new notebook. Reuse it instead
    // of leaving an orphan above the model's first cell.
    if (typeof args.index !== 'number' && model.cells.length === 1) {
      const only = model.cells.get(0);
      const code = only.type === 'code' ? (only as ICodeCellModel) : null;
      if (
        code &&
        type === 'code' &&
        only.sharedModel.getSource() === '' &&
        (code.executionCount ?? null) === null
      ) {
        code.sharedModel.setSource(source);
        panel.content.activeCellIndex = 0;
        this.persist(panel);
        return ok(`wrote into the empty cell ${only.id}`, {
          cell_id: only.id,
          index: 0,
          cell_type: type,
          reused: true,
        });
      }
    }

    const index = typeof args.index === 'number' ? args.index : model.cells.length;
    const at = Math.max(0, Math.min(index, model.cells.length));
    model.sharedModel.insertCell(at, {
      cell_type: type,
      source,
      metadata: {},
    });
    panel.content.activeCellIndex = at;
    panel.content.update();
    this.persist(panel);
    const cell = model.cells.get(at);
    return ok(`inserted ${cell.id} at ${at} (${type})`, {
      cell_id: cell.id,
      index: at,
      cell_type: type,
    });
  }

  private cellEdit(args: Record<string, unknown>): JobResult {
    const panel = this.requirePanel(args.notebook);
    const cell = this.findCell(panel, args.cell_id);
    cell.model.sharedModel.setSource(String(args.source ?? ''));
    this.persist(panel);
    return ok(`edited ${cell.model.id}`, { cell_id: cell.model.id });
  }

  private cellDelete(args: Record<string, unknown>): JobResult {
    const panel = this.requirePanel(args.notebook);
    const cell = this.findCell(panel, args.cell_id);
    const index = panel.content.widgets.indexOf(cell);
    panel.content.model?.sharedModel.deleteCell(index);
    this.persist(panel);
    return ok(`deleted ${cell.model.id}`, { cell_id: cell.model.id, index });
  }

  private cellMove(args: Record<string, unknown>): JobResult {
    const panel = this.requirePanel(args.notebook);
    const cell = this.findCell(panel, args.cell_id);
    const model = panel.content.model;
    if (!model) {
      return fail('notebook has no model');
    }
    const from = panel.content.widgets.indexOf(cell);
    const to = Math.max(0, Math.min(Number(args.index ?? 0), model.cells.length - 1));
    model.sharedModel.moveCell(from, to);
    this.persist(panel);
    return ok(`moved ${cell.model.id} to ${to}`, { cell_id: cell.model.id, index: to });
  }

  // ── execution ───────────────────────────────────────────────────

  private async cellRun(args: Record<string, unknown>): Promise<JobResult> {
    const panel = this.requirePanel(args.notebook);
    const cell = this.findCell(panel, args.cell_id);

    const started = performance.now();
    try {
      await this.ensureKernel(panel);
    } catch (err) {
      return fail(`kernel not available: ${messageOf(err)}`);
    }

    panel.content.activeCellIndex = panel.content.widgets.indexOf(cell);
    void panel.content.scrollToCell(cell, 'center').catch(() => undefined);

    let returned = false;
    let thrown: string | undefined;
    try {
      returned = await NotebookActions.runCells(
        panel.content,
        [cell],
        panel.sessionContext,
        this.dialogs ?? undefined,
        this.translator,
      );
    } catch (err) {
      thrown = messageOf(err);
    }

    const collected = await collectOutputs(cell, this.maxImageBytes);
    const duration = Math.round(performance.now() - started);
    const code = cell.model.type === 'code' ? (cell.model as ICodeCellModel) : null;
    const content = [...collected.content];
    if (thrown && !collected.hasError) {
      content.unshift({ type: 'text', text: `execution failed: ${thrown}` });
    }

    this.persist(panel);

    return {
      ok: returned && !thrown && !collected.hasError,
      content,
      structured: {
        cell_id: cell.model.id,
        duration_ms: duration,
        execution_count: code?.executionCount ?? null,
        outputs: collected.raw,
      },
      error: collected.errorText ?? thrown,
    };
  }

  private async notebookRunAll(args: Record<string, unknown>): Promise<JobResult> {
    const panel = this.requirePanel(args.path ?? args.notebook);
    const started = performance.now();
    try {
      await this.ensureKernel(panel);
    } catch (err) {
      return fail(`kernel not available: ${messageOf(err)}`);
    }

    let returned = false;
    let thrown: string | undefined;
    const errors: Array<{ cell_id: string; ename: string; evalue: string }> = [];

    try {
      returned = await NotebookActions.runAll(
        panel.content,
        panel.sessionContext,
        this.dialogs ?? undefined,
        this.translator,
      );
    } catch (err) {
      thrown = messageOf(err);
    }

    const model = panel.content.model;
    if (model) {
      for (let i = 0; i < model.cells.length; i++) {
        const cell = model.cells.get(i);
        if (cell.type !== 'code') {
          continue;
        }
        for (const out of (cell as ICodeCellModel).outputs.toJSON()) {
          if (out.output_type === 'error') {
            errors.push({
              cell_id: cell.id,
              ename: String(out.ename),
              evalue: String(out.evalue),
            });
          }
        }
      }
    }

    const duration = Math.round(performance.now() - started);
    const summary =
      `ran ${model?.cells.length ?? 0} cells in ${duration}ms; ` +
      (errors.length ? `${errors.length} error(s): ${errors.map(e => `${e.ename}: ${e.evalue}`).join('; ')}` : 'no errors');
    return {
      ok: returned && !thrown && errors.length === 0,
      content: [{ type: 'text', text: summary }],
      structured: { duration_ms: duration, errors, cells: model?.cells.length ?? 0 },
      error: thrown ?? (errors.length ? errors.map(e => `${e.ename}: ${e.evalue}`).join('; ') : undefined),
    };
  }

  private async kernelRestart(): Promise<JobResult> {
    const panel = this.requirePanel();
    try {
      await panel.sessionContext.restartKernel();
    } catch (err) {
      return fail(`restart failed: ${messageOf(err)}`);
    }
    return ok('kernel restarted; Python state is gone, notebook cells remain', {
      kernel: panel.sessionContext.session?.kernel?.name ?? null,
      status: panel.sessionContext.kernelDisplayStatus,
    });
  }

  // ── files ───────────────────────────────────────────────────────

  private async fileList(args: Record<string, unknown>): Promise<JobResult> {
    const dir = typeof args.path === 'string' ? args.path : '';
    const model = await this.contents.get(dir, { content: true });
    const items = (Array.isArray(model.content) ? model.content : []) as Contents.IModel[];
    const entries = items.map(i => ({ name: i.name, path: i.path, type: i.type, size: i.size ?? null }));
    return ok(entries.map(e => `${e.type === 'directory' ? 'dir ' : 'file'} ${e.path}`).join('\n') || '(empty)', {
      path: dir,
      entries,
    });
  }

  private async fileRead(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('missing path');
    }
    const model = await this.contents.get(path, { content: true, type: 'file', format: 'text' });
    const text = typeof model.content === 'string' ? model.content : '';
    const max = typeof args.max_chars === 'number' ? args.max_chars : 20_000;
    return ok(clip(text, max), { path, size: text.length });
  }

  private async fileWrite(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('missing path');
    }
    await this.ensureParent(path);
    await this.contents.save(path, {
      type: 'file',
      format: 'text',
      content: String(args.content ?? ''),
    });
    return ok(`wrote ${path}`, { path });
  }

  // ── helpers ─────────────────────────────────────────────────────

  private get contents(): Contents.IManager {
    return this.app.serviceManager.contents;
  }

  private panel(path?: unknown): NotebookPanel | null {
    if (typeof path === 'string' && path) {
      return this.tracker.find(p => p.context.path === path) ?? null;
    }
    return this.tracker.currentWidget;
  }

  private requirePanel(path?: unknown): NotebookPanel {
    const panel = this.panel(path);
    if (!panel) {
      throw new Error(
        typeof path === 'string' && path
          ? `notebook not open: ${path}`
          : 'no notebook open; call notebook_create first',
      );
    }
    return panel;
  }

  private findCell(panel: NotebookPanel, rawId: unknown): Cell {
    const id = String(rawId ?? '');
    const cell = panel.content.widgets.find(c => c.model.id === id);
    if (!cell) {
      throw new Error(`no such cell_id: ${id}; call notebook_read to list cells`);
    }
    return cell;
  }

  private async ensureKernel(panel: NotebookPanel, kernelName?: string): Promise<void> {
    if (!panel.sessionContext.session) {
      // `initialize()` resolves with `true` when the user must pick a kernel,
      // and rejects (rather than returning false) on a real failure.
      const needsSelection = await panel.sessionContext.initialize().catch(() => true);
      if (needsSelection && !panel.sessionContext.session) {
        await this.app.serviceManager.kernelspecs.ready;
        const preferred =
          kernelName ??
          (panel.context.model?.metadata?.kernelspec?.name as string | undefined) ??
          this.kernelSpec().name;
        const kernel = await panel.sessionContext
          .changeKernel({ name: preferred })
          .catch(() => null);
        if (!kernel && !panel.sessionContext.session) {
          throw new Error(`the kernel '${preferred}' did not start`);
        }
      }
    }
    await panel.sessionContext.ready;
    if (!panel.sessionContext.session) {
      throw new Error('no kernel session after start');
    }
  }

  private kernelSpec(name?: string): { name: string; display_name: string; language: string } {
    const specs = this.app.serviceManager.kernelspecs.specs;
    const name2 = name || specs?.default || 'python3';
    const spec = specs?.kernelspecs?.[name2]?.spec as
      | { display_name?: string; language?: string }
      | undefined;
    return {
      name: name2,
      display_name: spec?.display_name ?? name2,
      language: spec?.language ?? 'python',
    };
  }

  private async exists(path: string): Promise<boolean> {
    try {
      await this.contents.get(path);
      return true;
    } catch {
      return false;
    }
  }

  private async ensureParent(path: string): Promise<void> {
    const parts = path.split('/').slice(0, -1).filter(Boolean);
    let current = '';
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      if (!(await this.exists(current))) {
        try {
          await this.contents.save(current, { type: 'directory' });
        } catch {
          // racing creation is fine
        }
      }
    }
  }

  private persist(panel: NotebookPanel): void {
    void panel.context.save().catch(err => {
      console.warn('mcp-hub: save failed', err);
    });
  }
}

function ok(text: string, structured?: Record<string, unknown>): JobResult {
  return { ok: true, content: [{ type: 'text', text }], structured };
}

function fail(text: string): JobResult {
  return { ok: false, content: [{ type: 'text', text }], error: text };
}

function clip(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`;
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

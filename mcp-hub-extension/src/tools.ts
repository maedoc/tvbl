import type { JupyterFrontEnd } from '@jupyterlab/application';
import type { ISessionContextDialogs } from '@jupyterlab/apputils';
import type { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { PageConfig } from '@jupyterlab/coreutils';
import type { IDocumentManager } from '@jupyterlab/docmanager';
import { NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import type { INotebookTracker } from '@jupyterlab/notebook';
import type { Contents } from '@jupyterlab/services';
import type { ITranslator } from '@jupyterlab/translation';

import { collectOutputs, sanitizeOutputs, type CollectedImage } from './output';
import type { JobFrame, JobResult } from './types';

const SOURCE_CLIP = 6_000;
const OUTPUT_CLIP = 12_000;

class ToolError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly hint?: string,
  ) {
    super(message);
  }
}

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
    try {
      return await this.run(frame);
    } catch (err) {
      if (err instanceof ToolError) {
        return fail(err.code, err.message, err.hint);
      }
      console.error('mcp-hub: unhandled tool error', err);
      return fail('tool_error', messageOf(err), 'this is a runner bug; check the browser console');
    }
  }

  private async run(frame: JobFrame): Promise<JobResult> {
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
      case 'notebook_figures':
        return this.notebookFigures();
      case 'kernel_restart':
        return this.kernelRestart();
      case 'file_list':
        return this.fileList(args);
      case 'file_read':
        return this.fileRead(args);
      case 'file_write':
        return this.fileWrite(args);
      default:
        return fail('unknown_tool', `unknown job kind: ${frame.kind}`, 'this hub is newer than the runner; reload the page');
    }
  }

  // ── info / navigation ───────────────────────────────────────────

  private siteInfo(): JobResult {
    const panel = this.tracker.currentWidget;
    const session = panel?.sessionContext;
    const specs = this.app.serviceManager.kernelspecs.specs;
    const kernels = specs ? Object.keys(specs.kernelspecs) : [];
    const info = {
      notebook: panel?.context.path ?? null,
      cells: panel?.content.model?.cells.length ?? 0,
      kernel: session?.session?.kernel?.name ?? null,
      kernel_status: session?.kernelDisplayStatus ?? 'none',
      default_kernel: specs?.default ?? null,
      kernels,
      app: PageConfig.getOption('appName') || 'JupyterLite',
      app_version: PageConfig.getOption('appVersion') || null,
    };
    const lines = [
      `notebook: ${info.notebook ?? '(none open)'}`,
      `cells: ${info.cells}`,
      `kernel: ${info.kernel ?? '(not started)'} (${info.kernel_status})`,
      `default_kernel: ${info.default_kernel ?? '(unknown)'}`,
      `kernels: ${kernels.join(', ') || '(none)'}`,
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
        'no_notebook',
        typeof args.path === 'string' ? `notebook not open: ${args.path}` : 'no notebook is open',
        'call notebook_open or notebook_create first',
      );
    }
    const includeOutputs = args.include_outputs !== false;
    const maxCell = typeof args.max_cell_chars === 'number' ? args.max_cell_chars : SOURCE_CLIP;
    const model = panel.content.model;
    if (!model) {
      return fail('not_found', 'the notebook has no model');
    }

    const cells = [];
    let anyTruncated = false;
    for (let i = 0; i < model.cells.length; i++) {
      const cell = model.cells.get(i);
      const code = cell.type === 'code' ? (cell as ICodeCellModel) : null;
      const source = cell.sharedModel.getSource();
      const clipped = clip(source, maxCell);
      if (clipped.length !== source.length) {
        anyTruncated = true;
      }
      cells.push({
        cell_id: cell.id,
        index: i,
        type: cell.type,
        source: clipped,
        source_chars: source.length,
        source_truncated: clipped.length !== source.length,
        execution_count: code?.executionCount ?? null,
        execution_state: code?.executionState ?? null,
        ...(includeOutputs && code
          ? { outputs: sanitizeOutputs(code.outputs.toJSON()) }
          : {}),
      });
    }

    const summary = [
      `${panel.context.path}: ${cells.length} cells` +
        (includeOutputs ? '' : ' (outputs omitted)') +
        (anyTruncated ? ` (sources capped at ${maxCell} chars)` : ''),
    ]
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
      truncated: anyTruncated,
    });
  }

  private async notebookCreate(args: Record<string, unknown>): Promise<JobResult> {
    await this.app.serviceManager.kernelspecs.ready;
    const requested = typeof args.path === 'string' ? args.path : null;
    const spec = this.kernelSpec(typeof args.kernel === 'string' ? args.kernel : undefined);

    if (requested && (await this.exists(requested))) {
      if (args.if_exists === 'error') {
        return fail(
          'already_exists',
          `notebook already exists: ${requested}`,
          'call notebook_open to use it, or pass if_exists="open"',
        );
      }
      const widget = this.docManager.openOrReveal(requested);
      if (!(widget instanceof NotebookPanel)) {
        return fail('not_found', `could not open existing notebook ${requested}`);
      }
      this.app.shell.activateById(widget.id);
      await widget.context.ready;
      await this.ensureKernel(widget);
      const first = widget.content.model?.cells.get(0);
      return ok(`opened existing ${requested} (created: false)`, {
        path: requested,
        created: false,
        opened: true,
        cell_id: first?.id ?? null,
        kernel: widget.sessionContext.session?.kernel?.name ?? null,
      });
    }

    if (requested) {
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
        return fail('not_found', `opened ${requested} but it is not a notebook`);
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
          created: true,
          opened: true,
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
      return fail('not_found', 'notebook:create-new did not create a notebook');
    }
    await panel.context.ready;
    await this.ensureKernel(panel);
    return ok(`created and opened ${panel.context.path}`, {
      path: panel.context.path,
      created: true,
      opened: true,
      kernel: panel.sessionContext.session?.kernel?.name ?? null,
    });
  }

  private async notebookOpen(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('invalid_argument', 'missing path', 'pass the notebook path, e.g. "demo.ipynb"');
    }
    const widget = this.docManager.openOrReveal(path);
    if (!widget) {
      return fail('not_found', `could not open ${path}`, 'call notebook_list to see available paths');
    }
    this.app.shell.activateById(widget.id);
    await widget.context.ready;
    if (widget instanceof NotebookPanel) {
      await this.ensureKernel(widget);
    }
    return ok(`opened ${path}`, { path, opened: true });
  }

  // ── cell editing ────────────────────────────────────────────────

  private cellInsert(args: Record<string, unknown>): JobResult {
    const panel = this.requirePanel(args.notebook);
    const model = panel.content.model;
    if (!model) {
      return fail('not_found', 'the notebook has no model');
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
      return fail('not_found', 'the notebook has no model');
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
      return fail('kernel_failed', `kernel not available: ${messageOf(err)}`, 'call kernel_restart');
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

    const collected = await collectOutputs(cell, this.maxImageBytes, OUTPUT_CLIP);
    const images = await this.saveImages(panel, cell.model.id, collected.images);
    const duration = Math.round(performance.now() - started);
    const code = cell.model.type === 'code' ? (cell.model as ICodeCellModel) : null;
    const content = [...collected.content];
    if (images.length) {
      content.push({
        type: 'text',
        text: `[figures saved: ${images.map(i => i.path).join(', ')}]`,
      });
    }
    if (thrown && !collected.hasError) {
      content.unshift({ type: 'text', text: `execution failed: ${thrown}` });
    }

    this.persist(panel);

    const failed = !returned || Boolean(thrown) || collected.hasError;
    return {
      ok: !failed,
      content,
      structured: {
        cell_id: cell.model.id,
        duration_ms: duration,
        execution_count: code?.executionCount ?? null,
        outputs: collected.raw,
        images,
      },
      error: failed
        ? {
            code: 'execution_error',
            message: collected.errorText ?? thrown ?? 'the cell did not complete successfully',
            hint: 'the full traceback is in the cell output shown in the notebook',
          }
        : undefined,
    };
  }

  private async notebookRunAll(args: Record<string, unknown>): Promise<JobResult> {
    const panel = this.requirePanel(args.path ?? args.notebook);
    const started = performance.now();
    try {
      await this.ensureKernel(panel);
    } catch (err) {
      return fail('kernel_failed', `kernel not available: ${messageOf(err)}`, 'call kernel_restart');
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
      (errors.length
        ? `${errors.length} error(s): ${errors.map(e => `${e.ename}: ${e.evalue}`).join('; ')}`
        : 'no errors');
    const failed = !returned || Boolean(thrown) || errors.length > 0;
    return {
      ok: !failed,
      content: [{ type: 'text', text: summary }],
      structured: { duration_ms: duration, errors, cells: model?.cells.length ?? 0 },
      error: failed
        ? {
            code: 'execution_error',
            message: errors.map(e => `${e.ename}: ${e.evalue}`).join('; ') || thrown || 'run-all failed',
            hint: 'the failing cell is in the notebook with its traceback',
          }
        : undefined,
    };
  }

  private async notebookFigures(): Promise<JobResult> {
    const panel = this.requirePanel();
    await this.ensureKernel(panel);
    const kernel = panel.sessionContext.session?.kernel;
    if (!kernel) {
      return fail('kernel_failed', 'no kernel session');
    }

    const future = kernel.requestExecute({
      code: FIGURE_CODE,
      silent: false,
      store_history: false,
      user_expressions: {},
      allow_stdin: false,
      stop_on_error: false,
    });

    let out = '';
    future.onIOPub = (msg: any) => {
      const type = msg.header?.msg_type;
      if (type === 'stream') {
        out += String(msg.content?.text ?? '');
      } else if (type === 'error') {
        out += `${String(msg.content?.ename)}: ${String(msg.content?.evalue)}\n`;
      }
    };
    await future.done;

    const marker = out.split('\n').find(line => line.startsWith('MCP_FIGURES:'));
    let figures: unknown[] = [];
    if (marker) {
      try {
        const parsed = JSON.parse(marker.slice('MCP_FIGURES:'.length));
        if (Array.isArray(parsed)) {
          figures = parsed;
        }
      } catch {
        figures = [];
      }
    }

    if (figures.length === 0) {
      return ok(
        'No matplotlib figures are alive in the kernel. Run the plotting cell first; figures are cleared when the kernel restarts.',
        { figures: [], count: 0 },
      );
    }

    const lines = figures.map((f: any) => {
      const head = `figure ${f.figure} / axes ${f.axes}: title=${JSON.stringify(f.title ?? '')} x=${JSON.stringify(f.xlabel ?? '')} y=${JSON.stringify(f.ylabel ?? '')}`;
      const lineInfo = Array.isArray(f.lines)
        ? f.lines
            .map(
              (l: any) =>
                `${l.label} (${l.n} pts, x ${fmtRange(l.x)}, y ${fmtRange(l.y)})`,
            )
            .join('; ')
        : '';
      const collInfo = Array.isArray(f.collections)
        ? f.collections
            .map((c: any) => `${c.type} (${c.n} pts, x ${fmtRange(c.x)}, y ${fmtRange(c.y)})`)
            .join('; ')
        : '';
      return [head, lineInfo, collInfo].filter(Boolean).join(' | ');
    });

    return ok(
      `matplotlib figures (text summary; run the cell again to redraw):\n${lines.join('\n')}`,
      { figures, count: figures.length },
    );
  }

  private async kernelRestart(): Promise<JobResult> {
    const panel = this.requirePanel();
    try {
      await panel.sessionContext.restartKernel();
    } catch (err) {
      return fail('kernel_failed', `restart failed: ${messageOf(err)}`);
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
    const entries = items.map(i => ({
      name: i.name,
      path: i.path,
      type: i.type,
      size: i.size ?? null,
    }));
    return ok(
      entries.map(e => `${e.type === 'directory' ? 'dir ' : 'file'} ${e.path}`).join('\n') || '(empty)',
      { path: dir, entries },
    );
  }

  private async fileRead(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('invalid_argument', 'missing path', 'pass the file path, e.g. "README.md"');
    }
    const max = typeof args.max_chars === 'number' ? args.max_chars : 20_000;

    // The contents API defaults to text; asking for base64 on known binary
    // extensions avoids returning mojibake for figures and data files.
    const binary = /\.(png|jpe?g|gif|webp|pdf|zip|gz|npy|npz|h5|hdf5|so|wasm|nbc|nbi)$/i.test(path);
    let text = '';
    const encoding: 'utf8' | 'base64' = binary ? 'base64' : 'utf8';
    let size: number | null = null;
    try {
      const model = await this.contents.get(path, {
        content: true,
        type: 'file',
        format: binary ? 'base64' : 'text',
      });
      size = model.size ?? null;
      text = typeof model.content === 'string' ? model.content : '';
    } catch {
      throw new ToolError(
        'not_found',
        `could not read ${path}`,
        'call file_list to see available files',
      );
    }

    const body = clip(text, max);
    return ok(body || '(empty file)', {
      path,
      chars: text.length,
      size,
      encoding,
      truncated: body.length !== text.length,
      content: body,
    });
  }

  private async fileWrite(args: Record<string, unknown>): Promise<JobResult> {
    const path = String(args.path ?? '');
    if (!path) {
      return fail('invalid_argument', 'missing path', 'pass the file path, e.g. "notes/todo.md"');
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
      throw new ToolError(
        'no_notebook',
        typeof path === 'string' && path ? `notebook not open: ${path}` : 'no notebook is open',
        'call notebook_open or notebook_create first',
      );
    }
    return panel;
  }

  private findCell(panel: NotebookPanel, rawId: unknown): Cell {
    const id = String(rawId ?? '');
    if (!id) {
      throw new ToolError('invalid_argument', 'missing cell_id', 'call notebook_read to list cell ids');
    }
    const cell = panel.content.widgets.find(c => c.model.id === id);
    if (!cell) {
      throw new ToolError(
        'unknown_cell',
        `no such cell_id: ${id}`,
        'call notebook_read to list the current cells; ids change if the notebook was recreated',
      );
    }
    return cell;
  }

  private async ensureKernel(panel: NotebookPanel, kernelName?: string): Promise<void> {
    // The xeus/pyodide kernelspecs arrive with the service worker; starting a
    // session before they are registered fails with a generic error.
    await this.app.serviceManager.kernelspecs.ready;

    if (!panel.sessionContext.session) {
      // `initialize()` resolves with `true` when the user must pick a kernel,
      // and rejects (rather than returning false) on a real failure.
      const needsSelection = await panel.sessionContext.initialize().catch(() => true);
      if (needsSelection && !panel.sessionContext.session) {
        const preferred = this.pickKernelName(
          kernelName ?? (panel.context.model?.metadata?.kernelspec?.name as string | undefined),
        );
        const kernel = await panel.sessionContext
          .changeKernel({ name: preferred })
          .catch(() => null);
        if (!kernel && !panel.sessionContext.session) {
          // a preempted start can leave the second attempt returning null
          await new Promise(resolve => setTimeout(resolve, 1_000));
          if (!panel.sessionContext.session) {
            throw new ToolError(
              'kernel_failed',
              `the kernel '${preferred}' did not start (available: ${this.kernelNames().join(', ') || 'none'})`,
              'call site_info to see the kernels, then kernel_restart',
            );
          }
        }
      }
    }
    await panel.sessionContext.ready;
    if (!panel.sessionContext.session) {
      throw new ToolError('kernel_failed', 'no kernel session after start', 'call kernel_restart');
    }
  }

  private kernelNames(): string[] {
    const specs = this.app.serviceManager.kernelspecs.specs;
    return specs ? Object.keys(specs.kernelspecs) : [];
  }

  private pickKernelName(requested?: string): string {
    const specs = this.app.serviceManager.kernelspecs.specs;
    const available = specs ? Object.keys(specs.kernelspecs) : [];
    if (requested && (available.length === 0 || available.includes(requested))) {
      return requested;
    }
    if (specs?.default && (available.length === 0 || available.includes(specs.default))) {
      return specs.default;
    }
    return available[0] ?? requested ?? 'python';
  }

  private kernelSpec(name?: string): { name: string; display_name: string; language: string } {
    const specs = this.app.serviceManager.kernelspecs.specs;
    const name2 = this.pickKernelName(name);
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

  private async saveImages(
    panel: NotebookPanel,
    cellId: string,
    images: CollectedImage[],
  ): Promise<Array<{ path: string; mimeType: string; bytes: number }>> {
    if (!images.length) {
      return [];
    }
    const stem = (panel.context.path.split('/').pop() ?? 'notebook').replace(/\.ipynb$/, '');
    const dir = `_mcp/${stem}`;
    await this.ensureParent(`${dir}/x`);
    const saved: Array<{ path: string; mimeType: string; bytes: number }> = [];
    for (const [i, image] of images.slice(0, 8).entries()) {
      const ext = image.mimeType === 'image/jpeg' ? 'jpg' : 'png';
      const path = `${dir}/${cellId}-${i}.${ext}`;
      try {
        await this.contents.save(path, {
          type: 'file',
          format: 'base64',
          content: image.data,
        });
        saved.push({ path, mimeType: image.mimeType, bytes: image.bytes });
      } catch (err) {
        console.warn('mcp-hub: could not save figure', path, err);
      }
    }
    return saved;
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

function fail(code: string, text: string, hint?: string): JobResult {
  return {
    ok: false,
    content: [{ type: 'text', text: hint ? `${text}\n${hint}` : text }],
    error: { code, message: text, hint },
  };
}

function clip(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`;
}

function fmtRange(range: unknown): string {
  if (!Array.isArray(range) || range.length !== 2) {
    return '?';
  }
  const [a, b] = range as [number, number];
  const f = (n: number) => (Math.abs(n) >= 1000 || Math.abs(n) < 0.01 ? n.toExponential(2) : n.toFixed(3));
  return `${f(a)}..${f(b)}`;
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

const FIGURE_CODE = `
import json
try:
    import numpy as np
    import matplotlib.pyplot as plt

    figs = []
    for num in list(plt.get_fignums())[:12]:
        fig = plt.figure(num)
        for i, ax in enumerate(list(fig.axes)[:8]):
            e = {"figure": num, "axes": i, "title": ax.get_title(),
                 "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel()}
            lines = []
            for ln in list(ax.lines)[:12]:
                x = np.asarray(ln.get_xdata(), dtype=float).ravel()
                y = np.asarray(ln.get_ydata(), dtype=float).ravel()
                lines.append({
                    "label": str(ln.get_label()),
                    "n": int(y.size),
                    "x": [float(np.nanmin(x)), float(np.nanmax(x))] if x.size else None,
                    "y": [float(np.nanmin(y)), float(np.nanmax(y))] if y.size else None,
                })
            if lines:
                e["lines"] = lines
            cols = []
            for c in list(ax.collections)[:8]:
                try:
                    a = np.asarray(c.get_offsets(), dtype=float)
                    if a.ndim == 2 and a.shape[0]:
                        cols.append({
                            "type": type(c).__name__,
                            "n": int(a.shape[0]),
                            "x": [float(np.nanmin(a[:, 0])), float(np.nanmax(a[:, 0]))],
                            "y": [float(np.nanmin(a[:, 1])), float(np.nanmax(a[:, 1]))],
                        })
                except Exception:
                    pass
            if cols:
                e["collections"] = cols
            figs.append(e)
    print("MCP_FIGURES:" + json.dumps(figs))
except Exception as exc:
    print("MCP_FIGURES:[]")
    print("MCP_FIGURES_ERR:" + repr(exc))
`;

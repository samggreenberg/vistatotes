import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  HostListener,
  inject,
  input,
  OnInit,
  output,
  signal,
} from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { IconComponent } from '../../icon/icon.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { ActiveDetectorService } from '../../../services/active-detector.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { LabelSessionService } from '../../../services/label-session.service';
import { PluginTemplateVarsService } from '../../../services/plugin-template-vars.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { LabelFilter } from '../../../services/sorting-api.service';
import { ToastService } from '../../../services/toast.service';
import { ImporterField } from '../../../models/api.models';
import { DynamicFieldOptions } from '../../../utils/dynamic-field-options';
import {
  openBlankTab,
  openExternalUrl,
  safeExternalUrl,
} from '../../../utils/external-url';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import type { LabeledElement } from '../../../generated/api-client/models/labeled-element';

export interface ColumnDef {
  key: string;
  label: string;
  enabled: boolean;
  isMetadata?: boolean;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-export-modal',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalComponent,
    FieldHintIconComponent,
    ClipboardCopyComponent,
    IconComponent,
  ],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit {
  readonly detectorName = input('');
  /**
   * Name of the detector whose *persisted* labelset to export, when the
   * caller is naming one rather than exporting the pair it is working in.
   *
   * The Dashboard's row action is the case: it points at a detector in a
   * list, so the answer must not depend on which pair the top-bar pulldown
   * happens to be on — and must not be the whole collection when a Find run
   * has filled that pair's votes with the detector's per-item calls (issue
   * #3639). Empty (the Find and Train callers) keeps the live read.
   */
  readonly labelsetDetectorName = input('');
  /**
   * What the caller is exporting, which decides both the default category and
   * how a one-sided selection is framed.
   *
   * ``detector`` (the Dashboard row action and the train-mode right panel) is
   * exporting *this detector's own labelset* — the thing you re-import to
   * rebuild it. A good-only or bad-only slice of that is not a detector:
   * training rejects a single-class labelset outright (see
   * ``check_both_classes`` in ``vtscore/detectors/training.py``), so the
   * modal opens on ``both`` and says so if you narrow it (issue #3263).
   *
   * ``results`` (the Find view's work-queue and right-panel exports) is
   * exporting a *slice of a scored run* — "the good hits" is the normal ask
   * there, not a trap, so it defaults to whatever slice the caller asked for
   * and the note stays off.
   */
  readonly scope = input<'detector' | 'results'>('detector');
  /**
   * The filter the modal opens on. ``unverified`` / ``verified`` are
   * server-side partitions (by Find ``verified_ids``) that can't be derived
   * client-side, so the modal fetches them with that ``label_filter``; the
   * other values are client-side category filters over the full fetched set.
   *
   * Defaults to ``both`` for the detector-scoped callers, which pass nothing.
   */
  readonly initialFilter = input<LabelFilter>('both');
  readonly closed = output<void>();
  readonly exported = output<void>();

  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private readonly exportersApi = inject(ExportersApiService);
  private readonly labelSession = inject(LabelSessionService);
  private readonly templateVars = inject(PluginTemplateVarsService);
  private readonly sortingApi = inject(SortingApiService);
  private readonly activeDetector = inject(ActiveDetectorService);
  private readonly toast = inject(ToastService);

  // Two eager reads (dataset status + exporter list) load on creation; the
  // labels read is input-derived, so it waits for `ngOnInit` to set
  // `serverFilter` and flip `labelsReady`. All three wrap the generated-client
  // methods so the interceptor chain still applies, replacing the old
  // `ngOnInit` subscribes + teardown plumbing.
  private readonly statusResource = rxResource({
    stream: () => this.datasetsRegistryApi.getStatus(),
  });
  private readonly exportersResource = rxResource({
    stream: () => this.exportersApi.getExporters(),
  });
  private readonly labelsReady = signal(false);
  private readonly labelsResource = rxResource({
    params: () => (this.labelsReady() ? this.serverFilter : undefined),
    stream: () => {
      const labelFilter =
        this.serverFilter === 'both' ? undefined : this.serverFilter;
      return this.sortingApi.exportLabels(false, {
        enrich: true,
        labelFilter,
        detectorName: this.labelsetDetectorName(),
      });
    },
  });

  readonly exporters = computed<ExporterEntry[]>(() =>
    // Two filters, and they answer different questions. `hidden_from_picker`
    // is the plugin author saying "not in a generic list"; `supported_payloads`
    // is the framework saying "this one cannot read a labelset". Offering an
    // exporter that can't is how a labelset email used to go out empty and
    // still report success (#3219).
    (this.exportersResource.value() ?? []).filter(
      (e) =>
        !e.hidden_from_picker &&
        (e.supported_payloads ?? []).includes('labelset'),
    ),
  );

  /** Labels fetched from the server. */
  private readonly labelsList = computed<LabeledElement[]>(
    () => this.labelsResource.value()?.labels ?? [],
  );
  readonly labelsLoaded = computed(
    () =>
      this.labelsResource.hasValue() ||
      this.labelsResource.error() !== undefined,
  );

  /** Error from a failed export action; the read failures are merged in below. */
  private readonly actionError = signal('');
  readonly error = computed(
    () =>
      this.actionError() ||
      (this.exportersResource.error() ? 'Failed to load exporters' : '') ||
      (this.labelsResource.error() ? 'Failed to load labels' : ''),
  );
  // Written from the export-run subscribe (async); template-bound.
  readonly status = signal('');

  /** Column definitions with selection state, built dynamically from API response. */
  columns: ColumnDef[] = [];

  /** Client-side category filter over the fetched set (radio buttons). */
  labelFilter: 'good' | 'bad' | 'both' | 'corrections' = 'good';

  /**
   * Server-side partition the labels were fetched with. ``both`` fetches
   * everything (the radios then slice it); ``unverified`` / ``verified`` fetch
   * the Find work-queue / confirmed pile, which can't be sliced client-side.
   */
  serverFilter: 'both' | 'unverified' | 'verified' = 'both';

  /** Active export tab: 'clipboard' or an exporter name. */
  activeTab = 'clipboard';

  /** Exporter form state. */
  selectedExporter: ExporterEntry | null = null;
  formValues: Record<string, string> = {};
  readonly submitting = signal(false);

  /** Option lists for the active exporter's ``dynamic_options`` fields. An
   *  exporter whose destinations are only knowable at runtime (mailboxes,
   *  buckets, remote queues) fills its select from
   *  ``POST /api/exporters/field-options/<name>`` rather than from a list
   *  frozen at plugin-definition time (issue #3360). */
  readonly fieldOptions = new DynamicFieldOptions((key, values) =>
    this.exportersApi.getFieldOptions(this.activeExporterName, key, values),
  );

  /** Exporter the in-flight option fetches belong to. Read by the fetcher
   *  above at subscribe time, so it must be set before any refresh. */
  private activeExporterName = '';

  /** Dataset display name for default filenames. */
  private readonly datasetName = computed(
    () => this.statusResource.value()?.display_name || '',
  );

  /** Base columns that are always present. */
  private static readonly BASE_COLUMNS: { key: string; label: string }[] = [
    { key: 'label', label: 'Label' },
    { key: 'md5', label: 'MD5' },
    { key: 'filename', label: 'Filename' },
    { key: 'category', label: 'Category' },
  ];

  /**
   * Columns excluded from checkboxes/preview but always appended to *plugin*
   * exports (CSV/JSON files, webhooks), where they make the payload
   * re-importable: the exporter receives the whole `LabeledElement`s, so
   * `origin` reaches the file as a real dict and is JSON-serialized on the way
   * out (and JSON-parsed back on re-import).
   *
   * They are deliberately **not** appended to the clipboard, which flattens
   * rows to strings client-side: `origin` would stringify to `[object Object]`,
   * which no importer can parse back. Two unselectable junk columns that buy
   * zero matching power is a strictly worse paste, so the clipboard honours the
   * user's column selection exactly (issue #2770).
   */
  private static readonly ALWAYS_EXPORT_KEYS = ['origin', 'origin_name'];

  /** Detector/model name from any available source: the parent-supplied name
   *  (a Dashboard row action exports *that* row's detector, which need not be
   *  the active one), else the selected detector resolved through the registry
   *  (typical when this modal opens from the Find view), else the last name
   *  the label session carried. A signal, not a getter, so a name that lands
   *  after the modal opened still reaches the default filename below. */
  readonly effectiveDetectorName = computed(
    () =>
      this.detectorName() ||
      this.activeDetector.detectorName() ||
      this.labelSession.modelName,
  );

  constructor() {
    // Rebuild the column set when the labels read settles. The checkbox
    // `enabled` state lives on `columns`, so it stays a mutable field rather
    // than a pure computed; the effect mirrors the old subscribe's
    // `buildColumns(...)` call (with the no-arg fallback on error).
    effect(() => {
      if (this.labelsResource.hasValue()) {
        this.buildColumns(this.labelsResource.value()?.available_columns);
      } else if (this.labelsResource.error()) {
        this.buildColumns();
      }
    });

    // Re-apply the default filename when a name it is built from arrives late.
    // `applyDefaultFilename` runs once, at exporter-select time; the detector
    // registry and the dataset-status read can both settle after that, which
    // used to leave the filename permanently missing the piece that wasn't
    // loaded yet (issue #2819). Both sources are signals, so this effect
    // re-runs when either resolves — and only rewrites the field while it
    // still holds the value we generated, so a user edit is never clobbered.
    effect(() => {
      const detectorName = this.effectiveDetectorName();
      const datasetName = this.datasetName();
      const exporter = this.selectedExporter ?? this.activeTabExporter;
      if (!exporter) return;
      if (!detectorName && !datasetName) return;
      if (this.formValues['filepath'] === this.lastAutoFilename) {
        this.applyDefaultFilename(exporter);
      }
      // Same late-arrival story for any *other* field whose default is
      // templated on the detector name (issue #3199).
      this.reapplyTemplateDefaults(exporter);
    });
  }

  ngOnInit(): void {
    // Split the requested filter into a server-side partition (unverified /
    // verified are fetched with that label_filter) and a client-side category.
    const initialFilter = this.initialFilter();
    if (initialFilter === 'unverified' || initialFilter === 'verified') {
      this.serverFilter = initialFilter;
      this.labelFilter = 'both';
    } else if (initialFilter === 'unverified_good') {
      // The left work-queue export: the unverified partition (server-side),
      // sliced to the above-threshold good category (client-side).
      this.serverFilter = 'unverified';
      this.labelFilter = 'good';
    } else {
      this.serverFilter = 'both';
      this.labelFilter = initialFilter;
    }

    // Now that the input-derived `serverFilter` is set, release the labels read
    // (the dataset-status and exporter-list reads are eager and already in
    // flight). `buildColumns` rides the constructor effect on resolution.
    this.labelsReady.set(true);
  }

  /** Display labels for known metadata keys whose generic humanization would
   *  read poorly. ``name`` is a demo origin id (e.g. ``caltech101_m``) that
   *  reads confusingly as "Name" beside "Filename", so it surfaces as
   *  "Source"; ``origin_name`` (when present) as "Origin". */
  private static readonly KNOWN_COLUMN_LABELS: Record<string, string> = {
    name: 'Source',
    origin_name: 'Origin',
  };

  /** ``"media_type"`` → ``"Media type"``: humanize a raw metadata key for
   *  the column checkbox label. Known keys (see ``KNOWN_COLUMN_LABELS``) get a
   *  curated label; the rest are title-cased from the raw key. Keys that
   *  already read well ("Dimensions", "File Size") pass through unchanged; the
   *  export payload keeps the raw key either way. */
  private static humanizeColumnKey(key: string): string {
    const known = ExportModalComponent.KNOWN_COLUMN_LABELS[key];
    if (known) return known;
    const spaced = key.replace(/_/g, ' ').trim();
    return spaced ? spaced[0].toUpperCase() + spaced.slice(1) : key;
  }

  /** Build column definitions from available_columns or fall back to defaults. */
  private buildColumns(availableColumns?: string[]): void {
    const baseKeys = new Set(
      ExportModalComponent.BASE_COLUMNS.map((c) => c.key),
    );
    const alwaysKeys = new Set(ExportModalComponent.ALWAYS_EXPORT_KEYS);
    // Start with base columns
    this.columns = ExportModalComponent.BASE_COLUMNS.map((c) => ({
      key: c.key,
      label: c.label,
      enabled: true,
    }));
    // Add metadata columns discovered from the data (skip always-export columns)
    if (availableColumns) {
      for (const key of availableColumns) {
        if (!baseKeys.has(key) && !alwaysKeys.has(key)) {
          this.columns.push({
            key,
            label: ExportModalComponent.humanizeColumnKey(key),
            enabled: true,
            isMetadata: true,
          });
        }
      }
    }
  }

  get enabledColumns(): ColumnDef[] {
    return this.columns.filter((c) => c.enabled);
  }

  /** Tri-state of the column select-all control: how many columns are enabled. */
  get columnSelectionState(): 'none' | 'some' | 'all' {
    const enabled = this.enabledColumns.length;
    if (enabled === 0) return 'none';
    if (enabled >= this.columns.length) return 'all';
    return 'some';
  }

  /** From [x] (all enabled), clear every column; from [ ]/[-], enable them all. */
  toggleAllColumns(): void {
    const enable = this.columnSelectionState !== 'all';
    for (const col of this.columns) col.enabled = enable;
  }

  // --- Preview column resize ---
  //
  // The preview table's columns default to auto layout (`width: 100%`, each
  // column sized to its content up to a cap). Once the user grabs a divider we
  // switch to a pixel model: freeze every column's current width, flip the
  // table to `table-layout: fixed`, and let the grabbed column grow/shrink on
  // its own. The `.table-scroll` container already scrolls horizontally, so a
  // widened column reveals its full content instead of redistributing space
  // away from its neighbours. Widths are per-process view state, not persisted
  // (the preview is ephemeral).

  private static readonly MIN_COL_PX = 40;

  /** Pixel widths keyed by column key; populated on first resize. */
  colWidths: Record<string, number> = {};

  /** Once true, the table uses `table-layout: fixed` with explicit widths. */
  tableFixed = false;

  private colResize: {
    key: string;
    startX: number;
    startWidth: number;
  } | null = null;

  /** Begin dragging a column divider: freeze current widths, then track the
   *  grabbed column so `onColResizeMove` can size it. */
  startColResize(event: MouseEvent, key: string): void {
    event.preventDefault();
    event.stopPropagation();
    const th = (event.target as HTMLElement).closest('th');
    const table = th?.closest('table');
    if (!table) return;

    if (!this.tableFixed) {
      const ths = table.querySelectorAll('thead th') as NodeListOf<HTMLElement>;
      ths.forEach((cell) => {
        const cellKey = cell.getAttribute('data-col');
        if (cellKey) this.colWidths[cellKey] = cell.offsetWidth;
      });
      this.tableFixed = true;
    }

    this.colResize = {
      key,
      startX: event.clientX,
      startWidth: this.colWidths[key] ?? (th as HTMLElement).offsetWidth,
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  @HostListener('document:mousemove', ['$event'])
  onColResizeMove(event: MouseEvent): void {
    if (!this.colResize) return;
    const dx = event.clientX - this.colResize.startX;
    this.colWidths[this.colResize.key] = Math.max(
      ExportModalComponent.MIN_COL_PX,
      this.colResize.startWidth + dx,
    );
  }

  @HostListener('document:mouseup')
  onColResizeEnd(): void {
    if (!this.colResize) return;
    this.colResize = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }

  get filteredLabels(): LabeledElement[] {
    const labels = this.labelsList();
    if (this.labelFilter === 'good') {
      return labels.filter((e) => e.label === 'good');
    }
    if (this.labelFilter === 'bad') {
      return labels.filter((e) => e.label === 'bad');
    }
    if (this.labelFilter === 'corrections') {
      return labels.filter((e) => e.is_correction === true);
    }
    return labels;
  }

  /** Whether any labels are corrections (detector label was changed by user). */
  get hasCorrections(): boolean {
    return this.labelsList().some((e) => e.is_correction === true);
  }

  get previewLabels(): LabeledElement[] {
    return this.filteredLabels.slice(0, 50);
  }

  get hasLabels(): boolean {
    return this.filteredLabels.length > 0;
  }

  get hasExporterForm(): boolean {
    return this.selectedExporter !== null;
  }

  getCellValue(entry: LabeledElement, col: ColumnDef): string {
    if (col.isMetadata) {
      const meta = entry.custom_metadata;
      if (meta && col.key in meta) {
        return String(meta[col.key] ?? '');
      }
      return '';
    }
    return String((entry as unknown as Record<string, unknown>)[col.key] ?? '');
  }

  /** Columns sent to a plugin exporter: user-selected columns plus the
   *  always-export columns appended at the end. Clipboard copies use
   *  {@link enabledColumns} instead; see {@link ALWAYS_EXPORT_KEYS}. */
  private get exportColumns(): ColumnDef[] {
    const cols = [...this.enabledColumns];
    for (const key of ExportModalComponent.ALWAYS_EXPORT_KEYS) {
      cols.push({ key, label: key, enabled: true });
    }
    return cols;
  }

  /** Columns passed to the shared clipboard control (table mode): exactly the
   *  ones the user checked, matching what the preview table shows. */
  get clipboardColumns(): ClipboardColumn[] {
    return this.enabledColumns.map((c) => ({ key: c.key, label: c.label }));
  }

  /** Labels flattened to `{ columnKey: value }` rows for the clipboard control. */
  get clipboardRows(): Record<string, string>[] {
    const cols = this.enabledColumns;
    return this.filteredLabels.map((entry) => {
      const row: Record<string, string> = {};
      for (const c of cols) row[c.key] = this.getCellValue(entry, c);
      return row;
    });
  }

  /** Build a descriptive default filename for export.
   *  e.g. "Good-MyDetector-MyDataset.json" */
  private buildDefaultFilename(ext: string): string {
    const parts: string[] = [];
    if (this.serverFilter === 'unverified') parts.push('Unverified');
    else if (this.serverFilter === 'verified') parts.push('Verified');
    if (this.labelFilter === 'good') parts.push('Good');
    else if (this.labelFilter === 'bad') parts.push('Bad');
    else if (this.labelFilter === 'corrections') parts.push('Corrections');
    const detName = this.effectiveDetectorName();
    if (detName) parts.push(detName);
    const datasetName = this.datasetName();
    if (datasetName) parts.push(datasetName);
    if (parts.length === 0) parts.push('labels');
    // Sanitise: replace characters unsafe for filenames with hyphens
    const stem = parts.join('-').replace(/[\\/:*?"<>|]+/g, '-');
    return `${stem}.${ext}`;
  }

  /** The exporter's plugin-defined fields, narrowed to the legacy
   *  ImporterField shape (the OpenAPI spec types it as an open dict
   *  because plugin field schemas aren't part of the generated client). */
  private exporterFieldsOf(exporter: ExporterEntry): ImporterField[] {
    return (exporter.fields ?? []) as ImporterField[];
  }

  /** Initial form value for *field*: its declared default, or the first
   *  option when a select field has no default (so the form is never sitting
   *  on a blank pulldown that the user has to actively populate). */
  private defaultFor(field: ImporterField): string {
    if (field.default) {
      // A default that uses the field's declared `template_vars` is resolved
      // for display, so `"{detector_name}"` opens as the detector's actual
      // name rather than the raw placeholder (issue #3199). The server
      // substitutes again on submit - idempotently, since the placeholder is
      // already gone - and anything this client can't resolve (no detector
      // loaded yet) stays templated for the server to fill in, exactly as
      // before.
      return this.templateVars.resolveDefault(field, {
        detectorName: this.effectiveDetectorName(),
      });
    }
    if (
      field.field_type === 'select' &&
      !field.dynamic_options &&
      !field.allow_free_text &&
      (field.options?.length ?? 0) > 0
    ) {
      return field.options![0];
    }
    return '';
  }

  /** The filename this component last auto-filled into `filepath`. Compared
   *  against the live field value to tell "still ours" from "user-edited"
   *  before the constructor effect regenerates it. */
  private lastAutoFilename = '';

  /** Per-field values this component last auto-filled from a templated
   *  default, for the same "still ours" test as `lastAutoFilename` — the
   *  detector name can land after the form was built, and re-resolving must
   *  never overwrite something the user typed. */
  private lastAutoValues: Record<string, string> = {};

  /** Seed `formValues` from *exporter*'s field defaults, recording what was
   *  auto-filled so a late-resolving template var can refresh it. */
  private initFormValues(exporter: ExporterEntry): void {
    this.formValues = {};
    this.lastAutoValues = {};
    for (const f of this.exporterFieldsOf(exporter)) {
      const value = this.defaultFor(f);
      this.formValues[f.key] = value;
      this.lastAutoValues[f.key] = value;
    }
    // Drop the previous exporter's lists (and invalidate anything still in
    // flight for it) before fetching this one's.
    this.fieldOptions.reset();
    this.activeExporterName = exporter.name;
    this.fieldOptions.refreshAll(this.exporterFieldsOf(exporter), this.formValues);
  }

  /** Re-fetch the option lists of every field that depends on the one the
   *  user just edited. */
  onFieldChanged(changedKey: string): void {
    const exporter = this.activeTabExporter ?? this.selectedExporter;
    if (!exporter) return;
    this.fieldOptions.refreshDependentsOf(
      changedKey,
      this.exporterFieldsOf(exporter),
      this.formValues,
    );
  }

  /** Re-resolve templated defaults once a variable they need arrives.
   *
   *  `detector_name` is the case that bites: the detector registry can settle
   *  after the modal opened, which used to leave a `{detector_name}` default
   *  showing its placeholder forever. Only fields still holding exactly what
   *  we auto-filled are rewritten, so a user edit is never clobbered.
   *  `filepath` is skipped — `applyDefaultFilename` owns that one and builds a
   *  richer name (filter + detector + dataset) than the plugin's template. */
  private reapplyTemplateDefaults(exporter: ExporterEntry): void {
    for (const f of this.exporterFieldsOf(exporter)) {
      if (f.key === 'filepath') continue;
      if (!f.default || !f.template_vars?.length) continue;
      if (this.formValues[f.key] !== this.lastAutoValues[f.key]) continue;
      const value = this.defaultFor(f);
      this.formValues[f.key] = value;
      this.lastAutoValues[f.key] = value;
    }
  }

  /** Apply the dynamic default filename to the filepath form field if present. */
  private applyDefaultFilename(exporter: ExporterEntry): void {
    const filepathField = this.exporterFieldsOf(exporter).find(
      (f) => f.key === 'filepath',
    );
    if (filepathField) {
      const staticDefault = filepathField.default || '';
      // Derive extension from the static default (e.g. ".json", ".csv") or fall back
      const extMatch = staticDefault.match(/\.(\w+)$/);
      const ext = extMatch ? extMatch[1] : 'json';
      const filename = `data/${this.buildDefaultFilename(ext)}`;
      this.formValues['filepath'] = filename;
      this.lastAutoFilename = filename;
    }
  }

  /** Start exporter flow: if no fields, export immediately. */
  startExporter(exporter: ExporterEntry): void {
    const fields = this.exporterFieldsOf(exporter);
    if (fields.length === 0) {
      this.exportLabelsWith(exporter, {});
      return;
    }
    this.selectedExporter = exporter;
    this.initFormValues(exporter);
    this.applyDefaultFilename(exporter);
    this.actionError.set('');
    this.status.set('');
  }

  /** Select an exporter tab and initialise its form values. */
  selectExporterTab(exporter: ExporterEntry): void {
    this.activeTab = exporter.name;
    this.selectedExporter = exporter;
    this.initFormValues(exporter);
    this.applyDefaultFilename(exporter);
    this.actionError.set('');
    this.status.set('');
  }

  /** Re-generate the default filename when the label filter changes. */
  onLabelFilterChange(): void {
    const exp = this.activeTabExporter;
    if (exp) {
      this.applyDefaultFilename(exp);
    }
  }

  /** The exporter object for the currently active tab (null if clipboard). */
  get activeTabExporter(): ExporterEntry | null {
    if (this.activeTab === 'clipboard') return null;
    return this.exporters().find((e) => e.name === this.activeTab) || null;
  }

  /** Typed view of the active tab's plugin fields for the template (the
   *  generated ExporterEntry types `fields` as an open dict because plugin
   *  field schemas aren't part of the OpenAPI client). */
  get activeTabExporterFields(): ImporterField[] {
    const exp = this.activeTabExporter;
    return exp ? this.exporterFieldsOf(exp) : [];
  }

  /** Label for the action button on the active exporter tab. */
  get activeTabAction(): string {
    const exp = this.activeTabExporter;
    if (!exp) return 'Export';
    // A declared `opens_url` beats the name-sniffing below: the exporter has
    // told us the run ends in a new browser tab, so the button says so.
    if (exp.opens_url)
      return `Open Labelset in ${exp.display_name || exp.name}`;
    const name = (exp.display_name || exp.name).toLowerCase();
    if (name.includes('email') || name.includes('smtp')) return 'Send';
    if (name.includes('csv') || name.includes('file') || name.includes('json'))
      return 'Save';
    if (name.includes('webhook')) return 'Send';
    return 'Export';
  }

  /** Submit the currently active exporter tab. */
  submitExporterTab(): void {
    const exp = this.activeTabExporter;
    if (!exp) return;
    this.exportLabelsWith(exp, { ...this.formValues });
  }

  cancelExporterForm(): void {
    this.selectedExporter = null;
    this.actionError.set('');
    this.status.set('');
  }

  submitForm(): void {
    if (!this.selectedExporter) return;
    this.exportLabelsWith(this.selectedExporter, { ...this.formValues });
  }

  exportLabelsWith(
    exporter: ExporterEntry,
    fieldValues: Record<string, string>,
  ): void {
    const exporterLabel = exporter.display_name || exporter.name;
    const labelCount = this.filteredLabels.length;
    this.status.set(
      `Exporting ${labelCount.toLocaleString()} labels to ${exporterLabel}…`,
    );
    this.actionError.set('');
    this.submitting.set(true);

    const labelsData = {
      labels: this.filteredLabels,
      selected_columns: this.exportColumns.map((c) => c.key),
    };

    // Claim the tab *now*, while the click that got us here still counts as
    // user activation. Opening it from the response callback instead is what a
    // popup blocker exists to stop, and it gets swallowed silently (#2898).
    // Only exporters that declare `opens_url` are known to be heading for a new
    // tab; one that returns a URL without declaring it falls back to the
    // best-effort open in the callback, and to the toast's Open action.
    const pendingTab = exporter.opens_url ? openBlankTab() : null;

    this.exportersApi
      .runExport({
        exporter_name: exporter.name,
        field_values: fieldValues,
        results: labelsData,
        payload_kind: 'labelset',
      })
      .subscribe({
        next: (response) => {
          this.status.set('Labels exported.');
          this.submitting.set(false);
          this.selectedExporter = null;
          // An exporter can hand back a URL for the browser to open instead of
          // (or as well as) delivering the labelset somewhere — that's how a
          // third-party site with no ingest API gets the selection (#2855).
          const openUrl = safeExternalUrl(response?.open_url);
          let opened = false;
          if (openUrl) {
            opened = pendingTab
              ? pendingTab.navigate(openUrl)
              : openExternalUrl(openUrl);
          } else {
            // The exporter advertised a URL and didn't produce one; don't leave
            // a blank tab sitting there.
            pendingTab?.close();
          }
          const plural = labelCount === 1 ? '' : 's';

          // Three toast shapes: opened a tab, tried and got blocked, or a
          // plain delivery export. The blocked case survives the pre-opened
          // tab above (a blocker can refuse even a gesture-time popup), so say
          // why nothing happened rather than leaving the user staring at an
          // unchanged screen.
          let message = `Exported ${labelCount.toLocaleString()} label${plural} to ${exporterLabel}`;
          let detail = fieldValues['filepath']
            ? `Destination: ${fieldValues['filepath']}`
            : undefined;
          if (openUrl) {
            message = opened
              ? `Opened ${labelCount.toLocaleString()} label${plural} in ${exporterLabel}`
              : message;
            detail = opened ? undefined : 'Your browser blocked the new tab.';
          }

          // The parent closes the modal on `exported`, so the inline status
          // above is never seen. Fire a toast that outlives the modal so the
          // user gets confirmation the export actually succeeded (issue #2217).
          this.toast.success({
            message,
            detail,
            // The action button click *is* a user gesture, so this always gets
            // through — it doubles as the blocked-popup escape hatch and as a
            // way to reopen a tab the user closed by accident.
            action: openUrl
              ? {
                  label: 'Open',
                  title: openUrl,
                  onClick: () => openExternalUrl(openUrl),
                }
              : undefined,
            // A blocked tab makes that button the only way to reach the site,
            // so the toast has to outlive the 5s success default — otherwise
            // the escape hatch is gone before the user finishes reading why
            // nothing opened.
            autoDismissMs: openUrl && !opened ? 0 : undefined,
            dedupKey: 'export-labels-success',
          });
          this.exported.emit();
        },
        error: () => {
          pendingTab?.close();
          this.status.set('');
          this.actionError.set('Label export failed');
          this.submitting.set(false);
        },
      });
  }

  /** Map an exporter's emoji icon to an SVG icon type. */
  getExporterIconType(exp: ExporterEntry): string {
    const icon = exp.icon || '';
    if (icon === '📧' || icon.includes('📧')) return 'email';
    if (
      icon === '🖥️' ||
      icon === '\uD83D\uDDA5' ||
      icon === '\uD83D\uDDA5\uFE0F'
    )
      return 'server';
    if (icon === '🌐' || icon === '\uD83C\uDF10') return 'webhook';
    // An exporter that ends in a new browser tab reads as a link, whichever
    // emoji its author picked.
    if (exp.opens_url) return 'external-link';
    // Also match by name as fallback
    const name = (exp.name || '').toLowerCase();
    if (name.includes('email') || name.includes('smtp')) return 'email';
    if (name.includes('webhook')) return 'webhook';
    if (name.includes('server') || name.includes('file')) return 'server';
    return 'upload';
  }

  /** Modal heading, noting the server-side partition (and category slice) when present. */
  /**
   * Whether to warn that the current selection can't rebuild the detector:
   * a detector-scoped export narrowed to a single class. ``corrections`` is
   * a deliberate diff export rather than a labelset, and the Find view's
   * ``results`` scope is slicing hits, so neither warns.
   */
  get showPartialLabelsetNote(): boolean {
    return (
      this.scope() === 'detector' &&
      (this.labelFilter === 'good' || this.labelFilter === 'bad')
    );
  }

  get modalTitle(): string {
    if (this.serverFilter === 'unverified') {
      // The left work-queue export opens on the above-threshold good slice.
      return this.labelFilter === 'good'
        ? 'Export Unverified Good'
        : 'Export Unverified';
    }
    if (this.serverFilter === 'verified') return 'Export Verified';
    // Name the payload in the detector-scoped case: the user arrived here
    // from a *detector*, and what leaves is its labelset, not the detector
    // as an artifact.
    return this.scope() === 'detector' ? 'Export Labels' : 'Export';
  }

  close(): void {
    this.closed.emit();
  }
}

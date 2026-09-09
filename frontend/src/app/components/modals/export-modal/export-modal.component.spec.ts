import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { ExportModalComponent } from './export-modal.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ToastService } from '../../../services/toast.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleResource } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('ExportModalComponent', () => {
  let component: ExportModalComponent;
  let fixture: ComponentFixture<ExportModalComponent>;
  let httpMock: HttpTestingController;

  const mockExporters = [
    {
      name: 'server_json_file',
      display_name: 'Server JSON',
      fields: [],
      supported_payloads: ['find_results', 'labelset'],
    },
    {
      name: 'hidden',
      display_name: 'Hidden',
      hidden_from_picker: true,
      fields: [],
      supported_payloads: ['labelset'],
    },
    // Visible in the picker, but can only read a scored run — this modal
    // sends a labelset, so it must be filtered out (issue #3219).
    {
      name: 'find_only',
      display_name: 'Find Only',
      fields: [],
      supported_payloads: ['find_results'],
    },
  ];
  const mockLabels = {
    labels: [
      { md5: 'a', label: 'good', filename: 'a.wav' },
      { md5: 'b', label: 'bad', filename: 'b.wav' },
      { md5: 'c', label: 'good', filename: 'c.wav', is_correction: true },
    ],
    available_columns: [
      'label',
      'md5',
      'filename',
      'category',
      'extra',
      'name',
    ],
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExportModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ExportModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // The three init reads (dataset status, exporter list, labels) ride
  // `rxResource`, whose loaders run in a root effect rather than during
  // `detectChanges()`; tick to issue the GETs (the labels read also waits for
  // `ngOnInit` to set the input-derived filter), then settle before asserting.
  async function flushInit(
    exporters: unknown[] = mockExporters,
  ): Promise<void> {
    // Zoneless + rxResource: TestBed.tick() runs ngOnInit and the resource
    // loader effects to issue the GETs. whenStable() can't be used — a loading
    // rxResource holds the app unstable — so the rxResource specs drive CD with
    // tick()/settleResource() and never call detectChanges().
    TestBed.tick();
    // The eager status/exporter GETs fire on the tick; the labels GET is
    // released by `ngOnInit`'s signal flip a microtask later, so settle first
    // to let all three become pending before flushing.
    await settleResource();
    httpMock
      .expectOne('/api/dataset/status')
      .flush({ display_name: 'My Dataset' });
    httpMock.expectOne('/api/exporters').flush(exporters);
    httpMock.expectOne((r) => r.url === '/api/labels/export').flush(mockLabels);
    await settleResource();
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('loads the exporter list, filtering hidden entries', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(1);
    expect(component.exporters()[0].name).toBe('server_json_file');
  });

  // This modal sends a labelset. Offering an exporter that only reads a scored
  // run is how a labelset email used to go out empty and still report success
  // (issue #3219), so the picker filters on the capability, not just on
  // `hidden_from_picker`.
  it('drops exporters that cannot read a labelset', async () => {
    await flushInit();
    expect(component.exporters().map((e) => e.name)).not.toContain('find_only');
  });

  it('tells the API which payload kind it is sending', async () => {
    await flushInit();
    component.startExporter(mockExporters[0] as never);
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.body.payload_kind).toBe('labelset');
    req.flush({ success: true });
  });

  it('builds columns from available_columns once labels resolve', async () => {
    await flushInit();
    expect(component.labelsLoaded()).toBe(true);
    const keys = component.columns.map((c) => c.key);
    expect(keys).toContain('extra'); // discovered metadata column
    expect(keys).not.toContain('origin'); // always-export keys stay out of the checkboxes
  });

  it('gives known metadata keys a curated label instead of the raw key', async () => {
    await flushInit();
    const nameCol = component.columns.find((c) => c.key === 'name');
    // The raw key stays for the export payload; only the checkbox label is curated.
    expect(nameCol?.label).toBe('Source');
    const extraCol = component.columns.find((c) => c.key === 'extra');
    expect(extraCol?.label).toBe('Extra'); // unknown keys fall back to title-casing
  });

  it('reports the tri-state column selection and toggles all on/off', async () => {
    await flushInit();
    expect(component.columnSelectionState).toBe('all'); // every column starts enabled

    component.columns[0].enabled = false;
    expect(component.columnSelectionState).toBe('some');

    component.toggleAllColumns(); // 'some' -> select all
    expect(component.columnSelectionState).toBe('all');
    expect(component.columns.every((c) => c.enabled)).toBe(true);

    component.toggleAllColumns(); // 'all' -> deselect all
    expect(component.columnSelectionState).toBe('none');
    expect(component.columns.every((c) => !c.enabled)).toBe(true);

    component.toggleAllColumns(); // 'none' -> select all
    expect(component.columnSelectionState).toBe('all');
  });

  it('copies exactly the checked columns, without the always-export keys', async () => {
    await flushInit();
    for (const col of component.columns) col.enabled = col.key === 'md5';

    // The clipboard flattens rows client-side, so `origin` could only ever
    // reach the paste as `[object Object]`; it stays out entirely (issue #2770).
    expect(component.clipboardColumns.map((c) => c.key)).toEqual(['md5']);
    expect(component.clipboardRows[0]).toEqual({ md5: 'a' });

    // Plugin exports still get them appended - there the exporter receives the
    // real dict and serializes it as JSON.
    component.startExporter(mockExporters[0] as never);
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.body.results.selected_columns).toEqual([
      'md5',
      'origin',
      'origin_name',
    ]);
    req.flush({ message: 'ok' });
  });

  it('slices the fetched labels by the active category', async () => {
    await flushInit();
    component.labelFilter = 'good';
    expect(component.filteredLabels.length).toBe(2);
    component.labelFilter = 'bad';
    expect(component.filteredLabels.length).toBe(1);
    component.labelFilter = 'corrections';
    expect(component.filteredLabels.length).toBe(1);
  });

  // Issue #3263: the Dashboard row action and the train-mode right panel open
  // this modal with no filter at all, and what they are exporting is the
  // detector's *own* labelset — the thing that re-imports as the detector. A
  // good-only slice of that cannot rebuild it (training rejects a single-class
  // labelset), so the detector scope opens on All and says so when narrowed.
  describe('detector scope', () => {
    it('opens on All when the caller passes no filter', async () => {
      await flushInit();
      expect(component.labelFilter).toBe('both');
      expect(component.filteredLabels.length).toBe(3);
    });

    it('names the payload in the title', async () => {
      await flushInit();
      expect(component.modalTitle).toBe('Export Labels');
    });

    it('flags a one-sided selection, and only a one-sided one', async () => {
      await flushInit();
      expect(component.showPartialLabelsetNote).toBe(false); // opens on All
      component.labelFilter = 'good';
      expect(component.showPartialLabelsetNote).toBe(true);
      component.labelFilter = 'bad';
      expect(component.showPartialLabelsetNote).toBe(true);
      // A corrections export is a deliberate diff, not a would-be labelset.
      component.labelFilter = 'corrections';
      expect(component.showPartialLabelsetNote).toBe(false);
    });
  });

  // The Find view slices a scored run: "the good hits" is the normal ask
  // there, so it keeps the caller's filter and never warns.
  describe('results scope', () => {
    it('keeps the caller filter and stays quiet about it', async () => {
      fixture.componentRef.setInput('scope', 'results');
      fixture.componentRef.setInput('initialFilter', 'good');
      await flushInit();
      expect(component.labelFilter).toBe('good');
      expect(component.showPartialLabelsetNote).toBe(false);
      expect(component.modalTitle).toBe('Export');
    });
  });

  // The Dashboard row action names a detector in a list, so its read must be
  // that detector's persisted labelset rather than whatever pair the top-bar
  // pulldown is on — and must survive a live Find session, which fills that
  // pair's votes with the detector's call for every item (issue #3639).
  describe('labelset-scoped read', () => {
    it('asks for the named detector when one is given', async () => {
      fixture.componentRef.setInput('labelsetDetectorName', 'Sirens');
      TestBed.tick();
      await settleResource();
      httpMock
        .expectOne('/api/dataset/status')
        .flush({ display_name: 'My Dataset' });
      httpMock.expectOne('/api/exporters').flush(mockExporters);
      const req = httpMock.expectOne((r) => r.url === '/api/labels/export');
      expect(req.request.params.get('detector_name')).toBe('Sirens');
      req.flush(mockLabels);
      await settleResource();
    });

    it('reads the live session when no detector is named', async () => {
      TestBed.tick();
      await settleResource();
      httpMock
        .expectOne('/api/dataset/status')
        .flush({ display_name: 'My Dataset' });
      httpMock.expectOne('/api/exporters').flush(mockExporters);
      const req = httpMock.expectOne((r) => r.url === '/api/labels/export');
      expect(req.request.params.has('detector_name')).toBe(false);
      req.flush(mockLabels);
      await settleResource();
    });
  });

  it('reports correction availability', async () => {
    await flushInit();
    expect(component.hasCorrections).toBe(true);
  });

  it('emits exported after a successful export run', async () => {
    await flushInit();
    vi.spyOn(component.exported, 'emit');
    // A fieldless exporter exports immediately.
    component.startExporter(mockExporters[0] as never);
    httpMock.expectOne('/api/exporters/export').flush({ success: true });
    expect(component.exported.emit).toHaveBeenCalled();
  });

  it('fires a success toast that outlives the closing modal', async () => {
    await flushInit();
    const toast = TestBed.inject(ToastService);
    const successSpy = vi.spyOn(toast, 'success');
    component.labelFilter = 'good'; // two matching labels in the fixture
    component.startExporter(mockExporters[0] as never);
    httpMock.expectOne('/api/exporters/export').flush({ success: true });
    expect(successSpy).toHaveBeenCalledTimes(1);
    expect(successSpy.mock.calls[0][0].message).toBe(
      'Exported 2 labels to Server JSON',
    );
  });

  // An exporter can return an `open_url` for the browser to open in a new tab,
  // which is how a third-party site with no ingest API receives the labelset
  // (issue #2855).
  describe('open_url handling', () => {
    const openUrlExporter = {
      name: 'open_url',
      display_name: 'Open in Website',
      opens_url: true,
      fields: [],
      supported_payloads: ['find_results', 'labelset'],
    };

    /** A stand-in for the `Window` handle `window.open` hands back. */
    function fakeWindow() {
      return {
        closed: false,
        opener: {},
        location: { href: '' },
        close: vi.fn(),
      };
    }

    /** Stub `window.open`, returning *handle* as the opened window. */
    function stubWindowOpen(handle: unknown) {
      return vi
        .spyOn(window, 'open')
        .mockReturnValue(handle as unknown as Window);
    }

    // `window` outlives the TestBed, so a spy left installed on it carries its
    // call log into the next test and makes "was never called" assertions pass
    // or fail on the previous test's calls.
    afterEach(() => {
      vi.restoreAllMocks();
    });

    // The tab is claimed inside the click handler and navigated when the
    // response lands: a `window.open` deferred to the response callback is what
    // popup blockers eat, which is why nothing opened in issue #2898.
    it('claims the tab on click and navigates it when the URL arrives', async () => {
      await flushInit();
      const win = fakeWindow();
      const openSpy = stubWindowOpen(win);
      component.startExporter(openUrlExporter as never);

      // Opened before the response — with no URL yet to give it.
      expect(openSpy).toHaveBeenCalledWith('', '_blank');
      expect(win.location.href).toBe('');

      httpMock
        .expectOne('/api/exporters/export')
        .flush({ success: true, open_url: 'https://example.com/r?ids=a' });

      expect(win.location.href).toBe('https://example.com/r?ids=a');
      // `noopener` would make `window.open` return null even on success, so the
      // opener is severed by hand instead (see `utils/external-url.ts`).
      expect(openSpy).toHaveBeenCalledTimes(1);
      expect(win.opener).toBeNull();
    });

    it('offers an Open action on the toast so a blocked popup is recoverable', async () => {
      await flushInit();
      // `window.open` returning null is what a popup blocker looks like.
      const openSpy = stubWindowOpen(null);
      const successSpy = vi.spyOn(TestBed.inject(ToastService), 'success');
      component.startExporter(openUrlExporter as never);
      httpMock
        .expectOne('/api/exporters/export')
        .flush({ success: true, open_url: 'https://example.com/r' });

      const toast = successSpy.mock.calls[0][0];
      expect(toast.detail).toContain('blocked');
      expect(toast.action?.label).toBe('Open');
      // That button is the only way left to reach the site, so the toast must
      // not time out from under it.
      expect(toast.autoDismissMs).toBe(0);
      // The action's click is a real user gesture, so this one gets through.
      const win = fakeWindow();
      openSpy.mockReturnValue(win as unknown as Window);
      toast.action!.onClick();
      expect(openSpy).toHaveBeenLastCalledWith(
        'https://example.com/r',
        '_blank',
      );
    });

    it('reports an opened tab rather than an export in the toast message', async () => {
      await flushInit();
      stubWindowOpen(fakeWindow());
      const successSpy = vi.spyOn(TestBed.inject(ToastService), 'success');
      component.labelFilter = 'good'; // two matching labels in the fixture
      component.startExporter(openUrlExporter as never);
      httpMock
        .expectOne('/api/exporters/export')
        .flush({ success: true, open_url: 'https://example.com/r' });
      const toast = successSpy.mock.calls[0][0];
      expect(toast.message).toBe('Opened 2 labels in Open in Website');
      // Nothing to escape from: the default auto-dismiss applies.
      expect(toast.autoDismissMs).toBeUndefined();
    });

    // A user who closed the pre-opened tab while the export ran gets the same
    // recoverable toast as a blocked one.
    it('treats a closed pre-opened tab as a failure to open', async () => {
      await flushInit();
      const win = fakeWindow();
      stubWindowOpen(win);
      const successSpy = vi.spyOn(TestBed.inject(ToastService), 'success');
      component.startExporter(openUrlExporter as never);
      win.closed = true;
      httpMock
        .expectOne('/api/exporters/export')
        .flush({ success: true, open_url: 'https://example.com/r' });

      expect(win.location.href).toBe('');
      expect(successSpy.mock.calls[0][0].action?.label).toBe('Open');
    });

    it.each([
      'javascript:alert(1)',
      'data:text/html,x',
      'file:///etc/passwd',
      '/relative',
    ])('refuses to open a %s URL even if the server sent one', async (url) => {
      await flushInit();
      const win = fakeWindow();
      stubWindowOpen(win);
      component.startExporter(openUrlExporter as never);
      httpMock
        .expectOne('/api/exporters/export')
        .flush({ success: true, open_url: url });
      // The tab was claimed on click, but nothing unsafe is navigated to and
      // the blank tab doesn't outlive the export.
      expect(win.location.href).toBe('');
      expect(win.close).toHaveBeenCalled();
    });

    it('closes the claimed tab when the export fails outright', async () => {
      await flushInit();
      const win = fakeWindow();
      stubWindowOpen(win);
      component.startExporter(openUrlExporter as never);
      httpMock
        .expectOne('/api/exporters/export')
        .flush(
          { message: 'boom' },
          { status: 500, statusText: 'Server Error' },
        );
      expect(win.close).toHaveBeenCalled();
    });

    it('leaves a response without an open_url alone', async () => {
      await flushInit();
      const openSpy = stubWindowOpen(fakeWindow());
      // `mockExporters[0]` doesn't declare `opens_url`, so no tab is claimed.
      component.startExporter(mockExporters[0] as never);
      httpMock.expectOne('/api/exporters/export').flush({ success: true });
      expect(openSpy).not.toHaveBeenCalled();
    });

    it('labels the action button with the destination site', async () => {
      await flushInit([openUrlExporter]);
      component.selectExporterTab(openUrlExporter as never);
      expect(component.activeTabAction).toBe(
        'Open Labelset in Open in Website',
      );
    });
  });

  it('seeds formValues from field defaults and carries field_values on the POST', async () => {
    await flushInit();
    // An exporter with a required field opens its form (rather than exporting
    // immediately) with each field's default seeded into `formValues`.
    const exporter = {
      name: 'server_json_file',
      display_name: 'Server JSON',
      fields: [
        {
          key: 'format',
          field_type: 'select',
          options: ['json', 'csv'],
          default: 'csv',
          required: true,
        },
      ],
    };
    component.startExporter(exporter as never);
    expect(component.selectedExporter).toBe(exporter);
    expect(component.formValues['format']).toBe('csv');

    // Submitting the form carries the seeded field values on the run-export POST.
    component.submitForm();
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.body.field_values).toEqual({ format: 'csv' });
    req.flush({ success: true });
  });

  it('does not auto-select the first option for a free-text combobox field', async () => {
    await flushInit();
    const exporter = {
      name: 'free_text_exporter',
      display_name: 'Free Text Exporter',
      fields: [
        {
          key: 'q',
          field_type: 'select',
          options: ['a', 'b'],
          allow_free_text: true,
        },
      ],
    };
    component.startExporter(exporter as never);
    expect(component.formValues['q']).toBe('');
  });

  describe('default filename', () => {
    /** An exporter whose `filepath` field gets the generated default. */
    const fileExporter = {
      name: 'server_json_file',
      display_name: 'Server JSON',
      supported_payloads: ['labelset'],
      fields: [
        { key: 'filepath', field_type: 'text', default: 'data/labels.json' },
      ],
    };

    /** Land a detector registry naming `d1`, as the app-level refresh would. */
    function flushRegistry(): void {
      TestBed.inject(DatasetStateService).refresh();
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock
        .expectOne('/api/detectors/registry')
        .flush({ detectors: [{ id: 'd1', name: 'Birdsong' }] });
    }

    it('names the parent-supplied detector', async () => {
      fixture.componentRef.setInput('detectorName', 'Sirens');
      await flushInit();
      component.selectExporterTab(fileExporter as never);
      // No category prefix: a detector-scoped export opens on All, and the
      // whole labelset is what the plain name describes.
      expect(component.formValues['filepath']).toBe(
        'data/Sirens-My Dataset.json',
      );
    });

    it('prefixes the category once the export is narrowed to one', async () => {
      fixture.componentRef.setInput('detectorName', 'Sirens');
      // This one flips the radio *after* picking the tab, so the exporter has
      // to be resolvable by name off the loaded list, not just passed in.
      await flushInit([fileExporter]);
      component.selectExporterTab(fileExporter as never);
      component.labelFilter = 'good';
      component.onLabelFilterChange();
      expect(component.formValues['filepath']).toBe(
        'data/Good-Sirens-My Dataset.json',
      );
    });

    // The lifecycle gap behind issue #2819: the filename is built once, at
    // exporter-select time, so a detector registry that resolves afterwards
    // used to leave the detector out of it permanently.
    it('backfills the detector name when the registry resolves late', async () => {
      await flushInit();
      TestBed.inject(ActiveContextService).setActivePair('ds1', 'd1');
      component.selectExporterTab(fileExporter as never);
      expect(component.formValues['filepath']).toBe('data/My Dataset.json');

      flushRegistry();
      TestBed.tick();
      expect(component.effectiveDetectorName()).toBe('Birdsong');
      expect(component.formValues['filepath']).toBe(
        'data/Birdsong-My Dataset.json',
      );
    });

    it('leaves a user-edited filename alone when the name arrives late', async () => {
      await flushInit();
      TestBed.inject(ActiveContextService).setActivePair('ds1', 'd1');
      component.selectExporterTab(fileExporter as never);
      component.formValues['filepath'] = 'data/mine.json';

      flushRegistry();
      TestBed.tick();
      expect(component.formValues['filepath']).toBe('data/mine.json');
    });
  });

  // Issue #3199: a plugin field whose default is templated on the active
  // detector used to render its raw `{detector_name}` placeholder, because the
  // substitution only ever happened server-side, at run time.
  describe('templated field defaults', () => {
    /** An exporter with a non-`filepath` field templated on the detector. */
    const namedExporter = {
      name: 'labelset_api',
      display_name: 'Labelset API',
      fields: [
        {
          key: 'label_set_name',
          field_type: 'text',
          default: '{detector_name}',
          template_vars: ['detector_name'],
        },
        {
          key: 'undeclared',
          field_type: 'text',
          default: '{detector_name}',
        },
      ],
    };

    /** Land a detector registry naming `d1`, as the app-level refresh would. */
    function flushRegistry(): void {
      TestBed.inject(DatasetStateService).refresh();
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock
        .expectOne('/api/detectors/registry')
        .flush({ detectors: [{ id: 'd1', name: 'Birdsong' }] });
    }

    it('resolves a declared detector_name into the form value', async () => {
      fixture.componentRef.setInput('detectorName', 'Sirens');
      await flushInit();
      component.selectExporterTab(namedExporter as never);
      expect(component.formValues['label_set_name']).toBe('Sirens');
    });

    it('leaves a placeholder the field never declared', async () => {
      fixture.componentRef.setInput('detectorName', 'Sirens');
      await flushInit();
      component.selectExporterTab(namedExporter as never);
      // `portable_detector` withholds the declaration on purpose so it can
      // substitute per-detector itself; the preview must respect that.
      expect(component.formValues['undeclared']).toBe('{detector_name}');
    });

    it('leaves the placeholder for the server when no detector is known', async () => {
      await flushInit();
      component.selectExporterTab(namedExporter as never);
      expect(component.formValues['label_set_name']).toBe('{detector_name}');
    });

    it('backfills the name when the detector registry resolves late', async () => {
      await flushInit();
      TestBed.inject(ActiveContextService).setActivePair('ds1', 'd1');
      component.selectExporterTab(namedExporter as never);
      expect(component.formValues['label_set_name']).toBe('{detector_name}');

      flushRegistry();
      TestBed.tick();
      expect(component.formValues['label_set_name']).toBe('Birdsong');
    });

    it('leaves a user-edited value alone when the name arrives late', async () => {
      await flushInit();
      TestBed.inject(ActiveContextService).setActivePair('ds1', 'd1');
      component.selectExporterTab(namedExporter as never);
      component.formValues['label_set_name'] = 'mine';

      flushRegistry();
      TestBed.tick();
      expect(component.formValues['label_set_name']).toBe('mine');
    });
  });

  it('emits closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  describe('preview column resize', () => {
    /** Build a detached preview-style table and return its parts. */
    function makeTable(): {
      table: HTMLTableElement;
      th1: HTMLElement;
      handle: HTMLElement;
    } {
      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const tr = document.createElement('tr');
      const th1 = document.createElement('th');
      th1.setAttribute('data-col', 'label');
      const th2 = document.createElement('th');
      th2.setAttribute('data-col', 'md5');
      tr.append(th1, th2);
      thead.append(tr);
      table.append(thead);
      const handle = document.createElement('span');
      th1.append(handle);
      document.body.append(table);
      return { table, th1, handle };
    }

    function mousedownOn(handle: HTMLElement, clientX: number): MouseEvent {
      const ev = new MouseEvent('mousedown', { clientX });
      Object.defineProperty(ev, 'target', { value: handle });
      return ev;
    }

    it('freezes every column width and enters fixed layout on first grab', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      expect(component.tableFixed).toBe(true);
      expect(component.colWidths).toHaveProperty('label');
      expect(component.colWidths).toHaveProperty('md5');
      expect(document.body.style.cursor).toBe('col-resize');
      component.onColResizeEnd();
      table.remove();
    });

    it('resizes the grabbed column by the drag delta, clamped to a minimum', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      // offsetWidth is 0 under jsdom, so startWidth is 0; +80px drag → 80px.
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 180 }));
      expect(component.colWidths['label']).toBe(80);
      // A tiny drag can't shrink the column below the 40px floor.
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 110 }));
      expect(component.colWidths['label']).toBe(40);
      component.onColResizeEnd();
      table.remove();
    });

    it('ignores pointer motion once the drag ends and clears the cursor', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 180 }));
      component.onColResizeEnd();
      expect(document.body.style.cursor).toBe('');
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 400 }));
      expect(component.colWidths['label']).toBe(80); // unchanged after end
      table.remove();
    });
  });

  // An exporter whose destination list is only knowable at runtime. Before
  // issue #3360 the select rendered `field.options` verbatim, so a
  // `dynamic_options` field was permanently empty here.
  describe('dynamic_options exporter fields', () => {
    const dynamicExporter = {
      name: 'remote_queue',
      display_name: 'Remote Queue',
      supported_payloads: ['find_results', 'labelset'],
      fields: [
        {
          key: 'account',
          field_type: 'select',
          label: 'Account',
          options: ['personal', 'team'],
          default: 'personal',
        },
        {
          key: 'queue',
          field_type: 'select',
          label: 'Queue',
          required: true,
          dynamic_options: true,
          depends_on: ['account'],
        },
      ],
    };

    /** Flush one options POST, asserting the exporter and body it carried. */
    function flushOptions(
      values: Record<string, string>,
      options: { value: string; label: string }[],
    ): void {
      const req = httpMock.expectOne(
        '/api/exporters/field-options/remote_queue',
      );
      expect(req.request.body).toEqual({ field_key: 'queue', values });
      req.flush({ options });
    }

    it('fetches the option list when the exporter tab is selected', async () => {
      await flushInit([...mockExporters, dynamicExporter]);
      component.selectExporterTab(dynamicExporter as never);

      flushOptions({ account: 'personal', queue: '' }, [
        { value: 'p-1', label: 'Personal 1' },
        { value: 'p-2', label: 'Personal 2' },
      ]);
      await settleResource();

      const field = component.activeTabExporterFields[1];
      expect(component.fieldOptions.optionsFor(field)).toEqual([
        { value: 'p-1', label: 'Personal 1' },
        { value: 'p-2', label: 'Personal 2' },
      ]);
      // Required with no default: the first fetched option is selected, so the
      // export body carries a real value rather than a blank.
      expect(component.formValues['queue']).toBe('p-1');
    });

    it('renders the fetched options in the select', async () => {
      await flushInit([...mockExporters, dynamicExporter]);
      component.selectExporterTab(dynamicExporter as never);
      flushOptions({ account: 'personal', queue: '' }, [
        { value: 'p-1', label: 'Personal 1' },
      ]);
      await settleResource();

      const labels = Array.from(
        (fixture.nativeElement as HTMLElement).querySelectorAll(
          '.tab-field select option',
        ),
      ).map((o) => o.textContent?.trim());
      expect(labels).toContain('Personal 1');
    });

    it('re-fetches a dependent field when its dependency changes', async () => {
      await flushInit([...mockExporters, dynamicExporter]);
      component.selectExporterTab(dynamicExporter as never);
      flushOptions({ account: 'personal', queue: '' }, [
        { value: 'p-1', label: 'Personal 1' },
      ]);
      await settleResource();

      component.formValues['account'] = 'team';
      component.onFieldChanged('account');
      flushOptions({ account: 'team', queue: '' }, [
        { value: 't-1', label: 'Team 1' },
      ]);
      await settleResource();

      // The stale personal-account selection is gone, replaced by one the new
      // list offers.
      expect(component.formValues['queue']).toBe('t-1');
    });

    it('surfaces a fetch failure inline instead of silently emptying the dropdown', async () => {
      await flushInit([...mockExporters, dynamicExporter]);
      component.selectExporterTab(dynamicExporter as never);
      httpMock
        .expectOne('/api/exporters/field-options/remote_queue')
        .flush(
          { message: 'queue service down' },
          { status: 502, statusText: 'Bad Gateway' },
        );
      await settleResource();

      expect(component.fieldOptions.error()['queue']).toBe(
        'queue service down',
      );
      expect((fixture.nativeElement as HTMLElement).textContent).toContain(
        'queue service down',
      );
    });

    it('does not fetch anything for an exporter with only static fields', async () => {
      await flushInit([...mockExporters, dynamicExporter]);
      component.selectExporterTab(mockExporters[0] as never);
      // `httpMock.verify()` in `afterEach` fails the test if a request was made.
      expect(component.fieldOptions.options()).toEqual({});
    });
  });

});

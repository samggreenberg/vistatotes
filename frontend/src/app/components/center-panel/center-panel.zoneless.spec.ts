
import { HttpTestingController } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CenterPanelComponent } from './center-panel.component';
import { EmbedderInfo, Media } from '../../models/api.models';
import { RegionBox } from './image-viewer/image-viewer.component';
import { ANIMATIONS_OFF_CLASS, ANIMATIONS_ON_CLASS } from '../../utils/reduced-motion';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';
import { voteBodyWithoutProvenance } from '../../testing/mocks';

/**
 * Zoneless staleness canary for the center panel.
 * Phase 1.2 dropped the `zone.run(...)` re-entries from
 * `KeyboardService`, moving the change-detection trigger to this consumer, whose
 * shortcut-driven state (`isVoting`/`spinningVote`/`swipeClass`/`undoToastText`/
 * `volume`/`pendingBadConfirm` + the settings mirror) is now signalized.
 *
 * This spec runs under a zoneless `TestBed` and drives the component through the
 * *production channel* — a real `keydown` dispatched on `document` (handled by the
 * live `KeyboardService` listener, NOT a bound template event) — then asserts on
 * the rendered DOM after `settleZoneless()` with NO manual `detectChanges()`. The
 * keyboard callback and the vote/undo HTTP `.subscribe()` callbacks are all
 * un-bound, so if any of those state writes were a plain field instead of a signal
 * the scheduler would never be notified and the DOM would stay stale — failing
 * these assertions.
 */
describe('CenterPanelComponent (zoneless keyboard canary)', () => {
  let fixture: ComponentFixture<CenterPanelComponent>;
  let component: CenterPanelComponent;
  let httpMock: HttpTestingController;

  // Document media keeps the viewer fully inert in jsdom: it only sets a
  // sanitized iframe URL — no HTTP, no Loading placeholder (text viewer's
  // NG0100), no native `fetch()` of a relative URL (audio/video) and no
  // ViewChild settling quirk (image). The viewer is incidental here — the
  // assertions are on the always-rendered voting overlay / undo toast.
  const mockMedia: Media = {
    id: 1,
    media_type: 'document',
    filename: 'test.pdf',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    configureZoneless({
      imports: [CenterPanelComponent],
      providers: [...provideHttpTesting()],
    });
    fixture = TestBed.createComponent(CenterPanelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);

    // `media` is a decorator @Input; set it through `setInput` (the same channel
    // the parent's binding uses) so the write schedules CD.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);

    // init() wires the keyboard subscription + starts the document listener and
    // kicks the settings/embedders loads. The settings GET is driven by an
    // `rxResource`; while it is loading the zoneless app stays unstable, so we
    // must NOT `whenStable()` until it is flushed. Drain a macrotask + tick to
    // let the GETs be issued, flush them, THEN settle. show_animations:'hide'
    // takes the non-animated vote branch (no dangling timers) and is the
    // production path for "reduce motion".
    component.init();
    await new Promise<void>((resolve) => setTimeout(resolve));
    TestBed.tick();
    // label_hint_dismissed:true so the first vote's hint-dismissal does not fire
    // a settings PUT we'd have to flush.
    for (const req of httpMock.match((r) => r.url.includes('settings'))) {
      req.flush({ show_animations: 'hide', label_hint_dismissed: true });
    }
    for (const req of httpMock.match((r) => r.url.includes('embedders'))) {
      req.flush({ embedders: [] });
    }
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
    // The settings effect mirrors show_animations:'hide' onto <html>; undo it so
    // the class does not leak into other specs in the same jsdom document.
    document.documentElement.classList.remove(ANIMATIONS_OFF_CLASS);
  });

  function votingOverlay(): HTMLElement {
    return fixture.nativeElement.querySelector('vt-voting-overlay');
  }

  it('renders the vote as cast after a keyboard ArrowRight, with no manual detectChanges', async () => {
    expect(votingOverlay().querySelector('.btn-good.voted')).toBeNull();

    // Production channel: a real → keypress, handled by the live KeyboardService
    // listener (un-bound document callback), dispatches a 'vote' action that the
    // component's subscription turns into castVote('good').
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    await settleZoneless(fixture);

    // The optimistic vote POST; its response callback flips `isVoting` (a signal)
    // back, which is the only thing that can schedule CD for the un-bound chain.
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'good', click_time: 1 });
    await settleZoneless(fixture);

    expect(component.voteState.goodVotes.has(1)).toBe(true);
    // If `isVoting`/the vote state weren't signalized, this would still be null.
    expect(votingOverlay().querySelector('.btn-good.voted')).not.toBeNull();
  });

  it('shows the undo toast after a keyboard Cmd/Ctrl-Z, with no manual detectChanges', async () => {
    // Land a vote first so the undo stack has an entry to pop.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    await settleZoneless(fixture);
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'good', click_time: 1 });
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.undo-toast')).toBeNull();

    // Ctrl-Z → KeyboardService emits {type:'undo'} → voteState.undo() emits on
    // toast$, whose un-bound subscription writes the `undoToastText` signal.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true }));
    await settleZoneless(fixture);
    // undo() re-POSTs the reverted target ('none'); flush it.
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'none', click_time: 2 });
    await settleZoneless(fixture);

    const toast = fixture.nativeElement.querySelector('.undo-toast');
    expect(toast).not.toBeNull();
    expect(toast!.textContent).toContain('Undid vote on test.pdf');
  });
});

describe('CenterPanelComponent', () => {
  let component: CenterPanelComponent;
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;

  const mockMedia: Media = {
    id: 1,
    media_type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [CenterPanelComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(CenterPanelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should show placeholder when no media selected', () => {
    TestBed.tick();
    expect(fixture.nativeElement.textContent).toContain('Select a media item to view');
  });

  it('should show audio player for audio media', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-audio-player')).toBeTruthy();
  });

  it('should show image viewer for image media', () => {
    fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'image' });
    // The image-view-controls block gates on the `imageViewer` view query, now a
    // signal: it resolves after the first pass and schedules a clean follow-up
    // render, so no NG0100 workaround is needed.
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-image-viewer')).toBeTruthy();
  });

  it('should show video player for video media', () => {
    fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'video' });
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-video-player')).toBeTruthy();
  });

  it('should show text viewer for text media', () => {
    fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'text' });
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-text-viewer')).toBeTruthy();
    // The rendered text-viewer fetches its paragraph on init; flush it so the
    // afterEach httpMock.verify() sees no dangling request.
    httpMock.expectOne('/api/medias/1/text').flush({ text: '', paragraphs: [] });
  });

  it('should show document viewer for document media', () => {
    fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'document' });
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-document-viewer')).toBeTruthy();
  });

  it('should show voting overlay when media is selected', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('vt-voting-overlay')).toBeTruthy();
  });

  it('should display metadata', () => {
    fixture.componentRef.setInput('media', mockMedia);
    component.showMetadata.set(true);
    TestBed.tick();
    const text = fixture.nativeElement.textContent;
    expect(text).toContain('test.wav');
    expect(text).toContain('MD5');
    expect(text).toContain('abc123');
  });

  it('should send vote request on castVote', () => {
    fixture.componentRef.setInput('media', mockMedia);
    component.showAnimations.set(false);
    TestBed.tick();

    let emitted: { id: number; vote: string } | undefined;
    component.mediaVoted.subscribe((e: { id: number; vote: string }) => (emitted = e));

    component.castVote('good');

    // The vote POST sends the absolute target state, not a "vote" click; the
    // server's response reconciles the optimistic local view directly (no
    // follow-up GET /api/votes).
    const voteReq = httpMock.expectOne('/api/medias/1/vote');
    expect(voteBodyWithoutProvenance(voteReq)).toEqual({ target: 'good' });
    voteReq.flush({ state: 'good', click_time: 1 });

    expect(emitted).toEqual({ id: 1, vote: 'good' });
    expect(component.voteState.goodVotes.has(1)).toBe(true);
  });

  /**
   * The vote belongs to the item that was selected when the key was pressed.
   * Selection can move while the POST is in flight or during the 180ms swipe
   * animation (a click in the left list, hover-focus, a concurrent auto-advance),
   * so `mediaVoted` must carry the id captured at castVote() time. Emitting the
   * *current* selection's id pairs a new item with the old vote direction, which
   * makes find-view mark the wrong row optimistically verified and makes
   * label-view's auto-advance exclude the wrong id.
   */
  describe('mediaVoted id is pinned to the voted item', () => {
    afterEach(() => {
      vi.useRealTimers();
      document.documentElement.classList.remove(ANIMATIONS_ON_CLASS);
    });

    it('emits the voted id when selection moves while the vote is in flight', () => {
      fixture.componentRef.setInput('media', mockMedia);
      component.showAnimations.set(false);
      TestBed.tick();

      let emitted: { id: number; vote: string } | undefined;
      component.mediaVoted.subscribe((e: { id: number; vote: string }) => (emitted = e));

      component.castVote('good');
      const voteReq = httpMock.expectOne('/api/medias/1/vote');

      // The user clicks a different card before the server answers.
      fixture.componentRef.setInput('media', { ...mockMedia, id: 2, filename: 'other.pdf' });
      TestBed.tick();

      voteReq.flush({ state: 'good', click_time: 1 });

      expect(emitted).toEqual({ id: 1, vote: 'good' });
    });

    it('emits the voted id when selection moves during the swipe animation', async () => {
      vi.useFakeTimers();
      // The settings mirror put `animations-off` on <html> in beforeEach; swap it
      // for `animations-on` so prefersReducedMotion() reports false and castVote()
      // takes the animated (deferred-emit) branch.
      document.documentElement.classList.remove(ANIMATIONS_OFF_CLASS);
      document.documentElement.classList.add(ANIMATIONS_ON_CLASS);

      fixture.componentRef.setInput('media', mockMedia);
      component.showAnimations.set(true);
      TestBed.tick();

      let emitted: { id: number; vote: string } | undefined;
      component.mediaVoted.subscribe((e: { id: number; vote: string }) => (emitted = e));

      component.castVote('bad');
      httpMock.expectOne('/api/medias/1/vote').flush({ state: 'bad', click_time: 1 });
      // The deferred emit has not fired yet; selection moves inside the window.
      expect(emitted).toBeUndefined();

      fixture.componentRef.setInput('media', { ...mockMedia, id: 2, filename: 'other.pdf' });
      TestBed.tick();

      // Drain both the 180ms emit timer and the 300ms spin timer.
      await vi.advanceTimersByTimeAsync(300);
      TestBed.tick();

      expect(emitted).toEqual({ id: 1, vote: 'bad' });
    });
  });

  it('should prevent double voting', () => {
    fixture.componentRef.setInput('media', mockMedia);
    component.showAnimations.set(false);
    TestBed.tick();
    component.isVoting.set(true);
    component.castVote('good');
    httpMock.expectNone('/api/medias/1/vote');
  });

  it('should load votes via voteState', () => {
    component.voteState.loadVotes();
    const req = httpMock.expectOne('/api/votes');
    req.flush({ good: [1, 2], bad: [3], click_times: {}, learned_scores: {} });
    expect(component.voteState.goodVotes.has(1)).toBe(true);
    expect(component.voteState.goodVotes.has(2)).toBe(true);
    expect(component.voteState.badVotes.has(3)).toBe(true);
  });

  it('should clear swipe class when media changes', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    // Simulate swipe ending
    component.swipeClass.set('swipe-right');

    // Change to new media (triggers the media-change effect)
    fixture.componentRef.setInput('media', { ...mockMedia, id: 2, filename: 'next.wav' });
    TestBed.tick();

    expect(component.swipeClass()).toBe('');
  });

  it('should format metadata values', () => {
    expect(component.formatMetadataValue('File Size', 2048)).toBe('2.0 KB');
    expect(component.formatMetadataValue('Duration', 3.5)).toBe('3.5s');
    expect(component.formatMetadataValue('Frequency', 44100)).toBe('44100 Hz');
    expect(component.formatMetadataValue('Other', 'hello')).toBe('hello');
  });

  /**
   * v2 patch-embedder plan, item 15: vote-API contract for region annotations.
   * `region_box` must be present on a yes-vote when a box is drawn, absent on
   * a yes-vote without a box, and never present on any no-vote (no-votes are
   * region-agnostic (see "Vote attribution → v2" in docs/plans/patch-embedder.md).
   */
  describe('vote-API contract for region_box', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png' };
    const box: RegionBox = [0.1, 0.2, 0.5, 0.6];

    function setup(): void {
      fixture.componentRef.setInput('media', imageMedia);
      component.showAnimations.set(false);
      TestBed.tick();
    }

    it('attaches region_box to a yes-vote when a box is drawn', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.5, 0.6] });
      req.flush({ state: 'good', click_time: 1 });
    });

    it('omits region_box from a yes-vote when no box is drawn', () => {
      setup();
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'good' });
      req.flush({ state: 'good', click_time: 1 });
    });

    it('omits region_box from a no-vote even when a box is drawn (after confirm)', () => {
      setup();
      component.onRegionBoxChange(box);
      // First ← arms the discard-confirm; no request yet.
      component.castVote('bad');
      httpMock.expectNone('/api/medias/1/vote');
      expect(component.pendingBadConfirm()).toBe(true);
      // Second ← throws the box away and votes no.
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'bad' });
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('omits region_box from a no-vote when no box is drawn (no confirm armed)', () => {
      setup();
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm()).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });
  });

  /**
   * v2 patch-embedder plan, item 12: bad-vote-with-box requires two consecutive
   * ← presses (no timer). Esc, mouse-on-box, or item navigation while armed
   * clears the armed state and keeps the box.
   */
  describe('sticky bad-vote-confirm armed state', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png' };
    const box: RegionBox = [0.1, 0.2, 0.5, 0.6];

    function setup(): void {
      fixture.componentRef.setInput('media', imageMedia);
      component.showAnimations.set(false);
      TestBed.tick();
    }

    it('arms on first ← without firing a request, fires on second ←', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      httpMock.expectNone('/api/medias/1/vote');
      expect(component.pendingBadConfirm()).toBe(true);
      expect(component.currentRegionBox).toEqual(box);

      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm()).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('cancels armed state on onArmedConfirmCanceled (Esc/mouse-on-box) and keeps the box', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm()).toBe(true);

      // Esc-while-armed (or mousedown-on-box) routes through this handler from
      // the image viewer.
      component.onArmedConfirmCanceled();
      expect(component.pendingBadConfirm()).toBe(false);
      expect(component.currentRegionBox).toEqual(box);
      httpMock.expectNone('/api/medias/1/vote');
    });

    it('cancels armed state when the box is cleared (Esc-while-not-armed routes via regionBoxChange(null))', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm()).toBe(true);

      component.onRegionBoxChange(null);
      expect(component.pendingBadConfirm()).toBe(false);
      expect(component.currentRegionBox).toBeNull();
    });

    it('cancels armed state when the user navigates to another item', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm()).toBe(true);

      const next: Media = { ...imageMedia, id: 2, filename: 'next.png' };
      fixture.componentRef.setInput('media', next);
      TestBed.tick();
      expect(component.pendingBadConfirm()).toBe(false);
      expect(component.currentRegionBox).toBeNull();
    });

    it('does not arm when no box is drawn (single ← votes no immediately)', () => {
      setup();
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm()).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('uses the box on a yes-vote even after a first ← would have armed (yes wins over armed-only)', () => {
      setup();
      component.onRegionBoxChange(box);
      // Without arming first: yes-vote attaches the box immediately.
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(voteBodyWithoutProvenance(req)).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.5, 0.6] });
      req.flush({ state: 'good', click_time: 1 });
    });
  });

  /**
   * Structural-embedder plan: the matched-region overlay rides patch's
   * `best_region` machinery, so the Highlight toggle must show for *both*
   * patch-region and structural (geometric-verification) embedders — structural
   * embedders emit the RANSAC inlier box as `best_region` but leave
   * `supports_patch_regions` false. The marquee copy also nudges toward boxing
   * the pattern on structural datasets (the box defines the template).
   */
  describe('region-overlay capability + structural marquee copy', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png', embedder: 'enc' };

    function withEmbedders(infos: EmbedderInfo[]): void {
      (component as unknown as { embedderInfos: EmbedderInfo[] }).embedderInfos = infos;
    }

    function info(over: Partial<EmbedderInfo>): EmbedderInfo {
      return { name: 'enc', media_type_id: 'image', ...over };
    }

    it('hides the overlay toggle when the embedder is single-vector (neither capability)', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([info({})]);
      expect(component.regionOverlayCapable).toBe(false);
    });

    it('shows the overlay toggle for patch-region embedders', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([info({ supports_patch_regions: true })]);
      expect(component.regionOverlayCapable).toBe(true);
    });

    it('shows the overlay toggle for structural embedders (geometric verification)', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([info({ supports_geometric_verification: true })]);
      expect(component.regionOverlayCapable).toBe(true);
    });

    it('defaults to no overlay when the embedder is unknown / not yet loaded', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([]);
      expect(component.regionOverlayCapable).toBe(false);
    });

    it('nudges marquee copy toward boxing the pattern on structural datasets', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([info({ supports_geometric_verification: true })]);
      expect(component.structuralDataset).toBe(true);
      expect(component.marqueeTitle).toContain('box the pattern you want to match');
      expect(component.marqueeAriaLabel).toBe('Marquee: box the pattern to match');
    });

    it('keeps generic marquee copy on non-structural datasets', () => {
      fixture.componentRef.setInput('media', imageMedia);
      withEmbedders([info({ supports_patch_regions: true })]);
      expect(component.structuralDataset).toBe(false);
      expect(component.marqueeTitle).toContain('draw a region');
      expect(component.marqueeAriaLabel).toBe('Marquee: draw region');
    });
  });

  // The panel owns the vertical band below the image (toolbar, vote row,
  // metadata tray), so it is the only place a mousedown there can be caught and
  // handed to the viewer. It stays a pure delegation: the viewer decides whether
  // to claim the event.
  describe('off-canvas region-draw delegation', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png' };

    function stubViewer(): { calls: MouseEvent[] } {
      const calls: MouseEvent[] = [];
      (component as unknown as { imageViewer: () => unknown }).imageViewer = () => ({
        tryStartOffCanvasDraw: (e: MouseEvent) => calls.push(e),
        regionDrawActive: true,
      });
      return { calls };
    }

    function mousedown(): MouseEvent {
      return { button: 0, clientX: 10, clientY: 10, target: document.createElement('div') } as unknown as MouseEvent;
    }

    it('forwards a panel mousedown to the image viewer', () => {
      fixture.componentRef.setInput('media', imageMedia);
      const { calls } = stubViewer();
      component.onPanelMouseDown(mousedown());
      expect(calls.length).toBe(1);
    });

    it('does not forward for non-image media', () => {
      fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'audio' });
      const { calls } = stubViewer();
      component.onPanelMouseDown(mousedown());
      expect(calls.length).toBe(0);
    });

    it('reports regionDrawActive only for images, for the panel-wide crosshair', () => {
      fixture.componentRef.setInput('media', imageMedia);
      stubViewer();
      expect(component.regionDrawActive).toBe(true);
      fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'audio' });
      expect(component.regionDrawActive).toBe(false);
    });

    it('is inert when there is no image viewer mounted', () => {
      fixture.componentRef.setInput('media', imageMedia);
      (component as unknown as { imageViewer: () => unknown }).imageViewer = () => undefined;
      expect(component.regionDrawActive).toBe(false);
      expect(() => component.onPanelMouseDown(mousedown())).not.toThrow();
    });
  });
});

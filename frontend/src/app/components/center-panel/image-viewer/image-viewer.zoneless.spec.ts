import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ElementRef } from '@angular/core';

import { ImageViewerComponent, RegionBox } from './image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/**
 * Zoneless staleness canary for the image viewer.
 * Phase 2.3 signalized the viewer state that is written from
 * *un-patched* callbacks — the window-level `keydown`/`keyup`/`blur` (Shift) and
 * `mousemove`/`mouseup` drag handlers, the `ResizeObserver` rendered-size writes,
 * and the shake `setTimeout`. None of those callbacks are bound template/host
 * listeners, so under zoneless they schedule no change detection on their own; only
 * because the destinations are signals read in the template does a write notify the
 * scheduler.
 *
 * This spec runs under a zoneless `TestBed` and drives the component through the
 * *production channel* — a real `keydown`/`keyup` dispatched on `window` (handled by
 * the live constructor-registered listener, NOT a bound template event) — then
 * asserts on the rendered DOM after `settleZoneless()` with NO manual
 * `detectChanges()`. If `shiftHeld` were still a plain field, the `Shift` press
 * would never repaint the `.region-mode` crosshair affordance and these assertions
 * would fail.
 */
describe('ImageViewerComponent (zoneless Shift-drag canary)', () => {
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: Media = {
    id: 2,
    media_type: 'image',
    filename: 'test.png',
    md5: 'def456',
    custom_metadata: {},
  };

  beforeEach(async () => {
    configureZoneless({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    });
    fixture = TestBed.createComponent(ImageViewerComponent);
    // `media` is a decorator @Input; set it through `setInput` (the same channel
    // the parent's binding uses) so the first render has a media to read.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
  });

  function wrap(): HTMLElement {
    return fixture.nativeElement.querySelector('.image-wrap');
  }

  it('toggles the region-mode crosshair when Shift is pressed/released via the window listener, with no manual detectChanges', async () => {
    expect(wrap().classList.contains('region-mode')).toBe(false);

    // Production channel: a real Shift keydown, handled by the live constructor
    // listener (an un-bound window callback). It writes the `shiftHeld` signal,
    // which `regionDrawActive` reads in the template — the only thing that can
    // schedule CD for this un-bound chain under zoneless.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(true);

    // Releasing Shift (also an un-bound window callback) must repaint it away.
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(false);
  });

  it('clears region-mode on window blur via the un-bound blur listener', async () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(true);

    // alt-tab / focus loss fires a window 'blur' that drops the Shift state.
    window.dispatchEvent(new Event('blur'));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(false);
  });
});

describe('ImageViewerComponent', () => {
  let component: ImageViewerComponent;
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: Media = {
    id: 2,
    media_type: 'image',
    filename: 'test.png',
    md5: 'def456',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(ImageViewerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set imageSrc when media changes', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(component.imageSrc()).toBe('/api/medias/2/image');
  });

  it('should hide image until loaded to prevent flash of old image', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(component.imageReady()).toBe(false);

    component.onImageLoad();
    expect(component.imageReady()).toBe(true);
  });

  it('should reset imageReady when media changes', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    component.onImageLoad();
    expect(component.imageReady()).toBe(true);

    const nextMedia = { ...mockMedia, id: 3, filename: 'next.png' };
    fixture.componentRef.setInput('media', nextMedia);
    TestBed.tick();
    expect(component.imageReady()).toBe(false);
  });

  // Regression: when MediaMetadataCacheService hydrates the stub into a
  // fully-typed Media for the same id, MediaStateService.selectedMedia returns
  // a *new reference*, which used to retrigger the media-change reset. That
  // flipped `imageReady` back to false while leaving `imageSrc` unchanged, so
  // Angular's property binding skipped the DOM write and no fresh (load) event
  // fired, leaving the canvas hidden behind `visibility: hidden`. The id
  // guard keeps the loaded state stable across enrichment events.
  it('should not reset imageReady when the media reference changes but id is the same', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    component.onImageLoad();
    expect(component.imageReady()).toBe(true);

    // Same id, new object reference (typical metadata-cache enrichment).
    const enriched: Media = { ...mockMedia, filename: 'real-name.png' };
    fixture.componentRef.setInput('media', enriched);
    TestBed.tick();
    expect(component.imageReady()).toBe(true);
  });

  it('should show image on error to avoid stuck black screen', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(component.imageReady()).toBe(false);

    component.onImageError();
    expect(component.imageReady()).toBe(true);
  });

  it('should render image element', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('img')).toBeTruthy();
  });

  it('should expose control methods and properties', () => {
    expect(component.rotateLeft).toBeDefined();
    expect(component.rotateRight).toBeDefined();
    expect(component.resetView).toBeDefined();
    expect(component.onZoomInput).toBeDefined();
    expect(component.minZoom).toBe(1);
    expect(component.maxZoom).toBe(5);
  });

  it('should reset view', () => {
    component.zoom.set(2);
    component.rotation.set(90);
    component.resetView();
    expect(component.zoom()).toBe(1);
    expect(component.rotation()).toBe(0);
  });

  it('should rotate left', () => {
    component.rotateLeft();
    expect(component.rotation()).toBe(-90);
  });

  it('should rotate right', () => {
    component.rotateRight();
    expect(component.rotation()).toBe(90);
  });

  it('should generate transform string', () => {
    expect(component.imageTransform).toContain('scale(1)');
    expect(component.imageTransform).toContain('rotate(0deg)');
  });

  /**
   * Item 10 of the v2 patch-embedder plan (docs/plans/patch-embedder.md):
   * pure-function coverage for the screen↔image coordinate transform under
   * non-trivial pan / zoom / rotate. The transform is what lets the user draw
   * a box that stays anchored on the right image pixels regardless of how the
   * viewport is currently transformed; this is the math that backs that.
   */
  describe('screenToImageNormalized (coord transform)', () => {
    // Helper: wire up a 100×100 wrap centred at screen (50, 50) with a 100×100
    // rendered image. Lets each test focus on the transform math, not on DOM
    // plumbing.
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
    }

    function makeEvent(clientX: number, clientY: number): MouseEvent {
      return { clientX, clientY } as MouseEvent;
    }

    it('returns null when the image has not been laid out yet', () => {
      // renderedW/H still 0 → no transform possible.
      expect(component.screenToImageNormalized(makeEvent(50, 50))).toBeNull();
    });

    it('maps the wrap centre to the image centre at identity', () => {
      setupWrap(component);
      const local = component.screenToImageNormalized(makeEvent(50, 50));
      expect(local!.x).toBeCloseTo(0.5, 6);
      expect(local!.y).toBeCloseTo(0.5, 6);
    });

    it('maps corners correctly at identity', () => {
      setupWrap(component);
      const tl = component.screenToImageNormalized(makeEvent(0, 0))!;
      const br = component.screenToImageNormalized(makeEvent(100, 100))!;
      expect(tl.x).toBeCloseTo(0, 6);
      expect(tl.y).toBeCloseTo(0, 6);
      expect(br.x).toBeCloseTo(1, 6);
      expect(br.y).toBeCloseTo(1, 6);
    });

    it('compensates for zoom: screen offsets shrink in image coords as zoom grows', () => {
      setupWrap(component);
      component.zoom.set(2);
      // 25px right of centre at 2× zoom should be 12.5px in image coords =
      // 0.125 normalised, so image x = 0.625.
      const local = component.screenToImageNormalized(makeEvent(75, 50))!;
      expect(local.x).toBeCloseTo(0.625, 6);
      expect(local.y).toBeCloseTo(0.5, 6);
    });

    it('compensates for pan: translating the image shifts the inferred image coords', () => {
      setupWrap(component);
      component.zoom.set(2);
      // Image translated 20px right; a click at screen 70 used to map to image
      // 0.7 at zoom 2 (50px = 0.5 + 20/100/2 * 2... see math). Concretely:
      // dx = 70 - 50 - 20 = 0; sx = 0; image x = 0.5.
      component.panX.set(20);
      const local = component.screenToImageNormalized(makeEvent(70, 50))!;
      expect(local.x).toBeCloseTo(0.5, 6);
      expect(local.y).toBeCloseTo(0.5, 6);
    });

    it('inverts rotation: a click on the screen-right edge maps to the image-top edge at 90° CW', () => {
      setupWrap(component);
      component.rotation.set(90);
      // Positive rotation rotates the image clockwise, so screen-right is image-top.
      const local = component.screenToImageNormalized(makeEvent(100, 50))!;
      expect(local.x).toBeCloseTo(0.5, 5);
      expect(local.y).toBeCloseTo(0, 5);
    });

    it('compensates for an ancestor CSS zoom (visual getBoundingClientRect vs layout coords)', () => {
      // Regression: the app renders <html> at `zoom: 1.1` (styles.scss). A CSS
      // `zoom` scales an element's rendered box without changing its layout box,
      // so getBoundingClientRect() and MouseEvent client coords report VISUAL
      // pixels while clientWidth / renderedW / panX are LAYOUT pixels. Here the
      // wrap is 100 layout px wide but 110 visual px (Z = 1.1). Without the
      // visual→layout conversion, every offset from the wrap centre came out
      // 1.1× too large, drawing the box further from the view centre than the
      // cursor (proportional to distance from centre, sign-flipping across it).
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          clientWidth: 100,
          clientHeight: 100,
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 110,
            height: 110,
            right: 110,
            bottom: 110,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
      // Visual centre is at (55, 55). A click 11 visual px right of centre is
      // 10 layout px → 0.1 normalised, so image x = 0.6, y stays 0.5.
      const local = component.screenToImageNormalized(makeEvent(66, 55))!;
      expect(local.x).toBeCloseTo(0.6, 6);
      expect(local.y).toBeCloseTo(0.5, 6);
    });

    it('combines pan + zoom + rotate self-consistently', () => {
      setupWrap(component);
      component.zoom.set(2);
      component.panX.set(10);
      component.panY.set(-5);
      component.rotation.set(90);
      // Map a couple of points and verify the transform is invertible:
      // taking two points on screen and rotating by the matching angle, the
      // image-coord differences should respect rotation (a screen-x delta
      // becomes an image-y delta at +90°).
      const a = component.screenToImageNormalized(makeEvent(50, 50))!;
      const b = component.screenToImageNormalized(makeEvent(60, 50))!;
      // Screen Δx = +10 → image Δy = -10/zoom/renderedH = -0.05 (the inverse
      // rotation maps +x → -y); Δx in image coords ≈ 0.
      expect(b.x - a.x).toBeCloseTo(0, 5);
      expect(b.y - a.y).toBeCloseTo(-0.05, 5);
    });
  });

  /**
   * Item 11 of the v2 patch-embedder plan: a box drawn at one zoom level
   * stays anchored on the same image pixels when the user zooms in/out.
   * The box is stored in normalised image coordinates and the CSS overlay
   * lives inside `.region-stage` which is rotated/scaled with the image
   * transform, so the box should remain visually identical relative to the
   * image regardless of zoom.
   */
  describe('region box coord stability', () => {
    it('does not mutate the box coords when zoom changes', () => {
      component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
      component.zoom.set(2);
      expect(component.regionBox()).toEqual([0.1, 0.2, 0.5, 0.6]);
      component.zoom.set(4);
      expect(component.regionBox()).toEqual([0.1, 0.2, 0.5, 0.6]);
      component.zoom.set(1);
      expect(component.regionBox()).toEqual([0.1, 0.2, 0.5, 0.6]);
    });

    it('keeps regionBoxStyle (percent-of-stage) stable across zoom changes', () => {
      component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
      const before = component.regionBoxStyle;
      component.zoom.set(3);
      expect(component.regionBoxStyle).toEqual(before);
      component.rotation.set(45);
      expect(component.regionBoxStyle).toEqual(before);
    });

    it('returns null style when no box is drawn', () => {
      component.regionBox.set(null);
      expect(component.regionBoxStyle).toBeNull();
    });
  });

  /**
   * Item 13 of the v2 patch-embedder plan: a subsequent zero-area Shift-drag
   * click does not clear an already-drawn box. (Before the v2 sticky-armed-
   * state work this was a real bug: tooSmall release nuked the prior box.)
   */
  describe('region box preservation on zero-area Shift-drag', () => {
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
    }

    it('keeps the prior box when a Shift-click resolves to zero area', () => {
      setupWrap(component);
      const original: RegionBox = [0.2, 0.3, 0.6, 0.7];
      component.regionBox.set(original);
      component.shiftHeld.set(true);

      let lastEmitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (lastEmitted = v));

      // Click at (10, 10) with Shift held; mousedown then mouseup at same point.
      const ev: MouseEvent = {
        button: 0,
        clientX: 10,
        clientY: 10,
        preventDefault: () => {},
      } as unknown as MouseEvent;
      component.onMouseDown(ev);
      // During the click regionBox transiently becomes the zero-area anchor;
      // mouseup with no motion should restore the original box.
      (component as unknown as { onWindowMouseUp: () => void }).onWindowMouseUp();

      expect(component.regionBox()).toEqual(original);
      // Restored to a state the parent already knew; no emit needed.
      expect(lastEmitted).toBeUndefined();
    });

    it('leaves the box null when the canvas was already empty and the click is zero-area', () => {
      setupWrap(component);
      component.regionBox.set(null);
      component.shiftHeld.set(true);

      const ev: MouseEvent = {
        button: 0,
        clientX: 25,
        clientY: 25,
        preventDefault: () => {},
      } as unknown as MouseEvent;
      component.onMouseDown(ev);
      (component as unknown as { onWindowMouseUp: () => void }).onWindowMouseUp();

      expect(component.regionBox()).toBeNull();
    });
  });

  /**
   * Resize handles flip past the opposite edge instead of hitting a wall, matching
   * the draw flow: drag the west handle past the east edge and the dragged handle
   * becomes the new east edge (and parallel logic for corners).
   */
  describe('resize handle crossing (flip instead of wall)', () => {
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
    }

    function mouseEventAt(x: number, y: number): MouseEvent {
      return {
        button: 0,
        clientX: x,
        clientY: y,
        preventDefault: () => {},
        stopPropagation: () => {},
      } as unknown as MouseEvent;
    }

    type Internals = {
      onWindowMouseMove: (e: MouseEvent) => void;
      onWindowMouseUp: () => void;
    };

    it('flips the box when the west handle is dragged past the east edge', () => {
      setupWrap(component);
      component.regionBox.set([0.2, 0.3, 0.6, 0.7]);

      let lastEmitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (lastEmitted = v));

      component.onResizeHandleMouseDown('w', mouseEventAt(20, 30));
      // Drag the west handle to x=0.9, well past the east edge at 0.6.
      (component as unknown as Internals).onWindowMouseMove(mouseEventAt(90, 50));
      (component as unknown as Internals).onWindowMouseUp();

      // The dragged handle became the new east edge; the former east edge is now west.
      expect(component.regionBox()).toEqual([0.6, 0.3, 0.9, 0.7]);
      expect(lastEmitted).toEqual([0.6, 0.3, 0.9, 0.7]);
    });

    it('flips both axes when a corner handle is dragged past the opposite corner', () => {
      setupWrap(component);
      component.regionBox.set([0.2, 0.3, 0.6, 0.7]);

      // nw controls (x0, y0); their anchors are the start east (0.6) and south (0.7)
      // edges. Drag to (0.9, 0.95) past both.
      component.onResizeHandleMouseDown('nw', mouseEventAt(20, 30));
      (component as unknown as Internals).onWindowMouseMove(mouseEventAt(90, 95));
      (component as unknown as Internals).onWindowMouseUp();

      expect(component.regionBox()).toEqual([0.6, 0.7, 0.9, 0.95]);
    });

    it('restores the pre-resize box when a handle collapses it to zero area', () => {
      setupWrap(component);
      const original: RegionBox = [0.2, 0.3, 0.6, 0.7];
      component.regionBox.set(original);

      let lastEmitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (lastEmitted = v));

      // Release the east handle exactly on the west edge (0.2): zero width.
      component.onResizeHandleMouseDown('e', mouseEventAt(60, 50));
      (component as unknown as Internals).onWindowMouseMove(mouseEventAt(20, 50));
      (component as unknown as Internals).onWindowMouseUp();

      expect(component.regionBox()).toEqual(original);
      // Restored to a state the parent already knew; no emit needed.
      expect(lastEmitted).toBeUndefined();
    });
  });

  describe('marquee mode toggle', () => {
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
    }

    it('starts off and flips on toggle', () => {
      expect(component.marqueeMode()).toBe(false);
      component.toggleMarqueeMode();
      expect(component.marqueeMode()).toBe(true);
      component.toggleMarqueeMode();
      expect(component.marqueeMode()).toBe(false);
    });

    it('reports regionDrawActive when either Shift is held or marquee is on', () => {
      expect(component.regionDrawActive).toBe(false);
      component.shiftHeld.set(true);
      expect(component.regionDrawActive).toBe(true);
      component.shiftHeld.set(false);
      component.marqueeMode.set(true);
      expect(component.regionDrawActive).toBe(true);
    });

    it('shows a crosshair cursor when marquee mode is on', () => {
      expect(component.wrapCursor).not.toBe('crosshair');
      component.marqueeMode.set(true);
      expect(component.wrapCursor).toBe('crosshair');
    });

    it('starts a draw-drag on mousedown when marquee mode is on (no Shift required)', () => {
      setupWrap(component);
      component.marqueeMode.set(true);

      const ev: MouseEvent = {
        button: 0,
        clientX: 30,
        clientY: 30,
        preventDefault: () => {},
      } as unknown as MouseEvent;
      component.onMouseDown(ev);

      // Mid-drag the box should already exist as the zero-area anchor.
      expect(component.regionBox()).not.toBeNull();
      expect(component.regionBox()![0]).toBeCloseTo(0.3, 5);
      expect(component.regionBox()![1]).toBeCloseTo(0.3, 5);
    });

    it('persists across media changes', () => {
      fixture.componentRef.setInput('media', mockMedia);
      TestBed.tick();
      component.marqueeMode.set(true);
      const next: Media = { ...mockMedia, id: 99, filename: 'b.png' };
      fixture.componentRef.setInput('media', next);
      TestBed.tick();
      expect(component.marqueeMode()).toBe(true);
    });
  });

  // The sideways half of this gesture has always worked, because the letterbox
  // columns beside a tall image are still inside `.image-wrap`. Below the image
  // there is no such slack: the toolbar, vote row and metadata tray are siblings
  // under `.center-panel`, so the panel delegates their mousedowns here.
  describe('off-canvas draw start (tryStartOffCanvasDraw)', () => {
    // The wrap is 100x100 at the viewport origin; anything the panel forwards
    // is by construction outside it, so `contains` is always false here.
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW.set(100);
      component.renderedH.set(100);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          contains: () => false,
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);
    }

    function eventAt(x: number, y: number, target: HTMLElement, prevented: { value: boolean }): MouseEvent {
      return {
        button: 0,
        clientX: x,
        clientY: y,
        target,
        preventDefault: () => {
          prevented.value = true;
        },
      } as unknown as MouseEvent;
    }

    let prevented: { value: boolean };
    let plainTarget: HTMLElement;

    beforeEach(() => {
      setupWrap(component);
      prevented = { value: false };
      plainTarget = document.createElement('div');
    });

    it('anchors at the nearest point on the image for a drag started BELOW it', () => {
      component.shiftHeld.set(true);
      // 40px below the wrap's bottom edge, a third of the way across.
      component.tryStartOffCanvasDraw(eventAt(30, 140, plainTarget, prevented));

      expect(component.regionBox()).not.toBeNull();
      expect(component.regionBox()![0]).toBeCloseTo(0.3, 5);
      expect(component.regionBox()![1]).toBeCloseTo(1, 5);
    });

    it('anchors at the nearest point for a drag started ABOVE it', () => {
      component.shiftHeld.set(true);
      component.tryStartOffCanvasDraw(eventAt(70, -25, plainTarget, prevented));

      expect(component.regionBox()![0]).toBeCloseTo(0.7, 5);
      expect(component.regionBox()![1]).toBeCloseTo(0, 5);
    });

    it('preventDefaults so the drag paints no native text selection', () => {
      component.shiftHeld.set(true);
      component.tryStartOffCanvasDraw(eventAt(30, 140, plainTarget, prevented));
      expect(prevented.value).toBe(true);
    });

    it('works from the sticky Marquee toggle too, with no Shift held', () => {
      component.marqueeMode.set(true);
      component.tryStartOffCanvasDraw(eventAt(50, 160, plainTarget, prevented));
      expect(component.regionBox()).not.toBeNull();
    });

    it('does nothing when neither Shift nor marquee mode is active', () => {
      component.tryStartOffCanvasDraw(eventAt(30, 140, plainTarget, prevented));
      expect(component.regionBox()).toBeNull();
      expect(prevented.value).toBe(false);
    });

    // Every click on a control below the image keeps doing exactly what it did
    // before — including while the sticky Marquee toggle is on, when there is no
    // modifier to distinguish "draw" from "press the Good button".
    it('leaves interactive controls alone', () => {
      component.marqueeMode.set(true);
      const button = document.createElement('button');
      component.tryStartOffCanvasDraw(eventAt(30, 140, button, prevented));
      expect(component.regionBox()).toBeNull();
      expect(prevented.value).toBe(false);
    });

    // The tray is the bottom of the panel, far from the image and full of text
    // people select by dragging; it is out of the draw zone entirely, prose and
    // controls alike.
    it('leaves the whole metadata tray alone, not just its controls', () => {
      component.marqueeMode.set(true);
      const tray = document.createElement('div');
      tray.className = 'metadata-tray';
      const value = document.createElement('span');
      value.className = 'metadata-value';
      tray.appendChild(value);
      component.tryStartOffCanvasDraw(eventAt(30, 190, value, prevented));
      expect(component.regionBox()).toBeNull();
      expect(prevented.value).toBe(false);
    });

    it('leaves a control alone when the mousedown lands on a child of it', () => {
      component.marqueeMode.set(true);
      const button = document.createElement('button');
      const icon = document.createElement('span');
      button.appendChild(icon);
      component.tryStartOffCanvasDraw(eventAt(30, 140, icon, prevented));
      expect(component.regionBox()).toBeNull();
    });

    it('ignores an event that bubbled up from inside the canvas', () => {
      component.shiftHeld.set(true);
      (component as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () => ({
        nativeElement: {
          contains: () => true,
          getBoundingClientRect: () => ({
            left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>);

      component.tryStartOffCanvasDraw(eventAt(30, 30, plainTarget, prevented));
      // The wrap's own handler already owns this one; starting a second draw
      // here would overwrite `previousBox` with the anchor it just set.
      expect(component.regionBox()).toBeNull();
    });

    it('ignores a non-primary button', () => {
      component.shiftHeld.set(true);
      const ev = { ...eventAt(30, 140, plainTarget, prevented), button: 2 } as unknown as MouseEvent;
      component.tryStartOffCanvasDraw(ev);
      expect(component.regionBox()).toBeNull();
    });

    it('restores the prior box on a zero-area off-canvas click', () => {
      component.regionBox.set([0.1, 0.1, 0.6, 0.6]);
      component.shiftHeld.set(true);
      component.tryStartOffCanvasDraw(eventAt(30, 140, plainTarget, prevented));
      window.dispatchEvent(new MouseEvent('mouseup'));
      expect(component.regionBox()).toEqual([0.1, 0.1, 0.6, 0.6]);
    });
  });

  describe('best-match highlight overlay', () => {
    it('starts off and flips on toggle', () => {
      expect(component.highlightMode()).toBe(false);
      component.toggleHighlightMode();
      expect(component.highlightMode()).toBe(true);
      component.toggleHighlightMode();
      expect(component.highlightMode()).toBe(false);
    });

    it('is not visible until both the toggle is on and a box is present', () => {
      expect(component.highlightVisible).toBe(false);
      component.highlightMode.set(true);
      expect(component.highlightVisible).toBe(false); // no box yet
      fixture.componentRef.setInput('highlightBox', [0.1, 0.2, 0.6, 0.7]);
      expect(component.highlightVisible).toBe(true);
    });

    it('positions the box as percent-of-stage', () => {
      fixture.componentRef.setInput('highlightBox', [0.1, 0.2, 0.6, 0.8]);
      expect(component.highlightBoxStyle).toEqual({
        left: '10.000%',
        top: '20.000%',
        width: '50.000%',
        height: '60.000%',
      });
    });

    it('rejects malformed, degenerate, and near-full-image boxes', () => {
      fixture.componentRef.setInput('highlightBox', null);
      expect(component.highlightBoxStyle).toBeNull();
      fixture.componentRef.setInput('highlightBox', [0.5, 0.5, 0.4, 0.6]); // x1 < x0 -> zero/negative width
      expect(component.highlightBoxStyle).toBeNull();
      fixture.componentRef.setInput('highlightBox', [0, 0, 1, 1]); // whole-image fallback
      expect(component.highlightBoxStyle).toBeNull();
      fixture.componentRef.setInput('highlightBox', [0, 0, NaN, 0.5]);
      expect(component.highlightBoxStyle).toBeNull();
    });

    it('persists the toggle across media changes', () => {
      fixture.componentRef.setInput('media', mockMedia);
      TestBed.tick();
      component.highlightMode.set(true);
      const next: Media = { ...mockMedia, id: 77, filename: 'c.png' };
      fixture.componentRef.setInput('media', next);
      TestBed.tick();
      expect(component.highlightMode()).toBe(true);
    });
  });

  describe('armed-confirm cancel routing', () => {
    it('emits armedConfirmCanceled instead of clearing the box when Esc is pressed while armed', () => {
      component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
      fixture.componentRef.setInput('pendingBadConfirm', true);
      let canceled = false;
      let cleared = false;
      component.armedConfirmCanceled.subscribe(() => (canceled = true));
      component.regionBoxChange.subscribe((v) => {
        if (v === null) cleared = true;
      });
      const esc = new KeyboardEvent('keydown', { key: 'Escape' });
      (component as unknown as { onWindowKeyDown: (e: KeyboardEvent) => void }).onWindowKeyDown(esc);
      expect(canceled).toBe(true);
      expect(cleared).toBe(false);
      expect(component.regionBox()).toEqual([0.1, 0.2, 0.5, 0.6]);
    });

    it('clears the box on Esc when no armed confirm is pending', () => {
      component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
      fixture.componentRef.setInput('pendingBadConfirm', false);
      let emitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (emitted = v));
      const esc = new KeyboardEvent('keydown', { key: 'Escape' });
      (component as unknown as { onWindowKeyDown: (e: KeyboardEvent) => void }).onWindowKeyDown(esc);
      expect(component.regionBox()).toBeNull();
      expect(emitted).toBeNull();
    });
  });
});

describe('ImageViewerComponent Escape guard', () => {
  let component: ImageViewerComponent;

  beforeEach(async () => {
    await configureZoneless({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    const fixture = TestBed.createComponent(ImageViewerComponent);
    component = fixture.componentInstance;
  });

  function pressEscape(): void {
    (component as unknown as { onWindowKeyDown(e: KeyboardEvent): void }).onWindowKeyDown(
      new KeyboardEvent('keydown', { key: 'Escape' }),
    );
  }

  it('does not clear the region box while a modal is open', () => {
    // Regression: the Esc that closes a modal used to also fall through to
    // this window-level handler and discard the user's drawn box.
    component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop';
    document.body.appendChild(backdrop);
    try {
      pressEscape();
      expect(component.regionBox()).toEqual([0.1, 0.2, 0.5, 0.6]);
    } finally {
      backdrop.remove();
    }
  });

  it('clears the region box on Escape when no modal is open', () => {
    component.regionBox.set([0.1, 0.2, 0.5, 0.6]);
    pressEscape();
    expect(component.regionBox()).toBeNull();
  });
});

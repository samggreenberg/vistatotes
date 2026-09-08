import { ChangeDetectionStrategy, Component, computed, effect, ElementRef, inject, input, OnDestroy, output, signal, untracked, viewChild } from '@angular/core';
import { NgStyle } from '@angular/common';
import { Media, PayloadVariant } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

export type RegionBox = readonly [number, number, number, number];
export type ResizeHandle = 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se';

type DragMode =
  | { kind: 'pan'; startX: number; startY: number; originX: number; originY: number }
  | { kind: 'draw'; anchor: { x: number; y: number }; previousBox: RegionBox | null }
  | { kind: 'move'; startLocal: { x: number; y: number }; startBox: RegionBox }
  | { kind: 'resize'; handle: ResizeHandle; startBox: RegionBox };

const MIN_BOX_SIZE = 0.01; // 1% of the image; below this we treat a draw as a stray click

// What a region draw started from outside the canvas does NOT claim (see
// `tryStartOffCanvasDraw`). Two different reasons:
//
//   - Interactive controls keep their own click semantics. Without this, turning
//     the sticky Marquee toggle on would make the Good/Bad buttons and the zoom
//     slider unusable, because every mousedown on them would anchor a box.
//   - `.metadata-tray` is excluded wholesale, controls and prose alike. It sits at
//     the very bottom of the panel, far enough from the image that a drag begun
//     there reads as "select this text", not as "box that corner"; and selecting a
//     filename or an MD5 by dragging across it is a thing people actually do.
const OFF_CANVAS_DRAW_EXCLUDED =
  'button, input, select, textarea, a, label, [role="button"], [contenteditable], .metadata-tray';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-image-viewer',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './image-viewer.component.html',
  styleUrl: './image-viewer.component.scss',
})
export class ImageViewerComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);

  readonly media = input.required<Media>();
  /**
   * Which payload to show: `''` (canonical) or `'original'` (the pre-clean
   * snapshot of an item a cleaner rewrote at load time).  The parent's
   * Clean/Original toggle drives it; it only ever leaves `''` for media whose
   * `has_original` flag is set.
   */
  readonly variant = input<PayloadVariant>('');
  /**
   * True while the parent is in the v2 "bad-vote-with-box discard confirm" armed state.
   * The viewer uses it to (a) render the box with a sticky red pulse, and (b) route Esc /
   * mouse-on-box back to the parent via `armedConfirmCanceled` instead of clearing the box.
   */
  readonly pendingBadConfirm = input(false);
  /**
   * The region the detector/embedder matched best at inference, as a normalised
   * ``[x0, y0, x1, y1]`` box (the argmax over patch regions that produced this
   * media's score).  ``null`` when the active dataset isn't patch-region-aware
   * or the focused media hasn't been scored.  Rendered as a neutral white/black
   * dashed overlay (distinct from the user's solid yellow voting box) only while
   * the Highlight toggle (``highlightMode``) is on; purely informational, never
   * interactive (no drag/resize, no vote semantics).
   */
  readonly highlightBox = input<RegionBox | null>(null);
  readonly regionBoxChange = output<RegionBox | null>();
  /**
   * Fired when the user does something that cancels the armed bad-vote-confirm without
   * voting (Esc while armed, or any mousedown on the box body/handles, or starting a
   * fresh Shift-drag). The parent clears its armed state but keeps the box.
   */
  readonly armedConfirmCanceled = output<void>();

  // Public: the coord-transform specs substitute a stub wrap element here.
  readonly wrapRef = viewChild<ElementRef<HTMLDivElement>>('imageWrap');
  private readonly imageRef = viewChild<ElementRef<HTMLImageElement>>('imageEl');

  // Signals: written from the media-change effect below and read in the
  // template, so plain fields would leave the view stale under zoneless.
  readonly imageSrc = signal('');
  readonly imageReady = signal(false);
  // zoom/rotation are signals: they are written from the keyboard-shortcut
  // dispatch (an un-bound RxJS callback in CenterPanelComponent) and from the
  // parent's toolbar buttons (click handlers bound in the PARENT's template,
  // which mark only the parent dirty) — with a plain field, this OnPush view's
  // transform binding silently went stale until some unrelated CD repainted it.
  readonly zoom = signal(1);
  readonly rotation = signal(0);
  readonly zoomLabel = computed(() => {
    const z = this.zoom();
    return (z === Math.floor(z) ? z.toFixed(0) : z.toFixed(1)) + '×';
  });
  // Track the id of the media we last reset for. The `media` input reference
  // changes whenever `MediaMetadataCacheService` hydrates richer metadata for
  // the same id; re-running the media-change effect for those enrichments would clobber
  // `imageReady=true` back to false and, since `imageSrc` is the same string,
  // Angular wouldn't re-fire the `<img>` load event, leaving the canvas
  // permanently hidden behind `visibility: hidden`.
  private lastMediaId: number | null = null;
  // Same guard for the payload variant: flipping Clean/Original keeps the media
  // id, so the effect has to notice the variant changed to refetch the image.
  private lastVariant: PayloadVariant = '';

  // Region voting state (v2 of the patch-embedder plan, UI only; see docs/plans/patch-embedder.md).
  // These are signals because they are written from un-patched callbacks (window
  // mouse/key listeners, the shake `setTimeout`) that schedule no change detection
  // under zoneless; a signal write read in the template is on the notification path.
  readonly regionBox = signal<RegionBox | null>(null);
  readonly regionBoxShake = signal(false);
  readonly shiftHeld = signal(false);
  // Sticky toggle exposed by the Marquee button in .image-view-controls. While true the
  // viewer behaves as if Shift were held: cursor is a crosshair and a left-drag draws a
  // new region instead of panning. Shift+drag remains a power-user shortcut even when
  // the toggle is off; toggling the button just turns the gesture on without a modifier.
  // Signal for the same parent-toolbar reason as `zoom`: toggled from the
  // parent's template, read here (crosshair cursor / region-mode class).
  readonly marqueeMode = signal(false);
  // Sticky toggle exposed by the Highlight button in .image-view-controls. While
  // true the viewer overlays a neutral white/black dashed box (`highlightBox`)
  // around the region the detector matched best at inference. Independent of marqueeMode -
  // both can be on at once (the highlight is read-only and sits behind the
  // interactive voting box).
  readonly highlightMode = signal(false);
  // renderedW/H are written from a raw ResizeObserver callback (un-patched, no CD
  // under zoneless), so they are signals; reading them in the template keeps the
  // overlay geometry fresh when the wrap resizes.
  readonly renderedW = signal(0);
  readonly renderedH = signal(0);

  // panX/panY are signals (written from the window-level mousemove drag handler,
  // an un-patched callback). They are not `private` so tests can drive
  // screenToImageNormalized() with non-zero pan without simulating a full wheel +
  // drag sequence.
  readonly panX = signal(0);
  readonly panY = signal(0);
  private drag: DragMode | null = null;

  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private mouseUpHandler: (() => void) | null = null;
  private keyDownHandler: ((e: KeyboardEvent) => void) | null = null;
  private keyUpHandler: ((e: KeyboardEvent) => void) | null = null;
  private blurHandler: (() => void) | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private shakeTimer: ReturnType<typeof setTimeout> | null = null;

  readonly minZoom = 1;
  readonly maxZoom = 5;
  readonly zoomStep = 0.05;

  constructor() {
    this.setupWindowKeyListeners();

    // Reset the viewport for a new image when the media id changes. The id
    // guard skips same-id metadata-enrichment reference changes (see
    // lastMediaId above); zoom/rotation/marquee state intentionally persists.
    effect(() => {
      const media = this.media();
      const variant = this.variant();
      if (media.id === this.lastMediaId && variant === this.lastVariant) return;
      const sameMedia = media.id === this.lastMediaId;
      this.lastMediaId = media.id;
      this.lastVariant = variant;
      this.imageReady.set(false);
      this.imageSrc.set(this.activeContext.mediaUrl(`/api/medias/${media.id}/image`, { variant }));
      // A variant flip is the same item shown differently: keep the user's
      // zoom / pan and their voting box instead of resetting as for a new item.
      if (sameMedia) return;
      untracked(() => {
        this.resetView();
        this.clearRegionBox({ emit: true });
      });
    });
  }

  onImageLoad(): void {
    this.imageReady.set(true);
    this.recomputeRenderedSize();
    this.attachWrapResizeObserver();
  }

  onImageError(): void {
    this.imageReady.set(true);
  }

  ngOnDestroy(): void {
    this.removeWindowMouseListeners();
    this.removeWindowKeyListeners();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
  }

  onZoomInput(event: Event): void {
    this.zoom.set(parseFloat((event.target as HTMLInputElement).value));
    this.applyTransform();
  }

  rotateLeft(): void {
    this.rotation.update((r) => r - 90);
    this.applyTransform();
  }

  rotateRight(): void {
    this.rotation.update((r) => r + 90);
    this.applyTransform();
  }

  zoomIn(): void {
    this.zoom.update((z) => this.clampZoom(z + 0.15 * z));
    this.applyTransform();
  }

  zoomOut(): void {
    this.zoom.update((z) => this.clampZoom(z - 0.15 * z));
    this.applyTransform();
  }

  resetView(): void {
    this.zoom.set(1);
    this.rotation.set(0);
    this.panX.set(0);
    this.panY.set(0);
    this.applyTransform();
  }

  onWheel(event: WheelEvent): void {
    event.preventDefault();
    const oldZoom = this.zoom();
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    this.zoom.set(this.clampZoom(oldZoom + delta * oldZoom));

    const wrap = this.wrapRef()?.nativeElement;
    if (wrap) {
      const rect = wrap.getBoundingClientRect();
      const s = this.layoutScale(wrap, rect);
      const cx = (event.clientX - rect.left - rect.width / 2) / s;
      const cy = (event.clientY - rect.top - rect.height / 2) / s;
      const ratio = this.zoom() / oldZoom;
      this.panX.set(cx - ratio * (cx - this.panX()));
      this.panY.set(cy - ratio * (cy - this.panY()));
    }
    this.applyTransform();
  }

  /** True when a drag should draw a region (either Shift-held or Marquee toggle on). */
  get regionDrawActive(): boolean {
    return this.shiftHeld() || this.marqueeMode();
  }

  toggleMarqueeMode(): void {
    this.marqueeMode.update((v) => !v);
  }

  toggleHighlightMode(): void {
    this.highlightMode.update((v) => !v);
  }

  /** True when the best-match highlight box should be drawn over the image. */
  get highlightVisible(): boolean {
    return this.highlightMode() && this.highlightBoxStyle !== null;
  }

  /** Percent-position style for the best-match highlight overlay.  Returns null
   *  when the box is missing, malformed, degenerate, or covers (effectively) the
   *  whole image, so neither the near-full single-vector fallback box nor a
   *  patch dataset's winning image-level row paints a frame round everything.  A
   *  patch dataset's other outcome - one grid cell - is small but deliberate:
   *  it points at the exact patch the detector scored highest.  (This overlay is
   *  the only place a best-match region is drawn - thumbnails never render a
   *  best-region outline.) */
  get highlightBoxStyle(): { [k: string]: string } | null {
    const box = this.highlightBox();
    if (!box || box.length !== 4) return null;
    const [x0, y0, x1, y1] = box;
    if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) return null;
    const w = x1 - x0;
    const h = y1 - y0;
    if (w <= 0 || h <= 0) return null;
    if (w >= 0.99 && h >= 0.99) return null;
    return {
      left: pct(x0),
      top: pct(y0),
      width: pct(w),
      height: pct(h),
    };
  }

  onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;

    // Drag from anywhere on the canvas (Shift-held or marquee mode) starts a fresh box.
    if (this.beginDraw(event)) return;

    // Default: pan-when-zoomed.
    const max = this.getMaxPan();
    if (max.x <= 0 && max.y <= 0) return;
    this.drag = {
      kind: 'pan',
      startX: event.clientX,
      startY: event.clientY,
      originX: this.panX(),
      originY: this.panY(),
    };
    event.preventDefault();
    this.setupWindowMouseListeners();
  }

  /** Start a region draw from a point OUTSIDE the image canvas.
   *
   *  The horizontal case has always worked by accident of layout: the image is
   *  ``object-fit: contain`` inside a full-width ``.image-wrap``, so for a tall
   *  image the letterbox columns beside it are still *inside* the wrap and hit
   *  `onMouseDown` above; `screenToImageNormalized` returns ``x < 0`` and
   *  `clamp01` snaps the anchor to the nearest edge. Vertically there is no such
   *  slack - the wrap is exactly as tall as the media area, and below it sit the
   *  toolbar, the vote buttons and the metadata tray as SIBLINGS under
   *  ``.center-panel``, so a mousedown there never reached this component at all
   *  (it just started a native text selection over the buttons).
   *
   *  This is the same gesture, delegated from the panel. Nothing downstream needs
   *  to change: the anchor is clamped exactly as the sideways case is, and the
   *  move/up listeners were already on ``window``, so dragging out of the canvas
   *  has always worked once a drag was under way. Only the *start* was gated.
   *
   *  Interactive controls are deliberately excluded - a click on the Good/Bad
   *  buttons, the zoom slider or the metadata toggle keeps doing exactly what it
   *  does today, including while the sticky Marquee toggle is on. */
  tryStartOffCanvasDraw(event: MouseEvent): void {
    if (event.button !== 0 || !this.regionDrawActive) return;
    const target = event.target as HTMLElement | null;
    const wrap = this.wrapRef()?.nativeElement;
    // Inside the canvas the wrap's own handler owns the gesture; this delegated
    // one sees the bubbled event afterwards and must not start a second draw.
    if (!wrap || !target || wrap.contains(target)) return;
    if (target.closest?.(OFF_CANVAS_DRAW_EXCLUDED)) return;
    this.beginDraw(event);
  }

  /** Anchor a fresh draw at `event`, clamped to the image. Returns false (leaving
   *  the caller to fall through to its own default) when region-draw isn't active
   *  or the image isn't laid out yet. */
  private beginDraw(event: MouseEvent): boolean {
    if (!this.regionDrawActive || this.renderedW() <= 0 || this.renderedH() <= 0) return false;
    const local = this.screenToImageNormalized(event);
    if (!local) return false;
    const x = clamp01(local.x);
    const y = clamp01(local.y);
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    // Remember the prior box so we can restore it on a zero-area release;
    // a stray Shift-click on empty space must not throw away real work.
    this.drag = { kind: 'draw', anchor: { x, y }, previousBox: this.regionBox() };
    this.regionBox.set([x, y, x, y]);
    // Also what stops the drag from painting a native text selection across the
    // buttons and labels it passes over.
    event.preventDefault();
    this.setupWindowMouseListeners();
    return true;
  }

  onRegionBodyMouseDown(event: MouseEvent): void {
    const box = this.regionBox();
    if (event.button !== 0 || !box) return;
    event.stopPropagation();
    event.preventDefault();
    const local = this.screenToImageNormalized(event);
    if (!local) return;
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'move', startLocal: local, startBox: box };
    this.setupWindowMouseListeners();
  }

  onResizeHandleMouseDown(handle: ResizeHandle, event: MouseEvent): void {
    const box = this.regionBox();
    if (event.button !== 0 || !box) return;
    event.stopPropagation();
    event.preventDefault();
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'resize', handle, startBox: box };
    this.setupWindowMouseListeners();
  }

  /** Clear the current region box and notify the parent. */
  clearRegionBox(opts: { emit: boolean } = { emit: true }): void {
    if (this.regionBox() === null) return;
    this.regionBox.set(null);
    if (opts.emit) this.regionBoxChange.emit(null);
  }

  /** Visually flash the region box (used by bad-vote-confirm flow). */
  pulseRegionBox(): void {
    if (!this.regionBox()) return;
    this.regionBoxShake.set(true);
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
    this.shakeTimer = setTimeout(() => this.regionBoxShake.set(false), 500);
  }

  get imageTransform(): string {
    return `translate(${this.panX()}px, ${this.panY()}px) scale(${this.zoom()}) rotate(${this.rotation()}deg)`;
  }

  get wrapCursor(): string {
    if (this.regionDrawActive) return 'crosshair';
    const max = this.getMaxPan();
    return max.x > 0 || max.y > 0 ? 'grab' : '';
  }

  get regionBoxStyle(): { [k: string]: string } | null {
    const box = this.regionBox();
    if (!box) return null;
    const [x0, y0, x1, y1] = box;
    return {
      left: pct(x0),
      top: pct(y0),
      width: pct(x1 - x0),
      height: pct(y1 - y0),
    };
  }

  // Clamp the pan back into range for the current zoom/rotation; call after
  // any zoom/rotation change. (zoomLabel is a computed off `zoom` now.)
  private applyTransform(): void {
    const max = this.getMaxPan();
    this.panX.set(Math.max(-max.x, Math.min(max.x, this.panX())));
    this.panY.set(Math.max(-max.y, Math.min(max.y, this.panY())));
  }

  private clampZoom(val: number): number {
    return Math.min(this.maxZoom, Math.max(this.minZoom, val));
  }

  private recomputeRenderedSize(): void {
    const img = this.imageRef()?.nativeElement;
    const wrap = this.wrapRef()?.nativeElement;
    if (!img || !wrap) return;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    if (!natW || !natH || !wrapW || !wrapH) {
      this.renderedW.set(0);
      this.renderedH.set(0);
      return;
    }
    const imgAspect = natW / natH;
    const wrapAspect = wrapW / wrapH;
    if (imgAspect > wrapAspect) {
      this.renderedW.set(wrapW);
      this.renderedH.set(wrapW / imgAspect);
    } else {
      this.renderedH.set(wrapH);
      this.renderedW.set(wrapH * imgAspect);
    }
  }

  private attachWrapResizeObserver(): void {
    const wrap = this.wrapRef()?.nativeElement;
    if (!wrap || this.resizeObserver) return;
    if (typeof ResizeObserver === 'undefined') return;
    this.resizeObserver = new ResizeObserver(() => this.recomputeRenderedSize());
    this.resizeObserver.observe(wrap);
  }

  private getMaxPan(): { x: number; y: number } {
    const wrap = this.wrapRef()?.nativeElement;
    const renderedW = this.renderedW();
    const renderedH = this.renderedH();
    if (!wrap || !renderedW || !renderedH) return { x: 0, y: 0 };
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    const rot = ((this.rotation() % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
    const effW = swapped ? renderedH : renderedW;
    const effH = swapped ? renderedW : renderedH;
    return {
      x: Math.max(0, (effW * this.zoom() - wrapW) / 2),
      y: Math.max(0, (effH * this.zoom() - wrapH) / 2),
    };
  }

  /** Ratio between the wrap's on-screen (visual) size and its layout size.
   *  The app renders ``<html>`` at ``zoom: 1.1`` (see styles.scss), and a CSS
   *  ``zoom`` scales an element's rendered box without changing its layout box:
   *  ``getBoundingClientRect()`` and ``MouseEvent`` client coords come back in
   *  VISUAL pixels, while ``clientWidth`` / ``renderedW`` / ``panX`` are LAYOUT
   *  pixels. Dividing a client-space delta by this ratio converts it back to the
   *  layout space the region overlay is drawn in, so the box lands under the
   *  cursor. Returns 1 when there's no zoom (ratio is exactly 1) or the wrap
   *  isn't measurable (e.g. the test mock exposes no ``clientWidth``). */
  private layoutScale(wrap: HTMLElement, rect: DOMRect): number {
    const cw = wrap.clientWidth;
    return cw && rect.width ? rect.width / cw : 1;
  }

  /** Convert a screen-space mouse event to normalised image coords (pre-rotation).
   *  Returns null when the image isn't laid out yet. Public so tests can drive it
   *  with a mocked wrapRef + arbitrary pan/zoom/rotate state. */
  screenToImageNormalized(event: MouseEvent): { x: number; y: number } | null {
    const wrap = this.wrapRef()?.nativeElement;
    const renderedW = this.renderedW();
    const renderedH = this.renderedH();
    if (!wrap || !renderedW || !renderedH) return null;
    const rect = wrap.getBoundingClientRect();
    const s = this.layoutScale(wrap, rect);
    const dx = (event.clientX - (rect.left + rect.width / 2)) / s - this.panX();
    const dy = (event.clientY - (rect.top + rect.height / 2)) / s - this.panY();
    const sx = dx / this.zoom();
    const sy = dy / this.zoom();
    const rad = (-this.rotation() * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const rx = sx * cos - sy * sin;
    const ry = sx * sin + sy * cos;
    return {
      x: (rx + renderedW / 2) / renderedW,
      y: (ry + renderedH / 2) / renderedH,
    };
  }

  private setupWindowMouseListeners(): void {
    this.removeWindowMouseListeners();
    this.mouseMoveHandler = (e: MouseEvent) => this.onWindowMouseMove(e);
    this.mouseUpHandler = () => this.onWindowMouseUp();
    window.addEventListener('mousemove', this.mouseMoveHandler);
    window.addEventListener('mouseup', this.mouseUpHandler);
  }

  private removeWindowMouseListeners(): void {
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler);
      this.mouseMoveHandler = null;
    }
    if (this.mouseUpHandler) {
      window.removeEventListener('mouseup', this.mouseUpHandler);
      this.mouseUpHandler = null;
    }
  }

  private onWindowMouseMove(e: MouseEvent): void {
    if (!this.drag) return;
    const d = this.drag;
    if (d.kind === 'pan') {
      // clientX deltas are VISUAL px (the app renders at zoom: 1.1) but panX is
      // a LAYOUT-px transform value; convert so the image tracks the cursor 1:1.
      const wrap = this.wrapRef()?.nativeElement;
      const s = wrap ? this.layoutScale(wrap, wrap.getBoundingClientRect()) : 1;
      this.panX.set(d.originX + (e.clientX - d.startX) / s);
      this.panY.set(d.originY + (e.clientY - d.startY) / s);
      this.applyTransform();
      return;
    }
    const local = this.screenToImageNormalized(e);
    if (!local) return;
    if (d.kind === 'draw') {
      const ax = d.anchor.x;
      const ay = d.anchor.y;
      const bx = clamp01(local.x);
      const by = clamp01(local.y);
      this.regionBox.set([Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)]);
      return;
    }
    if (d.kind === 'move') {
      const dx = local.x - d.startLocal.x;
      const dy = local.y - d.startLocal.y;
      const [sx0, sy0, sx1, sy1] = d.startBox;
      const w = sx1 - sx0;
      const h = sy1 - sy0;
      const x0 = clamp(sx0 + dx, 0, 1 - w);
      const y0 = clamp(sy0 + dy, 0, 1 - h);
      this.regionBox.set([x0, y0, x0 + w, y0 + h]);
      return;
    }
    // resize: like draw, the dragged handle may cross the opposite edge and
    // flip the box rather than hitting a wall. The edge(s) the handle does not
    // control stay anchored at their start position; the controlled edge follows
    // the cursor, and per-axis min/max re-normalisation swaps which side is which
    // when the cursor crosses over. (e.g. drag the west handle past the east edge
    // and the dragged handle becomes the new east edge.)
    const [sx0, sy0, sx1, sy1] = d.startBox;
    const lx = clamp01(local.x);
    const ly = clamp01(local.y);
    let [x0, y0, x1, y1] = d.startBox;
    if (d.handle.includes('w')) [x0, x1] = [Math.min(lx, sx1), Math.max(lx, sx1)];
    else if (d.handle.includes('e')) [x0, x1] = [Math.min(sx0, lx), Math.max(sx0, lx)];
    if (d.handle.includes('n')) [y0, y1] = [Math.min(ly, sy1), Math.max(ly, sy1)];
    else if (d.handle.includes('s')) [y0, y1] = [Math.min(sy0, ly), Math.max(sy0, ly)];
    this.regionBox.set([x0, y0, x1, y1]);
  }

  private onWindowMouseUp(): void {
    if (!this.drag) {
      this.removeWindowMouseListeners();
      return;
    }
    const drag = this.drag;
    this.drag = null;
    this.removeWindowMouseListeners();
    const box = this.regionBox();
    if (!box) return;
    const [x0, y0, x1, y1] = box;
    const tooSmall = x1 - x0 < MIN_BOX_SIZE || y1 - y0 < MIN_BOX_SIZE;
    if (drag.kind === 'draw' && tooSmall) {
      // Zero-area Shift-drag (a stray click without motion). Restore the
      // prior box rather than discarding it; drawing a box is real work.
      // Don't emit: the parent's last-known state was already previousBox
      // (the transient zero-area draw was never emitted).
      this.regionBox.set(drag.previousBox);
      return;
    }
    if (drag.kind === 'resize' && tooSmall) {
      // The handle was released right at the flip point, collapsing the box to
      // (near) zero area. Restore the pre-resize box; startBox was the parent's
      // last-known state, so no emit is needed.
      this.regionBox.set(drag.startBox);
      return;
    }
    this.regionBoxChange.emit(box);
  }

  private setupWindowKeyListeners(): void {
    this.keyDownHandler = (e: KeyboardEvent) => this.onWindowKeyDown(e);
    this.keyUpHandler = (e: KeyboardEvent) => this.onWindowKeyUp(e);
    this.blurHandler = () => {
      // Releasing focus (alt-tab, etc.) drops the Shift state; don't leave the
      // user stuck in region mode invisibly.
      this.shiftHeld.set(false);
    };
    window.addEventListener('keydown', this.keyDownHandler);
    window.addEventListener('keyup', this.keyUpHandler);
    window.addEventListener('blur', this.blurHandler);
  }

  private removeWindowKeyListeners(): void {
    if (this.keyDownHandler) {
      window.removeEventListener('keydown', this.keyDownHandler);
      this.keyDownHandler = null;
    }
    if (this.keyUpHandler) {
      window.removeEventListener('keyup', this.keyUpHandler);
      this.keyUpHandler = null;
    }
    if (this.blurHandler) {
      window.removeEventListener('blur', this.blurHandler);
      this.blurHandler = null;
    }
  }

  private onWindowKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Shift') {
      this.shiftHeld.set(true);
      return;
    }
    if (e.key !== 'Escape' || this.isTyping()) return;
    // Skip when a modal is open (matching KeyboardService): the same Esc that
    // closes the modal must not also clear the drawn region box underneath.
    if (document.querySelector('.modal-backdrop')) return;
    // Esc while a bad-vote-with-box discard is armed cancels the armed state but
    // keeps the box (per the v2 patch-embedder plan, drawing a box is real work,
    // and Esc should be the "I changed my mind about voting no" out, not "throw
    // away the box"). Only consume the key if we actually had an action to take.
    if (this.pendingBadConfirm()) {
      e.preventDefault();
      this.armedConfirmCanceled.emit();
      return;
    }
    if (this.regionBox()) {
      e.preventDefault();
      this.clearRegionBox({ emit: true });
    }
  }

  private onWindowKeyUp(e: KeyboardEvent): void {
    if (e.key === 'Shift') this.shiftHeld.set(false);
  }

  private isTyping(): boolean {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const type = (el as HTMLInputElement).type;
      if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
    }
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if ((el as HTMLElement).isContentEditable) return true;
    return false;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

function pct(v: number): string {
  return (v * 100).toFixed(3) + '%';
}

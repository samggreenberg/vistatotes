import { ChangeDetectionStrategy, Component, DestroyRef, effect, inject, input, OnDestroy, output, signal, untracked, viewChild } from '@angular/core';
import { KeyValuePipe, TitleCasePipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { EmbedderInfo, Media, PayloadVariant } from '../../models/api.models';
import { MediasApiService } from '../../services/medias-api.service';
import { KeyboardService } from '../../services/keyboard.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { DatasetsListingsApiService } from '../../services/datasets-listings-api.service';
import { AudioPlayerComponent } from './audio-player/audio-player.component';
import { ImageViewerComponent, RegionBox } from './image-viewer/image-viewer.component';
import { VideoPlayerComponent } from './video-player/video-player.component';
import { TextViewerComponent } from './text-viewer/text-viewer.component';
import { DocumentViewerComponent } from './document-viewer/document-viewer.component';
import { VotingOverlayComponent } from './voting-overlay/voting-overlay.component';
import { CopyDetailButtonComponent } from '../copy-detail-button/copy-detail-button.component';
import { IconComponent } from '../icon/icon.component';
import { prefersReducedMotion } from '../../utils/reduced-motion';
import { formatMetadataValue as formatMetadataValueUtil } from '../../utils/format-metadata';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-center-panel',
  standalone: true,
  imports: [
    KeyValuePipe,
    TitleCasePipe,
    AudioPlayerComponent,
    ImageViewerComponent,
    VideoPlayerComponent,
    TextViewerComponent,
    DocumentViewerComponent,
    VotingOverlayComponent,
    CopyDetailButtonComponent,
    IconComponent,
  ],
  templateUrl: './center-panel.component.html',
  styleUrl: './center-panel.component.scss',
})
export class CenterPanelComponent implements OnDestroy {
  private mediasApi = inject(MediasApiService);
  private keyboard = inject(KeyboardService);
  voteState = inject(VoteStateService);
  private settingsState = inject(SettingsStateService);
  private sortState = inject(SortStateService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private destroyRef = inject(DestroyRef);

  readonly media = input<Media | null>(null);
  readonly disabled = input(false);
  readonly mediaVoted = output<{
    id: number;
    vote: 'good' | 'bad';
}>();

  readonly audioPlayer = viewChild(AudioPlayerComponent);
  readonly imageViewer = viewChild(ImageViewerComponent);
  readonly videoPlayer = viewChild(VideoPlayerComponent);

  // These fields are written from non-bound callbacks — the keyboard-shortcut
  // dispatch (`KeyboardService.action$`), HTTP/vote subscriptions, and timers —
  // and read in the template, so under zoneless CD they must be signals: a plain
  // field write from those callbacks would not notify the scheduler and the view
  // would silently go stale. (zoneless-migration.md, Phase 1.2 / Recipe B & F.)
  readonly isVoting = signal(false);
  readonly volume = signal(1);
  readonly audioPlaying = signal(true);
  readonly showAnimations = signal(true);
  readonly showMetadata = signal(false);
  readonly swipeClass = signal('');
  readonly spinningVote = signal<'good' | 'bad' | null>(null);

  /** Persisted dismissal of the zero-votes first-vote hint. Initialised
   *  to ``true`` so the hint never flashes before settings load resolves;
   *  loadSettings() flips it to ``false`` only when the server confirms
   *  the user has never dismissed it. */
  private readonly labelHintDismissed = signal(true);

  /** Transient text shown after Cmd/Ctrl-Z; auto-cleared after a short delay. */
  readonly undoToastText = signal<string | null>(null);
  private undoToastTimer: ReturnType<typeof setTimeout> | null = null;

  // Bad-vote-with-box state: drawing a box is real work, so a stray ← shouldn't
  // throw it away. The first ← arms a sticky discard-confirm state (no timeout);
  // the second ← throws the box away and votes no. Esc, a mousedown on the box,
  // a Shift-drag-redraw, or navigating to another item all clear the armed
  // state without voting and without discarding the box.
  // Public so the template can bind it into the image viewer.
  currentRegionBox: RegionBox | null = null;
  readonly pendingBadConfirm = signal(false);

  /** Which payload the viewer shows for an item a MediaCleaner rewrote at load
   *  time (``docs/plans/media-cleaners.md``): the canonical cleaned bytes, or
   *  the pre-clean snapshot.  Always back to Clean on a media change - the
   *  cleaned payload is the one the detector actually scored, so it is what the
   *  user should see by default on every new item. */
  readonly payloadVariant = signal<PayloadVariant>('');

  /** Embedder capability listings, loaded once in init(). Used to decide
   *  whether the active dataset's embedder emits a best-match region overlay
   *  (patch-region or structural), which gates the Highlight toggle, and
   *  whether it is structural, which tunes the marquee copy. */
  private embedderInfos: EmbedderInfo[] = [];

  private spinTimer: ReturnType<typeof setTimeout> | null = null;

  private _pausedByVisibility = false;

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      this.volume.set(settings.volume ?? 1);
      this.audioPlaying.set(settings.audio_playing !== false);
      this.showAnimations.set(settings.show_animations !== 'hide');
      this.showMetadata.set(settings.show_metadata === true);
      this.labelHintDismissed.set(settings.label_hint_dismissed === true);
    });

    // Any vote in any pane (center buttons, keyboard, hover-vote) retires the
    // first-vote hint for this user. VoteStateService is signal-backed, so an
    // effect tracking the vote sets covers every channel without each call site
    // knowing about the hint. The dismiss logic runs `untracked` because it both
    // reads and writes `labelHintDismissed` — tracking that read would loop the
    // effect (zoneless-migration.md, Phase 2.5).
    effect(() => {
      this.voteState.goodVotes;
      this.voteState.badVotes;
      untracked(() => this.maybeDismissLabelHint());
    });

    // Reset per-item transient state on every media change (including same-id
    // reference changes from metadata-cache hydration, matching the old
    // ngOnChanges behavior).
    effect(() => {
      this.media();
      this.swipeClass.set('');
      // Navigating to another item clears the armed bad-vote-confirm state.
      // ImageViewer also clears its own regionBox on media change and will emit null;
      // resetting eagerly here keeps state coherent across the swap.
      this.pendingBadConfirm.set(false);
      this.currentRegionBox = null;
      // Back to the canonical payload: "show me the original" is a per-item
      // question, and leaving it latched would silently show the next item's
      // pre-clean bytes (or nothing, for an item with no snapshot).
      this.payloadVariant.set('');
    });
  }

  ngOnDestroy(): void {
    this.stopPlayback();
    this.keyboard.stop();
    if (this.spinTimer) clearTimeout(this.spinTimer);
    if (this.undoToastTimer) clearTimeout(this.undoToastTimer);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
  }

  /** Stop all media playback (used on navigation away). */
  stopPlayback(): void {
    const audio = this.audioPlayer()?.audioRef()?.nativeElement;
    if (audio) {
      audio.pause();
    }
    const video = this.videoPlayer()?.videoRef()?.nativeElement;
    if (video) {
      video.pause();
    }
  }

  /** Initialize: load settings, start keyboard listener, listen for tab visibility. */
  init(): void {
    this.loadSettings();
    this.keyboard.start();
    this.datasetsListingsApi
      .getEmbedders()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (embedders) => (this.embedderInfos = embedders),
      });
    document.addEventListener('visibilitychange', this.onVisibilityChange);
    this.keyboard.action$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((action) => {
      switch (action.type) {
        case 'vote':
          if (this.media() && action.direction && !this.disabled()) {
            this.castVote(action.direction);
          }
          break;
        case 'volume':
          this.adjustVolume(action.volumeDelta ?? 0);
          break;
        case 'playback':
          this.togglePlayback();
          break;
        case 'zoom': {
          const imageViewer = this.imageViewer();
          if (imageViewer && action.zoomDirection) {
            if (action.zoomDirection === 'in') imageViewer.zoomIn();
            else imageViewer.zoomOut();
          }
          break;
        }
        case 'rotate': {
          const imageViewer = this.imageViewer();
          if (imageViewer && action.rotateDirection) {
            if (action.rotateDirection === 'left') imageViewer.rotateLeft();
            else imageViewer.rotateRight();
          }
          break;
        }
        case 'undo':
          if (!this.disabled() && !this.isVoting()) this.voteState.undo();
          break;
        case 'redo':
          if (!this.disabled() && !this.isVoting()) this.voteState.redo();
          break;
      }
    });
    this.voteState.toast$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((t) => this.showUndoToast(t.action, t.mediaName));
  }

  /** Persist the first-vote hint as dismissed once the first vote lands. */
  private maybeDismissLabelHint(): void {
    if (this.labelHintDismissed()) return;
    if (this.voteState.goodVotes.size === 0 && this.voteState.badVotes.size === 0) return;
    this.labelHintDismissed.set(true);
    this.settingsState.update({ label_hint_dismissed: true }).subscribe();
  }

  private showUndoToast(action: 'undo' | 'redo', mediaName: string): void {
    const verb = action === 'undo' ? 'Undid vote on' : 'Redid vote on';
    this.undoToastText.set(`${verb} ${mediaName}`);
    if (this.undoToastTimer) clearTimeout(this.undoToastTimer);
    this.undoToastTimer = setTimeout(() => {
      this.undoToastText.set(null);
      this.undoToastTimer = null;
    }, 2000);
  }

  /** Whether to offer the Clean/Original toggle: only items some cleaner
   *  actually rewrote carry a pre-clean snapshot to switch to. */
  get hasOriginalPayload(): boolean {
    return this.media()?.has_original === true;
  }

  setPayloadVariant(variant: PayloadVariant): void {
    this.payloadVariant.set(variant);
  }

  get mediaType(): string {
    return this.media()?.media_type || 'audio';
  }

  /** The embedder names bound to the focused media (the v3 trio).  Prefers the
   *  full ``embeddings`` key set the backend now ships under ``embedders``;
   *  falls back to the singular primary for older payloads. */
  private boundEmbedderNames(): string[] {
    const m = this.media();
    if (!m) return [];
    if (m.embedders && m.embedders.length > 0) return m.embedders;
    return m.embedder ? [m.embedder] : [];
  }

  /** Capability listing for an embedder name, or ``undefined`` when unknown. */
  private infoFor(name: string): EmbedderInfo | undefined {
    if (!name || this.embedderInfos.length === 0) return undefined;
    return this.embedderInfos.find((e) => e.name === name);
  }

  /** Whether ANY embedder bound to the active dataset emits a best-match region
   *  overlay.  Gates the Highlight toggle: patch-region embedders (DINOv2/v3,
   *  EUPE) emit the argmax-patch region and structural embedders (SIFT/VLAD)
   *  emit the RANSAC inlier box — both ride the same ``best_region`` overlay
   *  machinery, so either capability shows the toggle.  Checking the whole trio
   *  (not just the primary) means a dataset whose primary is a text embedder but
   *  also binds a patch/structural one still offers the overlay.  Defaults to
   *  false when the embedders are unknown so a dead toggle never appears. */
  get regionOverlayCapable(): boolean {
    return this.boundEmbedderNames().some((n) => {
      const info = this.infoFor(n);
      return info?.supports_patch_regions === true || info?.supports_geometric_verification === true;
    });
  }

  /** Whether the active dataset binds a structural embedder (instance matching).
   *  Drives the marquee copy: for structural datasets the region box is
   *  constitutive — it *defines* the template to match — so the affordance
   *  nudges "box the pattern you want to match" rather than the generic
   *  salient-area region hint. */
  get structuralDataset(): boolean {
    return this.boundEmbedderNames().some((n) => this.infoFor(n)?.supports_geometric_verification === true);
  }

  /** Marquee button tooltip. Structural datasets nudge toward boxing the
   *  pattern to match (the box defines the template); other datasets get the
   *  generic region-draw copy. */
  get marqueeTitle(): string {
    return this.structuralDataset
      ? 'Marquee: drag to box the pattern you want to match (Shift+drag also works)'
      : 'Marquee: drag to draw a region (Shift+drag also works)';
  }

  /** Accessible label for the marquee toggle, mirroring `marqueeTitle`. */
  get marqueeAriaLabel(): string {
    return this.structuralDataset ? 'Marquee: box the pattern to match' : 'Marquee: draw region';
  }

  /** The focused media's best-match region from the most recent sort/train,
   *  looked up by id from the in-memory sort results. On a patch dataset that is
   *  the winning row of the backend's max-pool: a **single grid cell** of the
   *  patch grid (~7% of the image's side on a DINOv3 14x14), or the whole image
   *  when the image-level row won. Structural datasets send the RANSAC inlier
   *  box instead. ``null`` when the media wasn't scored or carries no region.
   *  Passed to the image viewer, which draws it only while Highlight is toggled
   *  on - and which suppresses the whole-image case, so a winning image-level
   *  row draws no frame. */
  get highlightBox(): RegionBox | null {
    const media = this.media();
    if (!media) return null;
    const id = media.id;
    const box = this.sortState.sortOrder?.find((s) => s.id === id)?.bestRegion;
    if (!box || box.length !== 4) return null;
    return [box[0], box[1], box[2], box[3]];
  }

  get isGood(): boolean {
    const media = this.media();
    return media ? this.voteState.effectiveGood(media.id) : false;
  }

  get isBad(): boolean {
    const media = this.media();
    return media ? this.voteState.effectiveBad(media.id) : false;
  }

  /** True when the labeling session is fresh (no votes yet across either
   *  polarity) and the user has not previously dismissed the hint. The
   *  hint dismisses on the first vote in this session and persists. */
  get showFirstVoteHint(): boolean {
    if (this.labelHintDismissed()) return false;
    return this.voteState.goodVotes.size === 0 && this.voteState.badVotes.size === 0;
  }

  get customMetadata(): Record<string, unknown> {
    const media = this.media();
    if (!media) return {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (media as any)['custom_metadata'] as Record<string, unknown> || {};
  }

  /** Human-readable label for an item used in undo toasts. */
  private mediaDisplayName(media: Media): string {
    return media.filename || media.origin_name || `#${media.id}`;
  }

  /** Template-bound metadata cell; see {@link formatMetadataValue}. */
  formatMetadataValue(label: string, value: unknown): string {
    return formatMetadataValueUtil(label, value);
  }

  toggleMetadata(): void {
    this.showMetadata.set(!this.showMetadata());
    this.settingsState.update({ show_metadata: this.showMetadata() }).subscribe();
  }

  castVote(vote: 'good' | 'bad'): void {
    const media = this.media();
    if (!media || this.isVoting()) return;

    // Region annotations only attach to yes-votes (salient-area semantics).
    // A no-vote with a box drawn arms a sticky discard-confirm state; the first
    // ← shake-pulses the box and surfaces a hint; only a second ← while armed
    // throws the box away and votes no. The state has no timeout: a time-based
    // modal would expire silently and surprise the user. Esc, mouse-on-box, a
    // fresh Shift-drag, or item navigation clear armed without voting.
    if (vote === 'bad' && this.currentRegionBox && !this.pendingBadConfirm()) {
      this.pendingBadConfirm.set(true);
      this.imageViewer()?.pulseRegionBox();
      return;
    }

    const regionBox = vote === 'good' ? this.currentRegionBox : null;
    // The vote belongs to the item that was selected when the key was pressed.
    // Selection can move while the request is in flight (or during the 180ms
    // swipe animation below), so pin the id here rather than re-reading the
    // input later; otherwise the emit carries a new item's id with this vote.
    const votedId = media.id;
    this.pendingBadConfirm.set(false);
    this.isVoting.set(true);

    this.voteState
      .submitToggleVoteAndRecord(media.id, vote, this.mediaDisplayName(media), regionBox)
      .subscribe({
        next: () => {
          const animate = this.showAnimations() && !!this.media() && !prefersReducedMotion();
          if (animate) {
            this.swipeClass.set(vote === 'good' ? 'swipe-right' : 'swipe-left');
            this.spinningVote.set(vote);
            if (this.spinTimer) clearTimeout(this.spinTimer);
            this.spinTimer = setTimeout(() => this.spinningVote.set(null), 300);
            setTimeout(() => {
              this.mediaVoted.emit({ id: votedId, vote });
              this.isVoting.set(false);
            }, 180);
          } else {
            this.mediaVoted.emit({ id: votedId, vote });
            this.isVoting.set(false);
          }
        },
        error: () => {
          this.isVoting.set(false);
        },
      });
  }

  /** True while a Shift-drag (or the sticky Marquee toggle) would draw a region.
   *  Drives the panel-wide crosshair, which is the only affordance saying the
   *  gesture reaches past the image into the space below it. */
  get regionDrawActive(): boolean {
    return this.mediaType === 'image' && (this.imageViewer()?.regionDrawActive ?? false);
  }

  /** Off-canvas region-draw start: a Shift-drag begun on the toolbar strip, the
   *  vote row, the metadata tray or the gaps between them anchors a box at the
   *  nearest point on the image, exactly as one begun in the letterbox column
   *  beside it always has. The viewer decides whether to claim the event and
   *  ignores anything begun on an interactive control, so every click down here
   *  keeps its current behaviour. */
  onPanelMouseDown(event: MouseEvent): void {
    if (this.mediaType !== 'image') return;
    this.imageViewer()?.tryStartOffCanvasDraw(event);
  }

  onRegionBoxChange(box: RegionBox | null): void {
    this.currentRegionBox = box;
    // Clearing the box also clears any pending bad-vote confirmation;
    // there's nothing left to confirm against.
    if (!box) this.pendingBadConfirm.set(false);
  }

  /** Esc-while-armed or mouse interaction with the box: cancel armed, keep the box. */
  onArmedConfirmCanceled(): void {
    this.pendingBadConfirm.set(false);
  }

  private loadSettings(): void {
    this.settingsState.load();
  }

  private adjustVolume(delta: number): void {
    this.volume.set(Math.max(0, Math.min(1, this.volume() + delta)));
    this.audioPlayer()?.adjustVolume(delta);
    this.videoPlayer()?.adjustVolume(delta);
    this.settingsState.update({ volume: this.volume() }).subscribe();
  }

  onPlayingChanged(playing: boolean): void {
    if (this._pausedByVisibility) return;
    this.audioPlaying.set(playing);
    this.settingsState.update({ audio_playing: this.audioPlaying() }).subscribe();
  }

  private onVisibilityChange = (): void => {
    if (document.hidden) {
      if (this.audioPlaying()) {
        this._pausedByVisibility = true;
        this.stopPlayback();
      }
    } else {
      if (this._pausedByVisibility) {
        this._pausedByVisibility = false;
        this.resumePlayback();
      }
    }
  };

  private resumePlayback(): void {
    const audio = this.audioPlayer()?.audioRef()?.nativeElement;
    if (audio) audio.play().catch(() => {});
    const video = this.videoPlayer()?.videoRef()?.nativeElement;
    if (video) video.play().catch(() => {});
  }

  private togglePlayback(): void {
    this.audioPlaying.set(!this.audioPlaying());
    this.audioPlayer()?.togglePlayback();
    this.videoPlayer()?.togglePlayback();
    this.settingsState.update({ audio_playing: this.audioPlaying() }).subscribe();
  }
}

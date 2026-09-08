import { ChangeDetectionStrategy, Component, DestroyRef, effect, ElementRef, inject, input, OnInit, output, signal, viewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { Subject, takeUntil, timer, switchMap, filter, take } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { JobProgressComponent } from '../../job-progress/job-progress.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ChartsService } from '../../../services/charts.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { ProgressEventsService } from '../../../services/progress-events.service';
import {
  ErrorCostPoint,
  StabilityPoint,
  DiversityPoint,
} from '../../../models/api.models';
import type { EvalTrainAndScoreResponse } from '../../../generated/api-client/models/eval-train-and-score-response';

export type ProgressMetric = 'smart' | 'stable' | 'diverse';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-progress-modal',
  standalone: true,
  imports: [ModalComponent, JobProgressComponent],
  templateUrl: './progress-modal.component.html',
  styleUrl: './progress-modal.component.scss',
})
export class ProgressModalComponent implements OnInit {
  private sortingApi = inject(SortingApiService);
  private chartsService = inject(ChartsService);
  private settingsState = inject(SettingsStateService);
  private progressEvents = inject(ProgressEventsService);

  readonly metric = input<ProgressMetric>('smart');
  readonly closed = output<void>();

  // Optional query: the canvas only renders in the results `@else` branch.
  readonly chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('chartCanvas');

  // Signals, not plain fields: the app is zoneless and this component is
  // OnPush, and every one of these is written from an HTTP/SSE subscribe
  // callback. This component emits no output on the data path either, so
  // plain fields froze the modal on "Loading indicator history…" forever.
  readonly analyzing = signal(true);
  readonly analysisProgress = signal(0);
  readonly chartData = signal<ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[]>([]);
  readonly emptyHistory = signal(false);
  /** True once we've fallen back to the async train-and-score job, which
   *  swaps the brief "loading" line for a real progress bar + Cancel. */
  readonly runningJob = signal(false);
  /** Job id of the in-flight eval train-and-score run. Set once the
   *  backend hands back a job envelope; consumed by ``onCancel``. */
  private currentJobId: string | null = null;

  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    // The canvas lives in the results `@else` branch, so it only exists one
    // render pass after `chartData` lands. Keying the draw on the viewChild
    // signal *and* the data means the chart paints exactly when the canvas
    // materialises, rather than racing it on a fixed timeout.
    effect(() => {
      const canvas = this.chartCanvas()?.nativeElement;
      const data = this.chartData();
      if (!canvas || data.length === 0) return;
      this.renderChart(canvas, data);
    });
  }

  get title(): string {
    switch (this.metric()) {
      case 'smart':
        return 'Smart: Detector Accuracy Over Time';
      case 'stable':
        return 'Stable: How Often The Detector Changes Its Mind';
      case 'diverse':
        return 'Diverse: How Much Of Your Collection Your Votes Cover';
    }
  }

  /**
   * What to say when the series comes back empty.
   *
   * Smart and Stable plot one point per detector the app actually trained, so
   * they stay empty until a Learn sort has run — labelling alone adds nothing
   * to them. Telling the user to label more would send them somewhere that
   * never fills the chart. Diverse measures the votes themselves, so there the
   * original advice is the right one.
   */
  get emptyMessage(): string {
    return this.metric() === 'diverse'
      ? 'No coverage history available yet. Label more items to build up history.'
      : 'No history yet. Each point is a detector this session trained, so sort by Learn ' +
        '(or let Autopilot do it) and a point appears for every training run.';
  }

  ngOnInit(): void {
    // Always try the cached read first: it never advances the per-step cache,
    // so it returns immediately whether or not the cache is warm. When the
    // background `/api/labeling-status` worker has kept up (the common case)
    // the plot paints instantly; otherwise we fall back to the async job,
    // which does the retraining off the request thread with a progress bar.
    this.loadCachedHistory();
  }

  private loadCachedHistory(): void {
    this.analyzing.set(true);
    this.sortingApi
      .getIndicatorScoreHistory(this.metric())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          // `complete: false` means the per-step cache is behind the label
          // history. The endpoint deliberately does not advance it (that build
          // is what used to hang this modal for tens of seconds), so hand off
          // to the background job instead of rendering a truncated series.
          if (!res.complete) {
            this.runAnalysis();
            return;
          }
          this.analyzing.set(false);
          this.chartData.set(
            (res.history || []) as ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[],
          );
          this.emptyHistory.set(this.chartData().length === 0);
        },
        // A failed cached read is not fatal: the job path recomputes the
        // series from scratch, so fall back rather than showing "no history".
        error: () => this.runAnalysis(),
      });
  }

  private runAnalysis(): void {
    this.analyzing.set(true);
    this.runningJob.set(true);
    this.analysisProgress.set(0);

    // Progress comes from the `eval` SSE channel on /api/events. Use a
    // dedicated notifier to stop watching once the bar reaches 100%: the
    // backend emits the `idle/Done` eval frame *inside* `_run`, before the
    // job flips to `done`, so this fires while the result poller is still
    // polling. It must NOT tear down the poller — hence a subject scoped to
    // this one stream, rather than anything component-wide, which would kill
    // the poller too and leave `analyzing` hung forever.
    const stopWatchingProgress$ = new Subject<void>();
    this.progressEvents.votingIterations$
      .pipe(takeUntilDestroyed(this.destroyRef), takeUntil(stopWatchingProgress$))
      .subscribe({
        next: (res) => {
          if (res.total > 0) {
            this.analysisProgress.set(Math.round((res.progress / res.total) * 100));
          }
          if (res.done) {
            stopWatchingProgress$.next();
            stopWatchingProgress$.complete();
          }
        },
      });

    // Request train-and-score; the new endpoint returns a job envelope.
    // Teardown here guards the case where the modal is dismissed while the
    // POST is in flight: without it, a late `next` would arm `pollEvalJob()`
    // on a dead component, leaking a poller. `takeUntilDestroyed` closes that
    // from both ends — it stops this subscription at destroy, and a
    // subscription armed *after* destroy completes immediately (its
    // `destroyRef.destroyed` branch). A hand-rolled `takeUntil(destroy$)`
    // could not do the second half: RxJS `takeUntil` never fires on a
    // pre-completed notifier, so the poller would have run forever.
    this.sortingApi
      .trainAndScore(this.metric())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          if (res.status === 'done') {
            this.applyEvalResult(res);
          } else if (res.status === 'running') {
            this.currentJobId = res.job_id;
            this.pollEvalJob(res.job_id);
          } else {
            this.analyzing.set(false);
          }
        },
        error: () => {
          this.analyzing.set(false);
        },
      });
  }

  private pollEvalJob(jobId: string): void {
    timer(200, 500)
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        switchMap(() => this.sortingApi.getEvalTrainAndScoreResult(jobId)),
        filter((res) => res.status !== 'running'),
        take(1),
      )
      .subscribe({
        next: (res) => {
          this.currentJobId = null;
          if (res.status === 'done') {
            this.applyEvalResult(res);
          } else {
            this.analyzing.set(false);
          }
        },
        error: () => {
          this.currentJobId = null;
          this.analyzing.set(false);
        },
      });
  }

  /** Cancel the in-flight eval job (if any) and close the modal.
   *
   *  This is the single dismissal path for the modal: the in-body Cancel
   *  button, and Escape / the X / a backdrop click (routed here from the
   *  inner `vt-modal`'s `(closed)`) all land here, so every way of leaving
   *  the modal stops the running eval job rather than orphaning it. Safe on
   *  the cached-history path, where `currentJobId` is null and no cancel
   *  request is sent. */
  onCancel(): void {
    const jobId = this.currentJobId;
    this.currentJobId = null;
    if (jobId) {
      this.sortingApi.cancelEvalTrainAndScore(jobId).pipe(takeUntilDestroyed(this.destroyRef)).subscribe();
    }
    this.analyzing.set(false);
    this.close();
  }

  private applyEvalResult(res: EvalTrainAndScoreResponse): void {
    this.analyzing.set(false);
    this.runningJob.set(false);
    if (this.metric() === 'smart') {
      this.chartData.set((res.error_cost || []) as ErrorCostPoint[]);
    } else if (this.metric() === 'stable') {
      this.chartData.set((res.stability || []) as StabilityPoint[]);
    } else {
      this.chartData.set((res.diversity || []) as DiversityPoint[]);
    }
    // Same empty-state handling as the cached path: a job that legitimately
    // produces no points (too little history) shows the explanatory message
    // rather than an empty set of axes.
    this.emptyHistory.set(this.chartData().length === 0);
  }

  private renderChart(
    canvas: HTMLCanvasElement,
    data: ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[],
  ): void {
    switch (this.metric()) {
      case 'smart':
        this.chartsService.renderErrorCostChart(canvas, data as ErrorCostPoint[]);
        break;
      case 'stable':
        this.chartsService.renderStabilityChart(canvas, data as StabilityPoint[]);
        break;
      case 'diverse':
        this.chartsService.renderDiversityChart(
          canvas,
          data as DiversityPoint[],
          this.settingsState.settingsSignal()?.autopilot_goal_diversity ?? 40,
        );
        break;
    }
  }

  close(): void {
    this.closed.emit();
  }
}

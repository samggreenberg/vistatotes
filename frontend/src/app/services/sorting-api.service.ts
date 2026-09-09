import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { CoverageAtlasNextResponse } from '../generated/api-client/models/coverage-atlas-next-response';
import type { EvalTrainAndScoreCancelResponse } from '../generated/api-client/models/eval-train-and-score-cancel-response';
import type { EvalTrainAndScoreResponse } from '../generated/api-client/models/eval-train-and-score-response';
import type { ExampleSortResponse } from '../generated/api-client/models/example-sort-response';
import type { FillFromSortRequest } from '../generated/api-client/models/fill-from-sort-request';
import type { FillFromSortResponse } from '../generated/api-client/models/fill-from-sort-response';
import type { InclusionResponse } from '../generated/api-client/models/inclusion-response';
import type { IndicatorScoreHistoryResponse } from '../generated/api-client/models/indicator-score-history-response';
import type { LabelFileSortResponse } from '../generated/api-client/models/label-file-sort-response';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';
import type { LabelsExportResponse } from '../generated/api-client/models/labels-export-response';
import type { LabelsImportRequest } from '../generated/api-client/models/labels-import-request';
import type { LabelsImportResponse } from '../generated/api-client/models/labels-import-response';
import type { LearnedSortCancelResponse } from '../generated/api-client/models/learned-sort-cancel-response';
import type { LearnedSortResponse } from '../generated/api-client/models/learned-sort-response';
import type { OkResponse } from '../generated/api-client/models/ok-response';
import type { ServerMediaListResponse } from '../generated/api-client/models/server-media-list-response';
import type { ServerMediaUploadResponse } from '../generated/api-client/models/server-media-upload-response';
import type { SortResponse } from '../generated/api-client/models/sort-response';
import type { TextsortSuggestionsResponse } from '../generated/api-client/models/textsort-suggestions-response';
import type { VotesResponse } from '../generated/api-client/models/votes-response';
import { coverageAtlasNextGet } from '../generated/api-client/fn/sorting/coverage-atlas-next-get';
import { getInclusionRoute } from '../generated/api-client/fn/sorting/get-inclusion-route';
import { setInclusionRoute } from '../generated/api-client/fn/sorting/set-inclusion-route';
import { cancelLearnedSort } from '../generated/api-client/fn/sorting/cancel-learned-sort';
import { learnedSort } from '../generated/api-client/fn/sorting/learned-sort';
import { learnedSortResult } from '../generated/api-client/fn/sorting/learned-sort-result';
import { sortClips } from '../generated/api-client/fn/sorting/sort-clips';
import { sortPage } from '../generated/api-client/fn/sorting/sort-page';
import type { SortPageResponse } from '../generated/api-client/models/sort-page-response';
import { getTextsortSuggestionsRoute } from '../generated/api-client/fn/sorting/get-textsort-suggestions-route';
import { addTextsortSuggestionRoute } from '../generated/api-client/fn/sorting/add-textsort-suggestion-route';
import { clearVotesRoute } from '../generated/api-client/fn/sorting/clear-votes-route';
import { getVotes } from '../generated/api-client/fn/sorting/get-votes';
import { exportLabels } from '../generated/api-client/fn/labels/export-labels';
import { fillLabelsFromSort } from '../generated/api-client/fn/labels/fill-labels-from-sort';
import { importLabels } from '../generated/api-client/fn/labels/import-labels';
import { cancelEvalTrainAndScore } from '../generated/api-client/fn/eval/cancel-eval-train-and-score';
import { evalTrainAndScore } from '../generated/api-client/fn/eval/eval-train-and-score';
import { evalTrainAndScoreResult } from '../generated/api-client/fn/eval/eval-train-and-score-result';
import { indicatorScoreHistory } from '../generated/api-client/fn/eval/indicator-score-history';
import { labelingStatusIndicator } from '../generated/api-client/fn/eval/labeling-status-indicator';
import { exampleSortById } from '../generated/api-client/fn/media-server/example-sort-by-id';
import { exampleSortOrigin } from '../generated/api-client/fn/media-server/example-sort-origin';
import { exampleSortServer } from '../generated/api-client/fn/media-server/example-sort-server';
import { serverMediaFileFromMediaId } from '../generated/api-client/fn/media-server/server-media-file-from-media-id';
import { listServerMediaFiles } from '../generated/api-client/fn/media-server/list-server-media-files';

/**
 * Label partition used to open the export modal. The first six map directly to
 * the ``/api/labels/export`` ``label_filter`` query (server-side partitions or
 * client-side category slices). ``unverified_good`` is a UI-only combination —
 * the export modal fetches the server ``unverified`` partition and slices it to
 * the ``good`` category (the above-threshold work queue); it is never sent to
 * the backend as a ``label_filter`` value.
 */
export type LabelFilter =
  | 'good'
  | 'bad'
  | 'both'
  | 'corrections'
  | 'unverified'
  | 'verified'
  | 'unverified_good';

/**
 * The subset of {@link LabelFilter} the backend ``/api/labels/export``
 * ``label_filter`` query accepts. Excludes the UI-only ``unverified_good``,
 * which the export modal resolves to a server ``unverified`` fetch plus a
 * client-side ``good`` slice before any request is made.
 */
export type ServerLabelFilter = Exclude<LabelFilter, 'unverified_good'>;

@Injectable({ providedIn: 'root' })
export class SortingApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  sort(params: { text: string }): Observable<SortResponse> {
    return sortClips(this.http, this.config.rootUrl, { body: params }).pipe(map((r) => r.body));
  }

  /**
   * Fetch a deeper window of a windowed ranking (scalability.md S3/S17/S19).
   * `token` is the `sort_token` the sort response returned; a 404 means the
   * ranking was superseded or evicted and the caller should re-sort.
   */
  getSortPage(token: string, offset: number, limit: number): Observable<SortPageResponse> {
    return sortPage(this.http, this.config.rootUrl, { token, offset, limit }).pipe(map((r) => r.body));
  }

  /** Kick off a learned-sort training job.  The response will be ``done``
   *  immediately when the cached signature matches; otherwise the caller
   *  must poll {@link getLearnedSortResult} with the returned ``job_id``. */
  learnedSort(): Observable<LearnedSortResponse> {
    return learnedSort(this.http, this.config.rootUrl, { body: {} }).pipe(map((r) => r.body));
  }

  /** Poll for a learned-sort job's completion. */
  getLearnedSortResult(jobId: string): Observable<LearnedSortResponse> {
    return learnedSortResult(this.http, this.config.rootUrl, { job_id: jobId }).pipe(
      map((r) => r.body),
    );
  }

  /** Cancel an in-flight learned-sort job. */
  cancelLearnedSort(jobId: string): Observable<LearnedSortCancelResponse> {
    return cancelLearnedSort(this.http, this.config.rootUrl, { job_id: jobId }).pipe(
      map((r) => r.body),
    );
  }

  getVotes(): Observable<VotesResponse> {
    return getVotes(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  clearVotes(): Observable<OkResponse> {
    return clearVotesRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getInclusion(): Observable<InclusionResponse> {
    return getInclusionRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  setInclusion(value: number): Observable<InclusionResponse> {
    return setInclusionRoute(this.http, this.config.rootUrl, { body: { inclusion: value } }).pipe(
      map((r) => r.body),
    );
  }

  /** Fetch labels for export.
   *
   *  ``detectorName`` switches the read from the active pair's live labels to
   *  that detector's *persisted* labelset, which is what a caller naming a
   *  detector in a list (the Dashboard row action) means — see the route's
   *  ``detector_name`` param. */
  exportLabels(
    goodsOnly?: boolean,
    options?: { enrich?: boolean; labelFilter?: ServerLabelFilter; detectorName?: string },
  ): Observable<LabelsExportResponse> {
    return exportLabels(this.http, this.config.rootUrl, {
      goods_only: goodsOnly || undefined,
      enrich: options?.enrich || undefined,
      label_filter: options?.labelFilter || undefined,
      detector_name: options?.detectorName || undefined,
    }).pipe(map((r) => r.body));
  }

  importLabels(data: LabelsImportRequest): Observable<LabelsImportResponse> {
    return importLabels(this.http, this.config.rootUrl, { body: data }).pipe(map((r) => r.body));
  }

  fillFromSort(request: FillFromSortRequest): Observable<FillFromSortResponse> {
    return fillLabelsFromSort(this.http, this.config.rootUrl, { body: request }).pipe(
      map((r) => r.body),
    );
  }

  /** Multipart upload; stays on plain HttpClient because ng-openapi-gen
   *  doesn't model multipart bodies (the generated function's ``$Params``
   *  has no ``body`` field). */
  exampleSort(file: File, cropParams?: Record<string, unknown>): Observable<SortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (cropParams) {
      formData.append('crop_params', JSON.stringify(cropParams));
    }
    return this.http.post<SortResponse>('/api/example-sort', formData);
  }

  getServerMediaFiles(): Observable<ServerMediaListResponse> {
    return listServerMediaFiles(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Sort by one or more server-side example files. Multiple filenames
   *  rank the haystack against the centroid of the examples' embeddings. */
  exampleSortServer(params: {
    filenames: string[];
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return exampleSortServer(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  exampleSortOrigin(params: {
    origin: Record<string, unknown>;
    key: string;
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return exampleSortOrigin(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Sort the loaded snapshot by similarity to an already-loaded media.
   *  Skips re-embedding when ``crop_params`` is absent; the in-memory
   *  embedding is reused directly. */
  exampleSortById(params: {
    media_id: number;
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return exampleSortById(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Save a loaded media's bytes to example_media/ so the new-detector
   *  flow can reference it via ``media_example``. */
  saveServerMediaFromMediaId(params: {
    media_id: number;
    crop_params?: Record<string, unknown>;
  }): Observable<ServerMediaUploadResponse> {
    return serverMediaFileFromMediaId(this.http, this.config.rootUrl, {
      body: params,
    }).pipe(map((r) => r.body));
  }

  /** Multipart upload; see {@link exampleSort}. */
  uploadServerMediaFile(
    file: File,
    options?: { mediaType?: string; cropParams?: Record<string, unknown> },
  ): Observable<ServerMediaUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.cropParams) {
      formData.append('crop_params', JSON.stringify(options.cropParams));
    }
    // Appended independently of cropParams: the backend only *requires*
    // media_type alongside crop_params, but a caller passing mediaType
    // alone shouldn't have it silently dropped.
    if (options?.mediaType) {
      formData.append('media_type', options.mediaType);
    }
    return this.http.post<ServerMediaUploadResponse>('/api/server-media-files/upload', formData);
  }

  /** Multipart upload; see {@link exampleSort}. */
  labelFileSort(file: File): Observable<LabelFileSortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<LabelFileSortResponse>('/api/label-file-sort', formData);
  }

  getTextsortSuggestions(): Observable<TextsortSuggestionsResponse> {
    return getTextsortSuggestionsRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  addTextsortSuggestion(text: string): Observable<OkResponse> {
    return addTextsortSuggestionRoute(this.http, this.config.rootUrl, { body: { text } }).pipe(
      map((r) => r.body),
    );
  }

  /** ``/api/labeling-progress`` reads global state (votes, label history) and
   *  takes no request body; the spec's ``ApiLabelingProgressPost$Params``
   *  reflects that.  Production callers were removed; the method is kept for
   *  parity with the legacy surface. */
  getLabelingProgress(): Observable<unknown> {
    return this.http.post('/api/labeling-progress', {});
  }

  /** The left panel polls this every 2 s for the whole labeling session, and
   *  `adaptivePoll` already absorbs a failed tick (it skips that tick and keeps
   *  polling), so the caller handles the failure itself. Pass
   *  {@link SKIP_ERROR_TOAST} so a tick that lands mid-load does not raise a
   *  global toast: while `ContextSwitchService` is still loading the pair the
   *  poll's `X-Detector-Id` names a detector the backend has not registered
   *  yet, which is a 409 `detector_not_loaded` that resolves itself a moment
   *  later. Issue #3644 is what that toast looked like to a reviewer opening a
   *  new class. */
  getLabelingStatus(): Observable<LabelingStatusResponse> {
    return labelingStatusIndicator(
      this.http,
      this.config.rootUrl,
      undefined,
      new HttpContext().set(SKIP_ERROR_TOAST, true),
    ).pipe(map((r) => r.body));
  }

  getIndicatorScoreHistory(
    metric: 'smart' | 'stable' | 'diverse',
  ): Observable<IndicatorScoreHistoryResponse> {
    return indicatorScoreHistory(this.http, this.config.rootUrl, { metric }).pipe(
      map((r) => r.body),
    );
  }

  /** The POST branch carries an optional ``{scores, threshold}`` body that the
   *  backend reads via ``request.get_json(silent=True)``; the OpenAPI spec
   *  intentionally omits that body so GET and POST share one declaration, so
   *  the POST call stays on plain ``HttpClient``.  The GET branch uses the
   *  generated function. */
  getCoverageAtlasNext(
    scores?: Record<string, number>,
    threshold?: number,
  ): Observable<CoverageAtlasNextResponse> {
    if (scores) {
      return this.http.post<CoverageAtlasNextResponse>('/api/coverage-atlas/next', { scores, threshold });
    }
    return coverageAtlasNextGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Kick off an eval train-and-score job.  Like {@link learnedSort}, the
   *  response will be ``done`` immediately on a cache hit; otherwise poll
   *  {@link getEvalTrainAndScoreResult}. */
  trainAndScore(metric: 'smart' | 'stable' | 'diverse'): Observable<EvalTrainAndScoreResponse> {
    return evalTrainAndScore(this.http, this.config.rootUrl, { body: { metric } }).pipe(
      map((r) => r.body),
    );
  }

  getEvalTrainAndScoreResult(jobId: string): Observable<EvalTrainAndScoreResponse> {
    return evalTrainAndScoreResult(this.http, this.config.rootUrl, { job_id: jobId }).pipe(
      map((r) => r.body),
    );
  }

  /** Cancel an in-flight eval train-and-score job. */
  cancelEvalTrainAndScore(jobId: string): Observable<EvalTrainAndScoreCancelResponse> {
    return cancelEvalTrainAndScore(this.http, this.config.rootUrl, {
      job_id: jobId,
    }).pipe(map((r) => r.body));
  }
}

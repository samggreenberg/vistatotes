/**
 * Capture the deck's UI screenshots against a corpus of real photographs.
 *
 *   node slides/figs/src/shoot-ui-figs.mjs            # every shot
 *   node slides/figs/src/shoot-ui-figs.mjs train-loop # one group
 *
 * Four groups, and the first three are one continuous session rather than
 * three unrelated frames — they are the deck's click-by-click introduction to
 * the tool, and they are shot in the order a user does them:
 *
 *   make-detector  `figs/ui-make-detector[.buildN].webp`  — name the concept
 *   train-loop     `figs/ui-train-loop[.buildN].webp`     — answer, repeatedly
 *   find           `figs/ui-find*.webp`                    — score unseen media
 *   region-voting  `figs/ui-region-voting.webp`           — vote on a region
 *
 * The detector in the first group is the detector the second group trains and
 * the third group runs, created on camera through the same modal a user would
 * use. Nothing is staged through the API that the slide claims was done by
 * hand: `train-loop` votes by clicking Good and Bad, and which button it
 * clicks is decided by the filename of whatever autopilot chose to serve — so
 * the piles that accumulate in the right-hand panel are a real session's, and
 * the ranking `find` then shows is a real trained head's.
 *
 * These do not reuse the light-theme frames of the same-named shots in
 * `docs/user/screenshots.manifest.ts`, and that is deliberate. The docs shots are deliberately taken against the
 * synthetic fixture — the user guide talks the reader through `syn-imgs`, and
 * flat coloured shapes make a drawn region box unambiguous — but on a slide the
 * same frame is the audience's *first* sight of the tool, and what it shows
 * them is somebody voting on procedurally generated triangles. Nobody has that
 * problem. The screenshots have to look like the job.
 *
 * So this harness keeps the docs fixtures untouched and builds its own corpus
 * out of COCO val2017: a few hundred real photographs filed by subject, with
 * `book` — the deck's running example — as a real concept among real
 * near-misses (a laptop, a monitor, a keyboard: rectangular, printed, shelved).
 * `coco_fixture.py` downloads and materialises it. The detector is trained on
 * books, by voting, exactly as a user would — the ranking in the captured frame
 * is a real ranking from a real trained head.
 *
 * Like `scripts/screenshots/refresh.sh`, this drives a SINGLE running app
 * rather than booting its own: the box is RAM-tight and two instances would
 * load the image embedder twice. Start one with `python app.py --local` first,
 * or let this script start one.
 *
 * The expensive steps are idempotent — the two corpora are downloaded, filed
 * and embedded only if absent — so a re-run after a UI change is the captures
 * plus one session's worth of clicking. The session itself is not: the intro
 * detector is deleted and re-created every run, because a run that reused last
 * run's votes would be shooting a screen nobody ever sat in front of.
 */
import { launchChromium } from '../../../scripts/screenshots/launch.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = process.env.APP || 'http://localhost:5000';
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const FIGS = resolve(HERE, '..');
const FIXTURE_BUILDER = join(REPO, 'slides', 'figs', 'src', 'coco_fixture.py');

// A screenshot's text renders at (slot width / CSS width) of its authored size,
// so what matters is not how many pixels the PNG has but how wide the browser
// *window* was: the 1440px-wide frames these replaced landed in the 717px
// sidebar at 0.50x, which put the app's 13px chrome at 6px. A narrower window
// with the same slot is the only lever — 1180px gives 0.61x, and the pixel
// count is bought back with deviceScaleFactor so nothing is resampled up.
// The window is also nearly square, because the slot is: a 16:10 frame wastes
// two fifths of a `bg right:56%` box, which is the same as choosing to draw
// the whole thing smaller.
//
// The *height* is the app's own layout knob. At 940 the shot filled the slide
// top to bottom with no margin at all — a projector that overscans clips the
// chrome — and the centre viewer, whose photo is width-bound, spent the
// surplus on empty bands above and below it that pushed the Good/Bad buttons
// into the bottom eighth of the slide (#3301). A shorter window takes that
// surplus out of the app's own layout rather than out of the figure.
//
// How short is bounded by the headline, not by taste. The composed canvas is
// 16:9, so the app's width on the slide is `720·(1−2·SHOT_MARGIN)·1180/height`
// and what is left of 1280 is the column the title lives in. That column has
// to clear `slide_figure.TITLE_NOTCH_PX` — 300px at a 60px inset — so:
//
//     height ≥ 720·1180·(1 − 2·SHOT_MARGIN) / (1280 − 375)
//
// which at SHOT_MARGIN = 0.06 is 826. 830 takes it with 4px to spare.
const VIEWPORT = { width: 1180, height: 830 };
const SCALE = 2;

// White above and below the frame, as a fraction of the shot's own height, so
// the app does not bleed to the slide's edges. Spent out of the same 16:9
// canvas as the title column, which is why the two numbers are chosen
// together.
const SHOT_MARGIN = 0.06;

// The intro sequence's detector: created on camera in `make-detector`, trained
// in `train-loop`, and run over unseen media in `find`. Deleted and rebuilt on
// every run, because the first group's whole subject is a detector that does
// not exist yet — an idempotent "create it only if absent" would shoot the
// modal over a detector already in the list.
const INTRO_DETECTOR = 'Books';
const INTRO_TEXT = 'book';

// Where `train-loop` stops to take a picture, as a running vote count. The
// first five pages advance one vote at a time, because the claim the slide is
// making is that a session is one question repeated — a page that jumps from
// two votes to nine shows a result rather than a loop. The last page is the
// payoff: the same panel some way in.
const TRAIN_STAGES = [0, 1, 2, 3, 4];

// The last page is not a vote count but a condition, because a vote count is
// not something this script gets to decide: autopilot chooses what to serve and
// the corpus decides whether that is a book, so "vote twelve times" can end
// with twelve Good votes and a head that has never seen a negative. Vote until
// both piles are worth showing (and the detector is trainable at all), with a
// hard stop so a pathological ranking cannot loop forever.
const TRAIN_FINAL = { good: 8, bad: 3, maxVotes: 24 };

const REGION_VOTES = {
  good: [
    'book/000000262938.jpg', 'book/000000520077.jpg',
    'book/000000542776.jpg', 'book/000000395701.jpg',
  ],
  bad: { laptop: 2, tv: 1, dog: 1 },
};

// The region shot wants the opposite of a portrait: a photo where the book is
// a *part* of the frame, so that a box drawn round it is visibly a claim about
// where the evidence is rather than a box round the whole picture. Hence one
// named frame with a measured box rather than a preference list.
//
// It used to be a bookcase behind a television, with the box round one shelf.
// That taught the wrong thing twice over: a frame already filled with books
// makes the box look like a crop rather than a claim, and a box round a third
// of fourteen tiny spines is not a region anyone would actually draw (#3296).
// This is one book — a boxed game on a bed, a fifth of the frame — beside a
// camera lens, a phone and a remote that are not books. The box is COCO's own
// `book` annotation on that frame, as a fraction of the displayed image, which
// is why it is tight on the object rather than eyeballed round it.
const HERO_REGION = 'book/000000396729.jpg';
const REGION_BOX = { x0: 0.156, y0: 0.222, x1: 0.910, y1: 0.601 };

const log = (...a) => console.log('[slide-shots]', ...a);

/**
 * Screenshot, pad it out to 16:9, then re-encode as WebP.
 *
 * These two figures are photographs behind UI chrome, which is the one thing
 * PNG is bad at: the same frames weigh 2.7 MB as PNG and 0.4 MB as WebP at a
 * quality no projector will resolve the difference at — and unlike the deck's
 * plots, they are re-shot on every GUI change, so the cost is paid again and
 * again. Marp rasterises through Chromium, which reads WebP natively.
 *
 * Pillow does the encode (a project dependency; `scripts/screenshots/refresh.sh`
 * shells out to it for the same reason) because Playwright writes PNG or JPEG
 * and nothing else.
 */
async function shoot(page, name) {
  const png = await page.screenshot({ type: 'png' });
  // Padded on the left to exactly 16:9 before the encode, and by `SHOT_MARGIN`
  // above and below so the frame does not run to the slide's own edges. These go on
  // `_class: full` slides, which reserve their top-left corner for the
  // headline; a 1.25:1 frame letterboxes into that slot with white bands too
  // narrow to hold it, so the title landed across the app's own chrome. The
  // padding is the `slides/STYLE.md` "pan the frame" repair, and it is free
  // here for the same reason it is free there: the frame was height-bound, so
  // the widened canvas is drawn at the same scale and the app comes out the
  // same size on the slide — it just sits to the right of a real title
  // column instead of under a floating headline (#3246).
  execFileSync(
    'python',
    [
      '-c',
      'import sys;from io import BytesIO;from PIL import Image;'
        + 'shot=Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB");'
        + 'm=round(shot.height*float(sys.argv[2]));'
        + 'h=shot.height+2*m;'
        + 'w=max(shot.width,round(h*16/9));'
        + 'canvas=Image.new("RGB",(w,h),"white");'
        + 'canvas.paste(shot,(w-shot.width,m));'
        + 'canvas.save(sys.argv[1],"WEBP",quality=92,method=6)',
      join(FIGS, `${name}.webp`),
      String(SHOT_MARGIN),
    ],
    { cwd: REPO, input: png, stdio: ['pipe', 'inherit', 'inherit'] }
  );
  log(`wrote figs/${name}.webp`);
}
const only = process.argv.slice(2);
const wanted = (id) => only.length === 0 || only.includes(id);

// ── the corpus ───────────────────────────────────────────────────────────────
// COCO val2017, filed by subject, so "book" is a real concept with real
// near-misses in the pile. Which frames land in which category is decided by
// `coco_fixture.py` — deterministically, from the annotations — so the corpus
// is a pure function of the download and this file does not have to hold a
// second copy of the plan.

function buildCorpus(name) {
  return execFileSync('python', [FIXTURE_BUILDER, name], {
    cwd: REPO,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  }).trim();
}

// ── talking to the app ───────────────────────────────────────────────────────

async function api(path, { method = 'GET', body, dataset, detector } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (dataset) headers['X-Dataset-Id'] = dataset;
  if (detector) headers['X-Detector-Id'] = detector;
  const r = await fetch(APP + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${method} ${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(what, predicate, timeoutMs = 1_800_000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    const hit = await predicate();
    if (hit) return hit;
    await sleep(2000);
  }
  throw new Error(`timed out waiting for ${what}`);
}

const datasets = async () => (await api('/api/datasets/registry')).datasets || [];
const detectors = async () => (await api('/api/detectors/registry')).detectors || [];
const named = (rows, name) => rows.find((r) => r.name === name);

async function ensureDataset(name, embedder) {
  const existing = named(await datasets(), name);
  if (existing) {
    log(`dataset ${name} exists (${existing.num_items} items)`);
    // Registered is not loaded. A fresh import leaves the dataset in memory, so
    // the first run never needed this; a re-run against a restarted app finds
    // it on disk and unloaded, and every call after this one 409s with
    // `dataset_not_loaded`. Idempotent means idempotent across restarts too.
    if (!existing.loaded) {
      await api(`/api/datasets/registry/${existing.id}/load`, { method: 'POST' });
      await waitFor(`dataset ${name} to load`, async () => named(await datasets(), name)?.loaded);
    }
    return existing;
  }
  const path = buildCorpus(name);
  log(`importing ${name} (${embedder}) — embedding takes a while on CPU`);
  await api('/api/dataset/import/server_folder', {
    method: 'POST',
    body: {
      path,
      media_type: 'image',
      recursive: 'true',
      reference_files: 'true',
      dataset_name: name,
      embedder,
    },
  });
  const row = await waitFor(`dataset ${name}`, async () => named(await datasets(), name));
  log(`imported ${name} (${row.num_items} items)`);
  return row;
}

async function ensureDetector(name, dataset) {
  const existing = named(await detectors(), name);
  const row =
    existing ||
    (
      await api('/api/detectors/registry', {
        method: 'POST',
        dataset: dataset.id,
        body: { name, media_type: 'image', text_query: 'a photo of a book', trainable: true },
      })
    ).detector;
  await api('/api/detectors/registry/load', {
    method: 'POST',
    dataset: dataset.id,
    body: { detector_id: row.id },
  });
  await waitFor('detector load', async () => named(await detectors(), name)?.loaded);
  return row;
}

/**
 * Vote the way a user would: Good on books, Bad on the things that keep coming
 * back with them. Enough votes to have a trained head and a plausible pair of
 * piles, few enough to still look like the first two minutes of a session —
 * which is the situation the deck is describing.
 */
async function ensureVotes(dataset, detector, plan) {
  const ids = (await api('/api/medias/ids', { dataset: dataset.id, detector: detector.id })).map(
    (m) => m.id
  );
  const meta = await api('/api/medias/batch', {
    method: 'POST',
    dataset: dataset.id,
    detector: detector.id,
    body: { ids },
  });
  const byCategory = {};
  const byName = {};
  for (const m of meta) {
    const category = m.filename.split('/')[0];
    (byCategory[category] ||= []).push(m.id);
    byName[m.filename] = m.id;
  }
  const pick = (category, n) => (byCategory[category] || []).slice(0, n);
  const good = plan.good.map((name) => {
    const id = byName[name];
    if (!id) throw new Error(`${name} is not in the ${dataset.name} corpus`);
    return id;
  });
  const bad = Object.entries(plan.bad).flatMap(([category, n]) => pick(category, n));

  const vote = async (id, target) => {
    for (let attempt = 0; attempt < 20; attempt++) {
      const r = await fetch(`${APP}/api/medias/${id}/vote`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'X-Dataset-Id': dataset.id,
          'X-Detector-Id': detector.id,
        },
        body: JSON.stringify({ target }),
      });
      if (r.ok) return;
      // 409 is "the detector is still settling"; anything else is a real error.
      if (r.status !== 409) throw new Error(`vote ${id} -> ${r.status}`);
      await sleep(1000);
    }
    throw new Error(`vote ${id} still 409 after retries`);
  };
  for (const id of good) await vote(id, 'good');
  for (const id of bad) await vote(id, 'bad');
  for (const id of ids.filter((i) => !good.includes(i) && !bad.includes(i))) await vote(id, 'none');
  await sleep(3000);
  log(`voted ${good.length} good / ${bad.length} bad`);
  // Remembered so the centre viewer can be given an item nobody has answered
  // yet. A frame showing an already-voted item has its Good button filled in,
  // and the whole point of that panel is that the tool is *asking* (#3246).
  const voted = new Set();
  for (const m of meta) if (good.includes(m.id) || bad.includes(m.id)) voted.add(m.filename);
  return voted;
}

// ── capture ──────────────────────────────────────────────────────────────────

/**
 * Kill transitions and carets so the frame is stable, and hide the toast stack.
 *
 * The toasts are an artefact of the harness rather than of the product: this
 * drives a dev checkout, where `static/` is a build artefact that goes stale
 * the moment anything is committed, so `BuildSkewService` puts a large
 * non-dismissing "this page is running an out-of-date build" banner across the
 * top of every frame. It is doing its job — see the note in `CLAUDE.md` — and
 * it has nothing to do with the application a slide is showing.
 */
const STILL_CSS =
  '*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important}'
  + 'vt-toast-container,.toast-stack{display:none!important}';

async function enterLabelView(page, datasetName, detectorName) {
  await openDashboard(page);
  await selectOnly(page, 'tr[vt-dataset-card]', datasetName);
  await selectOnly(page, 'tr[vt-detector-card]', detectorName);
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 120000 });
  await page.waitForTimeout(3000);
}

async function leftTab(page, name) {
  // The tab strip is hidden while autopilot is collapsed, and panel state
  // persists across runs — so expand first, or the second shot of a run waits
  // for a tab that is not on the page.
  if ((await page.locator('.left-tab').count()) === 0) {
    await page.locator('.collapse-toggle').first().click();
    await page.waitForTimeout(1200);
  }
  await page.locator('.left-tab', { hasText: name }).first().click();
  await page.waitForTimeout(800);
}

/**
 * Hand the session to autopilot and fold its panel away to a rail.
 *
 * This is what the deck should be showing (#3246). Manual mode spends four
 * rows of the left panel on sort mode, selection strategy and inclusion before
 * the corpus grid even starts — every one of them a control the audience is
 * being asked to ignore. Autopilot replaces the lot with a five-step phase
 * list, and collapsing that leaves a rail a centimetre wide: what is left on
 * screen is the item and the votes, which is the whole interaction.
 *
 * Switching tabs starts autopilot (`left-panel.setTab`), which re-sorts — but
 * it keeps whatever item is already selected, so the caller can pick the frame
 * in Manual first and still end up here.
 */
async function collapseIntoAutopilot(page) {
  await leftTab(page, 'Autopilot');
  await page.waitForTimeout(9000);
  await page.locator('.collapse-toggle').first().click();
  await page.waitForTimeout(2500);
}

/**
 * Put an unanswered item in the centre viewer — the frame both shots are about.
 *
 * `voted` is excluded rather than merely deprioritised: an item that already
 * carries a vote renders its Good or Bad button filled, which reads as an
 * answer the tool has given itself instead of a question it is asking.
 */
async function serveItem(page, prefer = [], voted = new Set()) {
  const all = await page.locator('.thumbnail-wrap img').evaluateAll((es) => es.map((e) => e.alt));
  const shown = all.filter((n) => !voted.has(n));
  const target =
    prefer.find((name) => shown.includes(name)) ?? shown.find((n) => n.startsWith('book/'));
  const thumb = target
    ? page.locator(`.thumbnail-wrap:has(img[alt="${target}"])`).first()
    : page.locator('.thumbnail-wrap:visible').first();
  await thumb.click();
  await page.waitForSelector('.btn-good', { timeout: 30000 });
  await page.waitForTimeout(1500);
}

/**
 * Delete the intro detector if a previous run left one behind.
 *
 * The first shot's subject is the dialog you use to make a detector, on a
 * dashboard that does not have one yet; the second and third shots then need
 * *this* detector's votes and nobody else's. Both wants are the same want, and
 * neither survives reuse.
 */
async function resetIntroDetector() {
  for (const row of await detectors()) {
    if (row.name !== INTRO_DETECTOR) continue;
    await api(`/api/detectors/registry/${row.id}`, { method: 'DELETE' });
    log(`removed the previous ${INTRO_DETECTOR} detector`);
  }
}

/** Untick every row of *tag*, so the dashboard shows a clean card. */
async function deselectAll(page, tag) {
  const checked = `${tag} .select-checkbox[aria-checked="true"]`;
  for (let guard = 0; guard < 30 && (await page.locator(checked).count()); guard++) {
    await page.locator(checked).first().click();
    await page.waitForTimeout(300);
  }
}

/**
 * Drive every row of *tag* to what this shot needs, rather than only ticking
 * the one we want: selection persists server-side, so a rerun (or the previous
 * shot's fixture) can leave the wrong rows ticked.
 *
 * Match the name cell exactly, not the row's text — `photos` is a substring of
 * `photos-prod`, and a substring match ticks both, which leaves Train and Find
 * permanently disabled and looks exactly like a hung page.
 */
async function selectOnly(page, tag, name) {
  const rows = page.locator(tag);
  for (let i = 0; i < (await rows.count()); i++) {
    const row = rows.nth(i);
    const cell = row.locator('.name-cell').first();
    const label = (await cell.count()) ? (await cell.textContent()) || '' : '';
    const wanted = label.trim() === name;
    const box = row.locator('.select-checkbox').first();
    if (((await box.getAttribute('aria-checked')) === 'true') === wanted) continue;
    await box.click();
    await page.waitForTimeout(350);
  }
}

async function openDashboard(page) {
  await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.dash-table', { timeout: 60000 });
  await page.waitForTimeout(1500);
}

/**
 * Step 1 — name the concept.
 *
 * Four pages, and they are four clicks: the dashboard with a pile of media and
 * no detector, the dialog, the dialog with the concept written into it, and the
 * dashboard with the detector that was not there before. The dataset row is
 * selected first because the modal takes its media type and its embedder from
 * whatever is active — a detector created against nothing is a detector the
 * next two shots could not use.
 */
async function shootMakeDetector(page) {
  await openDashboard(page);
  await selectOnly(page, 'tr[vt-dataset-card]', 'photos');
  await deselectAll(page, 'tr[vt-detector-card]');
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  await shoot(page, 'ui-make-detector.build1');

  await page.locator('button[title="Create a new detector"]').click();
  await page.waitForSelector('.new-detector-form', { timeout: 20000 });
  await page.waitForTimeout(900);
  await shoot(page, 'ui-make-detector.build2');

  // The text tab is the default, and it is the one the deck's argument needs:
  // the whole claim of the slide before this is that the concept is a phrase
  // somebody can say and not a query they can write.
  await page.locator('.example-panel input.form-input').first().fill(INTRO_TEXT);
  await page.locator('#detector-name').fill(INTRO_DETECTOR);
  await page.waitForTimeout(700);
  await shoot(page, 'ui-make-detector.build3');

  await page.getByRole('button', { name: /^Creat/ }).last().click();
  await page.waitForSelector('.new-detector-form', { state: 'detached', timeout: 60000 });
  await waitFor(`the ${INTRO_DETECTOR} detector`, async () => named(await detectors(), INTRO_DETECTOR));
  await page.waitForTimeout(2000);
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  await shoot(page, 'ui-make-detector');
}

/**
 * Answer whatever autopilot just put on screen, truthfully.
 *
 * Truthfully is the point: the button is chosen from the served item's own
 * file name, and the corpus files a frame under `book/` only when COCO's
 * largest box in it is a book (see `coco_fixture._roster`). So the piles that
 * grow through the build are the piles a person would have produced, and the
 * one thing a staged screenshot cannot show — that the tool asks about items
 * it cannot call, and is sometimes told no — is visible in them.
 */
async function voteServed(page) {
  const viewer = page.locator('img.image-element').first();
  const before = await viewer.getAttribute('alt');
  const good = (before || '').startsWith('book/');
  await page.locator(good ? '.btn-good' : '.btn-bad').first().click();
  // The vote retrains the head and re-sorts, and autopilot then serves a
  // different item. Waiting on the served item *changing* waits for all of it;
  // waiting on a fixed delay waits for whichever part happens to be slowest.
  await page
    .waitForFunction(
      (prev) => document.querySelector('img.image-element')?.alt !== prev,
      before,
      { timeout: 120000 }
    )
    .catch(() => {});
  await page.waitForTimeout(1800);
  return good;
}

/**
 * Step 2 — answer, and answer again.
 *
 * Six pages of one screen: the votes cast so far accumulate in the right-hand
 * panel and nothing else on the slide moves, which is the build rule and also
 * the honest description of the interaction. Autopilot is collapsed to its rail
 * for the reason `collapseIntoAutopilot` gives — what is left is the item and
 * the two buttons.
 */
async function shootTrainLoop(page) {
  await enterLabelView(page, 'photos', INTRO_DETECTOR);
  await collapseIntoAutopilot(page);
  await page.waitForSelector('.btn-good', { timeout: 120000 });
  await page.waitForTimeout(1500);

  let cast = 0;
  const tally = { good: 0, bad: 0 };
  for (const stage of TRAIN_STAGES) {
    while (cast < stage) {
      tally[(await voteServed(page)) ? 'good' : 'bad']++;
      cast++;
    }
    const page_no = TRAIN_STAGES.indexOf(stage) + 1;
    await shoot(page, `ui-train-loop.build${page_no}`);
  }
  while (
    cast < TRAIN_FINAL.maxVotes
    && (tally.good < TRAIN_FINAL.good || tally.bad < TRAIN_FINAL.bad)
  ) {
    tally[(await voteServed(page)) ? 'good' : 'bad']++;
    cast++;
  }
  log(`train loop: ${cast} votes — ${tally.good} good / ${tally.bad} bad`);
  if (!tally.bad) throw new Error('no Bad votes: the detector has nothing to separate');
  await shoot(page, 'ui-train-loop');
}

/**
 * Scroll the left panel's virtual viewport, and let it re-render.
 *
 * It is a `cdk-virtual-scroll-viewport`, so what is in the DOM is a window onto
 * the ranking rather than the ranking — and the panel scrolls itself to the
 * selected item on arrival, which is how the first version of the Find shot
 * came out photographing position 4200px with the top of the list nowhere in
 * frame. Drive it explicitly instead of hoping.
 */
async function scrollResults(page, to) {
  const viewport = page.locator('.panel-left .cdk-virtual-scroll-viewport').first();
  await viewport.evaluate((el, top) => el.scrollTo({ top, behavior: 'instant' }), to);
  await page.waitForTimeout(1800);
}

/**
 * Step 3 — run it over the media nobody voted on.
 *
 * Two slides out of one session. `ui-find[.build1]` is the click-by-click one:
 * the dashboard with the *production* pile selected beside the detector, then
 * the top of the ranking it produces. `photos-prod` does not share a single
 * frame with `photos` (`coco_fixture.DISJOINT_FROM`), which is the only reason
 * that slide is allowed to say what it says.
 *
 * `ui-find-line` is the same screen scrolled down to the line the tool drew
 * through the ranking, and it is a separate slide rather than a third build
 * page for a reason the house rules are explicit about: a build page adds ink
 * and moves nothing, and this one moves the whole left panel. It is also not a
 * reveal but a second observation — the deck's hand-off into `vote-boundary`,
 * which spends the next ten minutes on where that line should go.
 */
async function shootFind(page) {
  await openDashboard(page);
  await selectOnly(page, 'tr[vt-dataset-card]', 'photos-prod');
  await selectOnly(page, 'tr[vt-detector-card]', INTRO_DETECTOR);
  await page.mouse.move(700, 120);
  await page.waitForTimeout(400);
  await shoot(page, 'ui-find.build1');

  await page.getByRole('button', { name: 'Find', exact: true }).click();
  await page.waitForSelector('.panel-right', { timeout: 300000 });
  await page.getByText('Verified Good').first().waitFor({ timeout: 300000 });
  // Scoring puts an overlay over the centre panel; wait it out rather than
  // photographing a progress bar.
  await page.waitForSelector('.find-wait-overlay', { state: 'detached', timeout: 300000 })
    .catch(() => {});
  await page.waitForTimeout(3000);

  // The best match in the centre, and the best matches beside it. Selecting the
  // top item re-scrolls the panel to it, so the scroll goes after the click.
  await scrollResults(page, 0);
  await page.locator('.panel-left .thumbnail-wrap').first().click();
  await page.waitForTimeout(1500);
  await scrollResults(page, 0);
  await shoot(page, 'ui-find');

  // Then the line. Its offset is read off the rendered list rather than
  // computed from a rank, because how many items make a row is the panel's
  // business (thumbnail size, panel width) and not something this script knows.
  const line = await page.evaluate(() => {
    const viewport = document.querySelector('.panel-left .cdk-virtual-scroll-viewport');
    const marker = viewport?.querySelector('.media-threshold-line');
    if (!viewport) return null;
    if (!marker) return 'offscreen';
    return viewport.scrollTop + marker.getBoundingClientRect().top
      - viewport.getBoundingClientRect().top;
  });
  if (line === null || line === 'offscreen') {
    // Virtualised: the marker is only in the DOM once it is near the window, so
    // walk down until it appears rather than guessing a pixel offset.
    for (let top = 0; top < 40000; top += 500) {
      await scrollResults(page, top);
      if (await page.locator('.media-threshold-line').count()) break;
    }
  } else {
    await scrollResults(page, Math.max(0, line - 260));
  }
  const marker = page.locator('.media-threshold-line').first();
  if (!(await marker.count())) throw new Error('no threshold line in the Find ranking');
  // Centre it: whatever the walk above landed on, the line should sit in the
  // middle of the panel with matches above it and rejects below.
  const centred = await page.evaluate(() => {
    const viewport = document.querySelector('.panel-left .cdk-virtual-scroll-viewport');
    const rect = viewport.querySelector('.media-threshold-line').getBoundingClientRect();
    const box = viewport.getBoundingClientRect();
    return viewport.scrollTop + rect.top - box.top - box.height / 2;
  });
  await scrollResults(page, Math.max(0, centred));
  await shoot(page, 'ui-find-line');
}

async function shootRegionVoting(page, voted) {
  await enterLabelView(page, 'photo-regions', 'books-regions');
  await leftTab(page, 'Manual');
  await serveItem(page, [HERO_REGION], voted);
  await collapseIntoAutopilot(page);
  // The drawing tools live in the centre panel, so the box can be drawn after
  // the left panel has been folded away.
  await page.locator('.ivc-btn-toggle, button[title*="Marquee" i]').first().click();
  await page.waitForTimeout(700);
  // The rendered *picture*, not the <img> element and not its wrapper. The
  // viewer sizes the element to the whole centre panel and uses
  // `object-fit: contain`, so the element's bounding box is much taller than
  // the photo inside it: fractions of the element put the drag outside the
  // picture, the app clamps the box back to the image edges, and a
  // hand-measured box comes out spanning the full height (#3246).
  const box = await page.locator('img.image-element').first().evaluate((img) => {
    const r = img.getBoundingClientRect();
    const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    return { x: r.x + (r.width - w) / 2, y: r.y + (r.height - h) / 2, width: w, height: h };
  });
  if (!box) throw new Error('no image in the centre viewer to draw on');
  const x0 = box.x + box.width * REGION_BOX.x0;
  const y0 = box.y + box.height * REGION_BOX.y0;
  const x1 = box.x + box.width * REGION_BOX.x1;
  const y1 = box.y + box.height * REGION_BOX.y1;
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 8 });
  await page.mouse.move(x1, y1, { steps: 8 });
  await page.mouse.up();
  await page.waitForSelector('.region-box', { timeout: 15000 });
  await page.waitForTimeout(700);
  // The drawn box is already the loudest thing on the screen and it is the
  // right colour for it. A second red box with a red caption inside it hides
  // the one the audience is meant to read (#3246).
  await shoot(page, 'ui-region-voting');
}

// ── main ─────────────────────────────────────────────────────────────────────

let appProcess = null;
async function ensureApp() {
  try {
    if ((await fetch(APP + '/api/version')).ok) return;
  } catch {
    /* not running */
  }
  log('no app running — starting one');
  appProcess = spawn('python', ['app.py', '--local'], {
    cwd: REPO,
    env: { ...process.env, VTSEARCH_TORCH_THREADS: '2' },
    stdio: 'ignore',
    detached: false,
  });
  await waitFor('the app', async () => {
    try {
      return (await fetch(APP + '/api/version')).ok;
    } catch {
      return false;
    }
  }, 300000);
}

// The three intro shots are one session and are taken together: the detector
// `make-detector` creates is the one `train-loop` votes on and `find` runs, so
// asking for a later group alone would shoot it against whatever the last full
// run left behind.
const INTRO = ['make-detector', 'train-loop', 'find'];
const intro = INTRO.some(wanted);

await ensureApp();
if (intro) await ensureDataset('photos', 'siglip');
// The pile the detector has never seen. Embedded up front rather than between
// the second and third shots, so the Find click in the captured session is the
// click a user makes and not a five-minute wait dressed up as one.
if (intro) await ensureDataset('photos-prod', 'siglip');
let regionsVoted = new Set();
if (wanted('region-voting')) {
  // A second detector, not the same one: a detector binds an embedder *type*
  // at creation, and a patch dataset offers `patch_semantic` where the SigLIP
  // one offers `semantic`. Point `books` at `photo-regions` and the app
  // correctly refuses the pair — which is the whole reason region voting needs
  // its own dataset in the first place.
  const regions = await ensureDataset('photo-regions', 'dinov2_patch');
  regionsVoted = await ensureVotes(regions, await ensureDetector('books-regions', regions), REGION_VOTES);
}

const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: SCALE });
  await page.addStyleTag({ content: STILL_CSS }).catch(() => {});
  await page.addInitScript((css) => {
    document.addEventListener('DOMContentLoaded', () => {
      const s = document.createElement('style');
      s.textContent = css;
      document.head.appendChild(s);
    });
  }, STILL_CSS);
  if (intro) {
    await resetIntroDetector();
    await shootMakeDetector(page);
    await shootTrainLoop(page);
    await shootFind(page);
  }
  if (wanted('region-voting')) await shootRegionVoting(page, regionsVoted);
} finally {
  await browser.close();
  if (appProcess) appProcess.kill();
}

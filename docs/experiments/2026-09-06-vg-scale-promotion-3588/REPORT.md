# Promoting the #3588 thirteen into *C* (#3588, #3604)

**`SCALE_CLASSES` goes from twelve classes to twenty-five, and the step between
a cleared review and a shipped class list is a measurement nobody had made.**
#3588 reviewed thirteen candidates at 300 images each and cleared all thirteen;
that says a human can label them consistently. It says nothing about which VG
spellings the class is *built* from, which is a separate measurement, and the
suite refuses a class in *C* that is missing from `SCALE_VG_NAMES_AUDITED` for
exactly that reason — `bicycle` shipped for the whole of #3156 built from one
spelling with every structural check passing (#3605).

**Three things a reader should take away before the detail.**

**The audit was worth running, and its largest gains are the classes whose plain
name is worst.** `cell phone` recovers **+1,280** images on the half of VG COCO
cannot score, against the 535 it could already see — **+239%** — and
`fire hydrant` goes from **44.7%** to **73.8%** of COCO's boxes on the strength
of one spelling. A class's plain name is not its vocabulary, and how badly it
misses is not predictable from the name (§2).

**Adopting a per-class audit wholesale deleted a class.** The audit adjudicates
one class at a time, so it offered `truck` — a class in *C* — as ambiguous
evidence for `car`. `lift_ambiguous` pops the box before its exemptions are
consulted, so adopting that took `truck` from **3,386 band-free positives to 0
in all three bands**, with the fold ledger still printing `truck+273/23` for a
class that no longer existed (§3).

**The measurement that sized the deep sibling could not see the alias table.**
`measure_supply.py` read the bare class names and skipped `canonicalise` /
`lift_ambiguous`, so it reported the supply of a dataset nobody builds. The tell
was that adding thirteen classes and 182 spellings changed its output by exactly
**zero images** (§4).

| | |
|---|---|
| Classes in *C* | 12 → **25** (75 cells) |
| Alias spellings | 49 → **182** (the thirteen bring **133**) |
| Ambiguous spellings | 58 → **254** (the thirteen bring **196**) |
| `SCALE_VG_NAMES_AUDITED` | 12 → **25** |
| `SCALE_N_NEG_SPARE` | 300 → **1,000** |
| Provable negative supply | 34,071 (12 classes) → **21,121** (25) against 9,900 needed |
| Realised cell prevalence | 0.85% (12) → **0.73%** (25) |
| `SCALE_DEEP_N_POS` | **900, unmoved** — all twenty-five clear it band-free |

Scripts: the audit is `vg_name_families.py` → `name_evidence.py` →
`name_coverage.py` at the cuts #3618 and #3636 shipped under (`--pooled
--include-context`); the supply is `measure_supply.py`, repaired here; the pool
is `negpool_supply.py` and `negpool_coverage.py` from #3670.

---

## 1. What a promotion actually is

Four edits, and the review is only the licence for the first:

1. the class into `SCALE_CLASSES`;
2. its measured spellings into `SCALE_VG_NAMES` / `SCALE_VG_AMBIGUOUS`, out of
   the candidate tables and **minus the class name itself**;
3. the class into `SCALE_VG_NAMES_AUDITED`, *after* running the audit — whatever
   the verdict, since an audit that found nothing is the result the flag exists
   to record;
4. the negative pool redrawn and the pile rebuilt, because a new class evicts
   whatever the pool holds of it (#3604).

Step 3 is the one #3604's plan did not name, and it has to happen before step 4:
the spellings decide which images are positives and which are clean, so running
the audit after the rebuild means rebuilding again.

There is a chicken-and-egg in the tooling worth recording. `name_evidence.py`
takes no `--classes`; it reads `SCALE_CLASSES`. So the thirteen have to be in
*C* before the audit that qualifies them for *C* can run. The order used here
was: add the class list, run the audit, write the tables, then add the audited
flag — with the suite red in between, which is the flag doing its job.

## 2. The audit

4,604 candidate names over the thirteen — the union of `coco_folds.py`'s
fold-in column and `vg_name_families.py`'s head-noun families — adjudicated
against COCO on the 51,411-image overlap. The audit proposes **131 alias** and
**198 ambiguous** spellings; after the adjudication below — four aliases demoted,
six merge spellings carried by hand, six cross-class names dropped — the thirteen
ship **133** alias and **196** ambiguous, taking the whole tables from 49 and 58
to **182** and **254**.

### What the alias table buys

| class | own % of COCO's boxes | +alias | non-COCO repaired | of which banded |
|---|---:|---:|---:|---:|
| `fire hydrant` | 44.7% | **73.8%** | +282 (on 411 own) | 276 |
| `cell phone` | 15.5% | **44.9%** | **+1,280** (on 535) | 1,197 |
| `car` | 23.2% | 31.4% | +1,588 (on 5,122) | 1,320 |
| `truck` | 25.6% | 29.4% | +191 (on 1,499) | 167 |
| `chair` | 22.2% | 24.9% | +361 (on 3,712) | 297 |
| `bottle` | 23.3% | 27.8% | +375 (on 2,085) | 320 |
| `cup` | 16.7% | 22.2% | +386 (on 1,616) | 345 |
| `bowl` | 30.5% | 32.1% | +51 (on 1,828) | 44 |
| `fork` | 41.9% | 42.9% | +34 (on 1,037) | 34 |

The two extremes are the point. `hydrant` is one spelling carrying 314 boxes at
87% on-class — taking `fire hydrant` alone throws away a third of the class.
`phone` carries 1,035 boxes at 54% box agreement and 68% precision over 884 sole
images: both above the cut, on a class `coco_folds.py` scores at 46% definition
risk because VG's `phone` is nearly half landlines. The table folds it and
`SCALE_CLASS_RULES["cell phone"]` settles the landline, which is the division of
labour the two are for.

### Where the audit had to be overruled, and why

Adopting the proposal verbatim is not "adopting at the shipped cuts", because
the shipped tables were also filtered by hand against the class rulings — `sign`
is excluded from `stop sign` despite being the largest fold-in in *C*, and there
is a regression test that keeps it out. Four proposed aliases were demoted to
ambiguous here on the same grounds:

- **`van` → ambiguous for `car`** (72% precision, 51% box). COCO splits `van`
  **261 truck / 318 car / 37 bus**, which is the measurement
  `SCALE_CLASS_RULES["truck"]` was written from. A 72% precision against COCO
  `car` is that coin-flip reported as an alias.
- **`vehicle` → ambiguous for `car`** (60%, 55%). A superordinate covering
  `truck`, `bus` and `bicycle` — three other classes in *C* — so folding it
  would make their images `car` positives.
- **`chair back` → ambiguous for `chair`** (62%, 56%). A part; a part's box is
  not the object. `beak` (86%) and `knife block` (79%) are in the ambiguous
  table at *higher* precision than this for the same reason.

### The audit cannot see a class merge

`cup` is `cup` ∪ `wine glass` (`SCALE_CLASS_MERGES`), and `name_evidence.py`
resolves "the class" to the single COCO class of the same name. So it scores the
stemware against COCO **`cup`** alone:

| name | sole | precision | box agreement | verdict |
|---|---:|---:|---:|---|
| `wine glass` | 151 | 38% | **2%** | `neither` |
| `wine glasses` | 22 | 18% | 3% | `neither` |
| `mug` | 193 | 88% | 82% | `alias` |

The 2% is the tell: those boxes land on COCO `wine glass` boxes, which the
scorer is not looking at. `mug`, whose object COCO really does call a cup,
scores normally — so the blind spot is specific to the merged half and silent
everywhere else. The six stemware spellings are carried by hand on #3588's own
measurement against COCO `wine glass` boxes. Filed as **#3700**.

## 3. Adopting a per-class audit deleted a class

`lift_ambiguous` drops an ambiguous spelling's boxes out of `labels` — and the
drop is unconditional, taken before the three exemptions its docstring describes
are consulted. So an entry naming a class in *C* deletes that class's own boxes
from every image in the corpus.

The audit adjudicates one class at a time, so it duly offered a neighbour's
object as ambiguous evidence for the class beside it. Six such names came out of
it, all on the truck/car and bowl/cup boundaries:

| name | proposed as | owned by |
|---|---|---|
| `truck` | ambiguous for `car` | **`truck`, a class in *C*** |
| `pick up`, `pick up truck`, `pickup truck` | ambiguous for `car` | alias of `truck` |
| `jeep` | ambiguous for `truck` | alias of `car` |
| `bowls` | ambiguous for `cup` | alias of `bowl` |

The shipped invariant `test_a_spelling_is_never_both_an_alias_and_ambiguous`
caught five of the six. It could not catch the sixth, and the sixth is the one
that mattered: `truck` is a table **key**, not one of its values, so a test
comparing the union of alias names against the union of ambiguous names never
sees it. Measured cost of adopting it: `truck` supply

| | small | medium | large | band-free |
|---|---:|---:|---:|---:|
| as adopted | **0** | **0** | **0** | **0** |
| after the fix | 477 | 1,596 | 1,454 | 3,527 |

and the build said nothing. The fold ledger still printed `truck+273/23`,
because `canonicalise` runs before `lift_ambiguous` and had already folded the
boxes that were about to be deleted.

### The zeroed supply is the visible half; the pool is the dangerous one

The #3670 session, reconciling the two changes, pointed out that the supply
collapse is not the worst of it. On a COCO-anchored image `anchor_to_coco` has
already replaced VG's labels with COCO's, so for an exhaustive image whose only
*C* label is a COCO-confirmed `truck`:

- the box is popped;
- **no** `unbanded` pair is added, because `iid in exhaustive` exempts it;
- `by_name` is now empty, so `band_candidates` files the image as **clean**;
- it is COCO-scored, so #3670's `provable` draw designates it.

That puts COCO-confirmed trucks into the negative pool whose entire claim is
that it holds none of *C* — and `--verify`'s composition check cannot catch it,
because those images really do carry the `coco_scored` stamp it tests. Filed as
**#3701**: the pool should be asked the question directly, by intersecting the
designated negatives with COCO's own annotation, because that invariant holds
regardless of which pass broke it.

### Why the ban, and not a narrower fix

`lift_ambiguous`'s condition is inverted rather than incomplete. It builds
`reverse` with `if n != cls`, which implements the docstring's third exemption —
*the name is the class name itself* — but asks "is this name its **own** class's
name?" when the question is "is this name a class in *C* **at all**?". The same
shape as #3618's fold-in being conditioned the wrong way round.

A narrower fix is available: suppress the `(image, class)` pair without popping a
box whose name is a class in *C*, which keeps the evidence and costs nothing in
the pool. It is not taken, because there is no evidence here to keep. The
ambiguous table is for names whose **referent is uncertain** — `bike` may be a
bicycle or a motorcycle, `van` is split 261/318/37 across three classes — and
neither is a class in *C*. A name that *is* a class has a certain referent; what
it has with a third class is **co-occurrence**. The audit's precision for
`truck`-as-evidence-for-`car` is measuring that trucks and cars share streets,
and under #3667 a COCO-scored truck image is precisely what `car` should be
scored against as a negative.

The cut therefore lands between "uncertain referent" and "certain referent,
correlated", and it costs no genuinely ambiguous name: `van` is not a class in
*C* and stays ambiguous for both `car` and `truck`.

`test_no_listed_spelling_is_itself_a_class_in_c` is the guard, and it is a suite
test rather than a check inside `lift_ambiguous` because that function takes the
table as an argument and is exercised with deliberately small ones.

## 4. The supply measurement could not see the alias table

`measure_supply.py` reproduces the loader's front half — except that it read
`set(SCALE_CLASSES)`, the bare class names, and never called `canonicalise` or
`lift_ambiguous`. Every alternate spelling in `SCALE_VG_NAMES` was therefore
invisible to it: it measured the supply of a dataset nobody builds.

The tell was availability of a control. Running it before and after a promotion
that added thirteen classes and 182 spellings changed its output by **exactly
zero images** — `fire hydrant` 1,138 both times, when `hydrant` alone carries a
third of that class. A measurement that cannot move when its input changes is
not measuring its input.

This matters beyond tidiness because **#3547 sized `SCALE_DEEP_N_POS` = 900 off
this script**. That number survives — `stop sign`, which has no alias row at
all, is still the binding class at 1,006 — so the conclusion was right by
accident, not by measurement. The classes that moved are the ones with alias
rows: `cell phone` 2,166 → **3,341**, `fire hydrant` 1,138 → **1,412**, `kite`
1,186 → 1,204.

Repaired here to call the loader's own passes in the loader's own order.

## 5. Supply, and the depth question

All twenty-five clear 900 positives band-free, so `vg_scale_deep` keeps its
depth *and* its whole class list — which was not a foregone conclusion. The
thinnest of the thirteen is `fire hydrant` at 1,412, still clear of `stop sign`
at 1,006; had one come in under 900 the choice would have been between a deep
set carrying fewer classes than the shallow one and a depth change that restates
#3547's published optimum.

Going deeper still costs classes: 1,200 drops `stop sign` alone, where before
the fold it dropped `kite` and `fire hydrant` too.

The thinnest single band across all twenty-five is 145 (`bus@small`), against
`SCALE_N_POS` = 100.

## 6. What the promotion does to #3670

#3670 landed hours earlier and had deferred its own rebuild so the pile would be
built once for both. Two of its numbers are conditioned on twelve classes and
are restated here at twenty-five:

| | 12 classes (#3670) | 25 classes |
|---|---:|---:|
| Provable negatives available | 34,071 | **21,121** |
| Headroom over `SCALE_N_NEG` = 9,900 | 3.4× | **2.1×** |
| Realised cell prevalence | 0.85% (0.844–0.856) | **0.73%** (0.72–0.75) |

The all-provable pool stays feasible, which is the load-bearing conclusion: 9,900
does not have to move. Prevalence falls again for the reason #3667 named — a
cell scores its shared negatives *plus* every other class's COCO-scored
positives, and there are now twenty-four other classes rather than eleven — so
the designed 1.00% is realised at 0.73%. Quote 0.73% (#3681).

**#3670 also dissolves a step this promotion was supposed to carry.** #3604's
plan called for a human correction pass over `car`'s negatives before the
rebuild, because `car`'s pool error was measured at 7.1% — the worst of the
thirteen. A pool drawn entirely from the COCO-scored half has **no** such error
by construction: COCO annotates all eighty of its classes on any image it
touches, so "holds no car" is a fact there. The pass was the right call against
a 45%-COCO pool and is redundant against a 100% one.

## 7. The rebuild, and what it does to the negative review

Growing *C* does not merely add cells. An image sitting in the shared negative
pool because it held none of the twelve may hold one of the thirteen, and then it
cannot be a negative for *anything* — so a correct rebuild must drop it. Measured
on this build: **1,670 rostered negatives disqualified** at the moment of the
promotion.

`check_review_coverage.py` forgave exactly one reason for a reviewed image
leaving the pool — a correction removed it — so it read every one of those as
**lost review**, which is the alarm that protects the most expensive input this
dataset has. A 25-class build fails that gate at ~60.8%. The fix
(`disqualified_negatives`, contributed by the parallel #3588 session and
cherry-picked here as `5f1e2e91b`) records the disqualification in the roster at
the one moment it is knowable, because by the time the gate runs the reason is
gone.

Two traps in it, both found by that session and both worth keeping in view:

- **It has to accumulate.** `load` runs once per embedder and rewrites the roster
  each time, so the first cell sees the old negatives and records the event while
  every later cell reads a roster whose negatives are already clean, computes an
  empty set, and overwrites the fact. It survived exactly one cell of a five-cell
  build before the function existed — the same shape as the counter placed before
  the pass that discards its work (#3637).
- **The event is consumed by the first build that sees it.** A rebuild from a
  roster that has already crossed the 12→25 transition records an empty set even
  with correct code, and an empty set looks exactly like success. This build was
  therefore launched from the archived **pre-expansion** roster
  (`archive/pre-3670-negpool/`, 36 cells and 3,900 negatives), per #3698's
  "start from the roster the change starts from". An earlier attempt inherited a
  25-class roster from a parallel branch and was discarded for that reason.

**The retirements are legitimate, and the attribution is what says so.** Of the
345 retired *reviewed* negatives, **79%** hold a newly-added class (`car` 118,
`chair` 48, `truck` 43, `cell phone` 35, `bench` 30), 9% are ambiguous
suppression, 7% a correction, 6% an original class, and **none** unexplained
(`why_retired.py`, measured by the parallel #3588 session). The gate was blind to
the largest cause, not wrong about the loss.

One more measurement from that session, which retires a blocker rather than
raising one: the **global ambiguous exclusion** (#3655 — one ambiguous name costs
every class the image) withholds **2,142 images at twenty-five classes, 4.4% of
the clean pool** (48,345 → 46,203). It had been queued as a blocker on the
strength of the twelve-class figure; at this scale it is not one.

### What the built pile verifies as

| check | result |
|---|---|
| `build_pile.py --verify` | all fifteen `vg_scale*` cells ok, 18,050 medias (deep 31,832) |
| reviewed negatives | 743 reviewed, 591 ruled out by composition, 49 by a fix, **103 of 103 kept — 100.0%** |
| triaged negatives | 1,742 reviewed, **261 of 261 kept — 100.0%** |
| reviewed positives | **314 of 360 — 87.2%** |
| COCO-scored designated negatives | **9,900 of 9,900** |
| …holding any class of *C* | **0** — `POOL_CLEAN` |

The last row is #3701's check run once by hand, and it is the one worth having:
it tests #3670's claim **directly** rather than through the `coco_scored` stamp,
which is exactly the distinction the `truck` defect turns on — those images would
have carried a valid stamp while holding a confirmed truck. It also settles
`car`: the 7.1% pool error that #3604 planned a human pass around is measured at
zero, not merely unmeasured.

`reviewed positives` at 87.2% is the one number not at 100%. It is 46 of 360, and
that is exactly the figure #3698 records for #3667's rebuild ("five rulings, 46 of
360 reviewed positives"). The roster this build started from post-dates #3667, so
these are the same 46 carried forward rather than new losses — stated as a match
rather than a proof, since the id sets were not diffed.

## 8. Follow-ups

- **#3700** — the VG-name audit is blind to `SCALE_CLASS_MERGES`, so it scores
  stemware as `neither` for `cup` and under-reports that class's coverage.
- **#3701** — `--verify` should ask COCO directly whether the provable pool holds
  any class of *C*, rather than testing the `coco_scored` stamp and trusting the
  passes upstream of the draw.
- **#3696** (#3670's) — an audit stratum drawn from `clean` measures VG's
  silence-error on a population `lift_ambiguous` has already cleaned of its
  hardest cases, so it under-estimates the rate that motivated `provable`. The
  stratum has to be drawn before the lift, or from a frame recording which
  suppressions it kept.

<!-- _class: full -->

![bg fit](figs/ui-train-loop.webp)

## Twenty Questions

<!-- build: figs/ui-train-loop.build1.webp -->

<!-- build: figs/ui-train-loop.build2.webp -->

<!-- build: figs/ui-train-loop.build3.webp -->

<!-- build: figs/ui-train-loop.build4.webp -->

<!-- build: figs/ui-train-loop.build5.webp -->

<!-- **a** — Train, one second after opening it, and the interaction is already
     complete: one item in the middle, **Good** and **Bad** under it, the answers
     so far on the right. Nothing else is asked of anybody. The sliver on the far
     left is autopilot, folded to a rail — it is choosing what to put in front of
     you, so nobody is scrolling a result list deciding what to judge. -->

<!-- **b** — First answer, and it lands in the Good pile. Behind it: the head
     retrains and the whole corpus re-ranks. That is a fraction of a second,
     because it is a small linear model on frozen embeddings — the heavy network
     ran once, at import, and never runs again. -->

<!-- **c** — Second. Watch the middle as much as the right: the item on screen
     changed, and it changed because the model that just retrained picked the
     thing it is now least able to call. -->

<!-- **d** — Third, and this one is a No. That is the loop working rather than
     failing. A corpus with magazine racks, boxed DVD sets and spiral notebooks
     in it has plenty of items nobody can call from the seed phrase, and those
     are precisely the ones worth spending a question on. -->

<!-- **e** — Fourth. The rhythm is the whole product and it does not develop:
     look, click, look, click. There is no second mode to learn, no threshold to
     tune, nothing to configure between answers. -->

<!-- **f** — And here it is some way in. Two piles, a couple of dozen clicks, a
     few minutes — which is the entire budget the task was ever going to get.
     (For anyone asking where the rest of the corpus went: there is a manual
     mode with the whole pile in a grid, sort controls and a threshold slider.
     That slider is a character in the second half of the talk.) -->

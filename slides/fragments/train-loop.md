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
     so far — none — on the right. Nothing else is asked of anybody. The sliver
     on the far left is autopilot, folded to a rail: it is choosing what to put
     in front of you, so nobody is scrolling a result list deciding what to
     judge. -->

<!-- **b** — First answer, and it lands in the Good pile. Behind it, the head
     retrains and all 228 items re-rank. That is a fraction of a second, because
     it is a small linear model on frozen embeddings — the heavy network ran
     once, at import, and never runs again. -->

<!-- **c** — Second. Watch the middle as much as the right: the item changed,
     and it changed because the model that just retrained went looking for what
     it could least call. Which is why it is now showing you teddy bears in
     front of a shelf of DVD box sets. -->

<!-- **d** — Third. Same again. There is no second mode to learn, no threshold
     to set, nothing to configure between answers — the rhythm is the whole
     product and it does not develop. -->

<!-- **e** — Fourth, and the first **No**: that cat in front of a television was
     a Bad, and the pile on the right now has two halves. This is the loop
     working rather than failing. A corpus with magazine racks, DVD cases and
     spiral notebooks in it is full of items nobody can call from the seed
     phrase, and those are precisely the ones worth spending a question on. -->

<!-- **f** — And here it is fourteen questions in — eight Good, six Bad, a few
     minutes, which is the whole budget this task was ever going to get. Note
     what it is asking about *now*: a desk flat-lay with a laptop, a phone,
     keys, coins and one paperback in the middle of them. Nobody in this room
     agrees on that one either, which is the point. -->

<!-- If someone asks where the rest of the corpus went: there is a manual mode
     with the whole pile in a grid, sort controls and a threshold slider. That
     slider is a character in the second half of the talk. -->

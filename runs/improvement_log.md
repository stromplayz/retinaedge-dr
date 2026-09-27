# RetinaEdge-DR Improvement Campaign Log

Goal: referable-DR accuracy >= 0.97 on the held-out validation split (Wilson CI lower
bound gated). Each entry below is one ladder rung executed by
`retinaedge.train.auto_improve`; state lives in `runs/improve_state.json`.

> Pre-campaign note: a local plumbing probe (600-image slice, random init, 1 epoch)
> verified the full loop (train -> TTA -> soup -> threshold search -> Wilson CI ->
> markdown log) end-to-end. Its numbers were a pipeline check, not a model result;
> the campaign state was reset before the first in-GitHub run.
## s1-baseline — budget epochs at default resolution
*2026-09-26 23:45:00 UTC | overrides: `train.epochs=2` | 701s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.5220 | 0.5140 | 0.5388 | 0.1319 | 0.7743 | 0.6233 | 0.8878 |
| tta | 0.5186 | 0.5089 | 0.5366 | 0.1405 | 0.7727 | 0.6599 | 0.8574 |
| soup | 0.5802 | 0.7243 | 0.5507 | 0.5860 | 0.8226 | 0.8028 | 0.8375 |
| soup+tta | 0.5778 | 0.7276 | 0.5502 | 0.5974 | 0.8242 | 0.7712 | 0.8641 |

**Best variant:** `soup+tta` — acc_refer **0.8242** (95% Wilson CI [0.8062, 0.8409], n=1843), QWK 0.7276
**Goal 97.00%:** not reached (gap +14.58%)

## s2-longer-ema — longer schedule + EMA weight averaging
*2026-09-27 00:06:49 UTC | overrides: `train.epochs=4 train.ema=true train.patience=4` | 1309s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.6045 | 0.7373 | 0.5502 | 0.6224 | 0.8329 | 0.7788 | 0.8736 |
| tta | 0.6024 | 0.7407 | 0.5502 | 0.6023 | 0.8318 | 0.7573 | 0.8878 |
| soup | 0.6394 | 0.7629 | 0.5659 | 0.5838 | 0.8437 | 0.7750 | 0.8954 |
| soup+tta | 0.6324 | 0.7707 | 0.5616 | 0.5822 | 0.8437 | 0.7863 | 0.8869 |

**Best variant:** `soup+tta` — acc_refer **0.8437** (95% Wilson CI [0.8264, 0.8596], n=1843), QWK 0.7707
**Goal 97.00%:** not reached (gap +12.63%)


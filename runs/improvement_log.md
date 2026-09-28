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

## s3-sharper — higher input resolution
*2026-09-28 09:52:54 UTC | overrides: `data.img_size=192 train.epochs=4 train.ema=true` | 631s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.5922 | 0.7338 | 0.5328 | 0.5317 | 0.8285 | 0.7585 | 0.8812 |
| tta | 0.5819 | 0.7268 | 0.5301 | 0.5100 | 0.8226 | 0.7547 | 0.8736 |
| soup | 0.5806 | 0.7446 | 0.4938 | 0.5524 | 0.8323 | 0.7472 | 0.8964 |
| soup+tta | 0.5968 | 0.7517 | 0.5057 | 0.5882 | 0.8318 | 0.7484 | 0.8945 |

**Best variant:** `soup` — acc_refer **0.8323** (95% Wilson CI [0.8146, 0.8487], n=1843), QWK 0.7446
**Goal 97.00%:** not reached (gap +13.77%)

## s4-backbone — stronger backbone (efficientnet_lite0)
*2026-09-28 10:36:20 UTC | overrides: `model.backbone=efficientnet_lite0 data.img_size=192 train.epochs=4 train.ema=true` | 2607s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.6641 | 0.7866 | 0.5724 | 0.6446 | 0.8492 | 0.7813 | 0.9002 |
| tta | 0.6597 | 0.7939 | 0.5735 | 0.6202 | 0.8481 | 0.7775 | 0.9011 |
| soup | 0.6920 | 0.8007 | 0.5627 | 0.6413 | 0.8524 | 0.8129 | 0.8821 |
| soup+tta | 0.6911 | 0.8064 | 0.5654 | 0.6234 | 0.8546 | 0.8331 | 0.8707 |

**Best variant:** `soup+tta` — acc_refer **0.8546** (95% Wilson CI [0.8378, 0.8699], n=1843), QWK 0.8064
**Goal 97.00%:** not reached (gap +11.54%)

## s5-fusion — stronger backbone + highest resolution + EMA
*2026-09-28 11:33:52 UTC | overrides: `model.backbone=efficientnet_lite0 data.img_size=224 train.epochs=4 train.ema=true train.patience=6` | 3451s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.6460 | 0.8241 | 0.5871 | 0.6636 | 0.8752 | 0.8205 | 0.9163 |
| tta | 0.6386 | 0.8239 | 0.5844 | 0.6913 | 0.8828 | 0.8483 | 0.9087 |
| soup | 0.7179 | 0.8434 | 0.6517 | 0.6994 | 0.8877 | 0.8445 | 0.9202 |
| soup+tta | 0.7218 | 0.8450 | 0.6560 | 0.6907 | 0.8904 | 0.8546 | 0.9173 |

**Best variant:** `soup+tta` — acc_refer **0.8904** (95% Wilson CI [0.8753, 0.9039], n=1843), QWK 0.8450
**Goal 97.00%:** not reached (gap +7.96%)


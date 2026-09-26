## s1-baseline — budget epochs at default resolution
*2026-09-26 23:17:48 UTC | overrides: `(config defaults)` | 34s*

| variant | QWK | QWK(cuts) | acc | acc(cuts) | acc_refer | sens | spec |
|---|---|---|---|---|---|---|---|
| base | 0.0000 | 0.0000 | 0.0500 | 0.2250 | 0.6933 | 0.0000 | 1.0000 |
| tta | 0.0000 | 0.0000 | 0.0500 | 0.2250 | 0.6933 | 0.0000 | 1.0000 |
| soup | 0.0000 | 0.0011 | 0.0500 | 0.2250 | 0.6933 | 0.0054 | 0.9976 |
| soup+tta | 0.0000 | 0.0000 | 0.0500 | 0.2250 | 0.6933 | 0.0000 | 1.0000 |

**Best variant:** `soup` — acc_refer **0.6933** (95% Wilson CI [0.6553, 0.7289], n=600), QWK 0.0011
**Goal 97.00%:** not reached (gap +27.67%)


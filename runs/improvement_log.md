# RetinaEdge-DR Improvement Campaign Log

Goal: referable-DR accuracy >= 0.97 on the held-out validation split (Wilson CI lower
bound gated). Each entry below is one ladder rung executed by
`retinaedge.train.auto_improve`; state lives in `runs/improve_state.json`.

> Pre-campaign note: a local plumbing probe (600-image slice, random init, 1 epoch)
> verified the full loop (train -> TTA -> soup -> threshold search -> Wilson CI ->
> markdown log) end-to-end. Its numbers were a pipeline check, not a model result;
> the campaign state was reset before the first in-GitHub run.

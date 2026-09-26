package com.retinaedge.app.dr

import kotlin.math.exp

/**
 * Pure post-processing math. No Android dependencies — fully unit-tested.
 *
 * Mirrors `retinaedge.models.ordinal_ops` and the Android section of
 * `docs/INTERFACES.md`: grade = argmax(probs); referable = sum(probs[2..4]) >= 0.5.
 */
object DrPostprocess {

    /** Dequantize a uint8 tensor buffer: `value = (q - zeroPoint) * scale`. */
    fun dequantize(bytes: ByteArray, scale: Float, zeroPoint: Int): FloatArray =
        FloatArray(bytes.size) { i -> ((bytes[i].toInt() and 0xFF) - zeroPoint) * scale }

    /** Numerically stable logistic function. */
    fun sigmoid(x: Float): Float = when {
        x >= 0f -> 1f / (1f + exp(-x))
        else -> {
            val e = exp(x)
            e / (1f + e)
        }
    }

    /**
     * Convert K-1 cumulative ("grade > k") logits into K grade probabilities.
     *
     * Exact mirror of `retinaedge.models.ordinal_ops.ordinal_probs`:
     * cum = [1, sigmoid(g0), ..., sigmoid(gK-2), 0]; probs = first differences,
     * clamped at 0 and renormalized for numerical safety (the binary heads are
     * not guaranteed monotone).
     *
     * Used as a defensive fallback when the exported graph exposes the raw
     * ordinal head (K-1 logits) instead of the InferenceWrapper (K probs).
     */
    fun probsFromOrdinalLogits(logits: FloatArray): FloatArray {
        require(logits.isNotEmpty()) { "ordinal logits must not be empty" }
        val k = logits.size + 1
        val cum = FloatArray(k + 1)
        cum[0] = 1f
        for (i in logits.indices) cum[i + 1] = sigmoid(logits[i])
        val probs = FloatArray(k)
        var sum = 0f
        for (i in 0 until k) {
            probs[i] = (cum[i] - cum[i + 1]).coerceAtLeast(0f)
            sum += probs[i]
        }
        if (sum <= 1e-12f) {
            probs[0] = 1f
            return probs
        }
        for (i in probs.indices) probs[i] /= sum
        return probs
    }

    /** Argmax grade (first index on ties), matching torch.argmax semantics closely enough for 5 probs. */
    fun gradeArgmax(probs: FloatArray): Int {
        require(probs.isNotEmpty()) { "probs must not be empty" }
        var best = 0
        for (i in 1 until probs.size) if (probs[i] > probs[best]) best = i
        return best
    }

    /** P(referable DR) = sum of probs[referableGrade..] (contract: >= 2). */
    fun referableProb(probs: FloatArray, referableGrade: Int = DrContract.REFERABLE_GRADE): Float {
        var s = 0f
        for (i in referableGrade until probs.size) s += probs[i]
        return s
    }

    fun isReferable(referProb: Float, threshold: Float = DrContract.REFERABLE_THRESHOLD): Boolean =
        referProb >= threshold

    /**
     * Decide whether a rank-4 input shape is NCHW `(1,3,H,W)` or NHWC `(1,H,W,3)`.
     * NCHW is the documented export contract; NHWC appears with onnx2tf-style
     * conversions, so both must work.
     */
    fun isNchw(shape: IntArray): Boolean {
        require(shape.size == 4) { "expected rank-4 input, got ${shape.toList()}" }
        val cIsDim1 = shape[1] == 3
        val cIsDim3 = shape[3] == 3
        if (cIsDim1 && !cIsDim3) return true
        if (cIsDim3 && !cIsDim1) return false
        // Ambiguous (e.g. (1,3,3,3)); square spatial dims are the NCHW convention here.
        return shape[2] == shape[3]
    }

    /** Spatial (height, width) of a rank-4 input shape in the given layout. */
    fun spatialDims(shape: IntArray, nchw: Boolean): Pair<Int, Int> =
        if (nchw) shape[2] to shape[3] else shape[1] to shape[2]
}

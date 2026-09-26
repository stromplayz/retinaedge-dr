package com.retinaedge.app.dr

import android.graphics.Bitmap
import kotlin.random.Random

/**
 * Synthetic fallback used **only** when `models/dr_model.tflite` is absent, so
 * the UI can be exercised without a trained checkpoint. Predictions are
 * deterministic per image and always rendered with a DEMO banner — they are
 * not model outputs and must never be mistaken for real screening results.
 */
object DemoSynthesizer {

    /** Plausible-looking 5-class probabilities derived deterministically from [seed]. */
    fun probs(seed: Int): FloatArray {
        val rnd = Random(seed)
        // Base prior ~ class distribution of screening populations; occasionally inject referable.
        val referableCase = rnd.nextFloat() < 0.25f
        val base = if (referableCase) {
            floatArrayOf(0.10f, 0.12f, 0.34f, 0.24f, 0.20f)
        } else {
            floatArrayOf(0.62f, 0.16f, 0.10f, 0.06f, 0.06f)
        }
        val raw = FloatArray(5) { i -> base[i] * (0.6f + 0.8f * rnd.nextFloat()) }
        val sum = raw.sum()
        for (i in raw.indices) raw[i] /= sum
        return raw
    }

    fun seedFor(bitmap: Bitmap): Int {
        // Cheap, stable perceptual seed: sample a small pixel grid.
        val w = bitmap.width
        val h = bitmap.height
        var acc = 1469598103 // FNV offset basis
        var i = 0
        val samples = 64
        val step = maxOf(1, (w * h) / samples)
        while (i < w * h) {
            val x = i % w
            val y = i / w
            val p = bitmap.getPixel(x, y)
            acc = acc xor (p and 0xFF)
            acc *= 16777619
            i += step
        }
        return acc
    }

    fun predict(bitmap: Bitmap, reason: String): DrPrediction {
        val t0 = System.currentTimeMillis()
        val probs = probs(seedFor(bitmap))
        val grade = DrPostprocess.gradeArgmax(probs)
        val refer = DrPostprocess.referableProb(probs)
        return DrPrediction(
            grade = grade,
            gradeLabel = DrContract.DEFAULT_LABELS[grade],
            probs = probs.toList(),
            referProb = refer,
            referable = DrPostprocess.isReferable(refer),
            latencyMs = (System.currentTimeMillis() - t0).coerceAtLeast(1),
            demo = true,
            notes = listOf("demo mode: $reason"),
        )
    }
}

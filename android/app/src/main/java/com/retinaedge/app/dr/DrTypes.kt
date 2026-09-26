package com.retinaedge.app.dr

/**
 * Frozen constants shared with `docs/INTERFACES.md` (contract v1.0).
 *
 * Grade scheme (ICDRSS, fixed everywhere — Python, Kotlin, docs):
 *   0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
 * Referable DR = grade >= 2, decision rule P(referable) >= 0.5.
 */
object DrContract {
    const val NUM_GRADES = 5
    const val REFERABLE_GRADE = 2
    const val REFERABLE_THRESHOLD = 0.5f

    /** Labels list order must match the grade scheme. */
    val DEFAULT_LABELS = listOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")

    /** Asset path fixed by the interface contract. */
    const val MODEL_ASSET_PATH = "models/dr_model.tflite"

    /** Optional label files, tried in order; falls back to [DEFAULT_LABELS]. */
    val LABEL_ASSET_CANDIDATES = listOf("models/labels.txt", "labels.txt")

    /** Export graph normalization (ImageNet), channel order R, G, B. */
    val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
}

/** Static description of the loaded model, shown in the UI and useful for debugging. */
data class ModelInfo(
    val assetPath: String,
    val sizeBytes: Long,
    val inputShape: List<Int>,
    val inputIsUint8: Boolean,
    val outputShape: List<Int>,
    val outputIsUint8: Boolean,
    /** True for (1,3,H,W) NCHW graphs, false for (1,H,W,3) NHWC graphs. */
    val nchw: Boolean,
    /** Where the label list came from (asset path or "built-in"). */
    val labelsSource: String,
    val numThreads: Int,
    val backend: String,
)

/** Result of trying to load `models/dr_model.tflite` from assets. */
sealed interface LoadState {
    data object Idle : LoadState
    data class Ready(val info: ModelInfo) : LoadState
    /** Model missing/invalid — the app falls back to clearly-labelled demo mode. */
    data class Missing(val reason: String) : LoadState
}

/** One screening prediction. */
data class DrPrediction(
    val grade: Int,
    val gradeLabel: String,
    /** Per-grade probabilities, length 5, sums to ~1. */
    val probs: List<Float>,
    /** P(referable) = probs[2] + probs[3] + probs[4]. */
    val referProb: Float,
    val referable: Boolean,
    val latencyMs: Long,
    val demo: Boolean,
    /** Non-fatal notes, e.g. "raw ordinal head converted on-device". */
    val notes: List<String> = emptyList(),
)

package com.retinaedge.app.dr

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Loads `file:///android_asset/models/dr_model.tflite` + labels and runs
 * inference, honouring the export contract in `docs/INTERFACES.md`:
 *
 *  - input `float32 (1,3,H,W)` ImageNet-normalized  -> output `float32 (1,5)` probs
 *  - full-integer int8 graphs with uint8 in/out are quantize/dequantized on-device
 *  - both NCHW (direct conversion) and NHWC (onnx2tf) layouts are auto-detected
 *  - a raw (1,4) ordinal head is converted on-device via CORAL first differences
 */
class DrClassifier(private val context: Context) : Closeable {

    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var inputBuffer: ByteBuffer? = null
    private var outFloats: FloatArray? = null
    private var outBytes: ByteArray? = null

    var info: ModelInfo? = null
        private set
    private var nchw = true
    private var inputUint8 = false
    private var outputUint8 = false
    private var labels: List<String> = DrContract.DEFAULT_LABELS

    /** Load the asset model. Returns [LoadState.Ready] or [LoadState.Missing] (demo mode). */
    fun load(): LoadState {
        close()
        val buffer = try {
            loadModelBuffer(DrContract.MODEL_ASSET_PATH)
        } catch (t: Throwable) {
            Log.w(TAG, "model asset missing/unreadable: ${t.message}")
            return LoadState.Missing(
                "models/dr_model.tflite not found in assets (${t.message}). " +
                    "Export it with retinaedge.export.export_tflite and copy it to " +
                    "android/app/src/main/assets/models/ — see assets/README.md."
            )
        }

        val threads = maxOf(1, Runtime.getRuntime().availableProcessors() - 1).coerceAtMost(4)
        val (interp, backend, _) = try {
            buildInterpreter(buffer, false, threads)
        } catch (t: Throwable) {
            return LoadState.Missing("TFLite failed to build interpreter: ${t.message}")
        }
        interpreter = interp

        return try {
            val inShape = interp.getInputTensor(0).shape()
            val outShape = interp.getOutputTensor(0).shape()
            require(inShape.size == 4) { "expected rank-4 input, got ${inShape.toList()}" }
            val inFlat = inShape.fold(1) { a, b -> a * b }
            val outFlat = outShape.fold(1) { a, b -> a * b }
            require(outFlat == DrContract.NUM_GRADES || outFlat == DrContract.NUM_GRADES - 1) {
                "unsupported output flat size $outFlat (expected 5 probs or 4 ordinal logits)"
            }

            nchw = DrPostprocess.isNchw(inShape)
            inputUint8 = interp.getInputTensor(0).dataType() == org.tensorflow.lite.DataType.UINT8
            outputUint8 = interp.getOutputTensor(0).dataType() == org.tensorflow.lite.DataType.UINT8

            inputBuffer = ByteBuffer.allocateDirect(inFlat * if (inputUint8) 1 else 4)
                .order(ByteOrder.nativeOrder())
            if (outputUint8) outBytes = ByteArray(outFlat) else outFloats = FloatArray(outFlat)

            val (lbls, source) = loadLabels()
            labels = lbls
            require(labels.size == DrContract.NUM_GRADES) {
                "labels.txt must list exactly ${DrContract.NUM_GRADES} labels, got ${labels.size}"
            }

            val qParams = interp.getInputTensor(0).quantizationParams()
            info = ModelInfo(
                assetPath = DrContract.MODEL_ASSET_PATH,
                sizeBytes = buffer.limit().toLong(),
                inputShape = inShape.toList(),
                inputIsUint8 = inputUint8,
                outputShape = outShape.toList(),
                outputIsUint8 = outputUint8,
                nchw = nchw,
                labelsSource = source,
                numThreads = threads,
                backend = backend,
            )
            // Keep quantization params handy via info consumers re-reading the tensor;
            // classify() reads them live from the interpreter.
            Log.i(TAG, "model ready: $info (qScale=${qParams.scale}, qZero=${qParams.zeroPoint})")
            LoadState.Ready(info!!)
        } catch (t: Throwable) {
            close()
            Log.w(TAG, "model rejected: ${t.message}")
            LoadState.Missing("incompatible dr_model.tflite: ${t.message}")
        }
    }

    /**
     * Rebuild the interpreter with (or without) the NNAPI accelerator.
     * Falls back to CPU when NNAPI is unavailable or rejects the graph.
     */
    fun setUseNnapi(useNnapi: Boolean) {
        val current = info ?: return
        val threads = current.numThreads
        val newTriple = try {
            buildInterpreter(loadModelBuffer(current.assetPath), useNnapi, threads)
        } catch (t: Throwable) {
            Log.w(TAG, "interpreter rebuild failed (useNnapi=$useNnapi): ${t.message}")
            return
        }
        val (newInterp, backend, newDelegate) = newTriple
        if (useNnapi && newDelegate == null) {
            // NNAPI construction failed inside buildInterpreter and it already fell back to CPU.
            newInterp.close()
        }
        val oldInterp = interpreter
        val oldDelegate = nnApiDelegate
        interpreter = newInterp
        nnApiDelegate = newDelegate
        oldInterp?.close()
        oldDelegate?.close()
        info = current.copy(backend = backend)
    }

    /**
     * Run screening on a photo. The bitmap may be any size/orientation;
     * fundus crop + resize + normalization happen here.
     */
    fun classify(photo: Bitmap, useNnapi: Boolean): DrPrediction {
        val interp = checkNotNull(interpreter) { "classifier not loaded" }
        val t0 = SystemClock.elapsedRealtime()
        val notes = mutableListOf<String>()

        val cropped = FundusPreprocess.cropFundus(photo)
        val shape = interp.getInputTensor(0).shape()
        val (w, h) = DrPostprocess.spatialDims(shape, nchw)
        val resized = FundusPreprocess.resizeToModel(cropped, w, h)

        val inQ = interp.getInputTensor(0).quantizationParams()
        val buf = checkNotNull(inputBuffer)
        FundusPreprocess.fillInput(resized, buf, nchw, inputUint8, inQ.scale, inQ.zeroPoint)

        val raw: FloatArray
        if (outputUint8) {
            val out = checkNotNull(outBytes)
            interp.run(buf, out)
            val outQ = interp.getOutputTensor(0).quantizationParams()
            raw = DrPostprocess.dequantize(out, outQ.scale, outQ.zeroPoint)
        } else {
            val out = checkNotNull(outFloats)
            interp.run(buf, out)
            raw = out
        }

        val probs: FloatArray = when (raw.size) {
            DrContract.NUM_GRADES -> raw
            DrContract.NUM_GRADES - 1 -> {
                notes += "graph exposed the raw ordinal head (4 logits); converted on-device with CORAL first differences"
                DrPostprocess.probsFromOrdinalLogits(raw)
            }
            else -> error("unexpected output size ${raw.size}")
        }

        val grade = DrPostprocess.gradeArgmax(probs)
        val refer = DrPostprocess.referableProb(probs)
        val latency = SystemClock.elapsedRealtime() - t0
        return DrPrediction(
            grade = grade,
            gradeLabel = labels.getOrElse(grade) { DrContract.DEFAULT_LABELS[grade] },
            probs = probs.toList(),
            referProb = refer,
            referable = DrPostprocess.isReferable(refer),
            latencyMs = latency,
            demo = false,
            notes = notes,
        )
    }

    private fun buildInterpreter(
        model: ByteBuffer,
        useNnapi: Boolean,
        threads: Int,
    ): Triple<Interpreter, String, NnApiDelegate?> {
        if (useNnapi) {
            try {
                val delegate = NnApiDelegate()
                val interp = Interpreter(
                    model,
                    Interpreter.Options().setNumThreads(threads).addDelegate(delegate),
                )
                return Triple(interp, "NNAPI", delegate)
            } catch (t: Throwable) {
                Log.w(TAG, "NNAPI delegate unavailable, using CPU: ${t.message}")
            }
        }
        return Triple(
            Interpreter(model, Interpreter.Options().setNumThreads(threads)),
            "CPU (XNNPACK enabled by default)",
            null,
        )
    }

    private fun loadModelBuffer(assetPath: String): ByteBuffer =
        try {
            context.assets.openFd(assetPath).use { afd ->
                FileInputStream(afd.fileDescriptor).channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    afd.startOffset,
                    afd.declaredLength,
                )
            }
        } catch (t: Throwable) {
            // Uncompressed-assets mmap can fail on some OEM builds — fall back to a full read.
            val bytes = context.assets.open(assetPath).use { it.readBytes() }
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                put(bytes)
                rewind()
            }
        }

    /** Read labels, preferring `models/labels.txt`, then `labels.txt`, then built-in defaults. */
    private fun loadLabels(): Pair<List<String>, String> {
        for (path in DrContract.LABEL_ASSET_CANDIDATES) {
            try {
                val lines = context.assets.open(path).bufferedReader().readLines()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                if (lines.size == DrContract.NUM_GRADES) return lines to path
            } catch (_: Throwable) {
                // try next candidate
            }
        }
        return DrContract.DEFAULT_LABELS to "built-in"
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        nnApiDelegate?.close()
        nnApiDelegate = null
        inputBuffer = null
        outFloats = null
        outBytes = null
        info = null
    }

    private companion object {
        const val TAG = "DrClassifier"
    }
}

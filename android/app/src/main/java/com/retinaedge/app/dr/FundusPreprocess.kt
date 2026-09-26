package com.retinaedge.app.dr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * On-device preprocessing, per `docs/INTERFACES.md` (Android section):
 *
 *   crop fundus circle -> resize to model input -> ImageNet normalize
 *   (NO CLAHE on device: the model is trained with clahe_prob=0.5 and is
 *   robust to both) -> quantize if the input tensor is uint8.
 */
object FundusPreprocess {

    private const val MASK_MAX_DIM = 256
    private const val BRIGHTNESS_THRESHOLD = 16
    private const val MIN_MASK_COVERAGE = 0.05f
    private const val CROP_MARGIN = 1.03f
    private const val FALLBACK_SCALE = 0.92f

    /**
     * Decode a content URI to a bitmap (EXIF-rotated), downsampled to at most
     * [maxDim] on the long edge — plenty for a 224px model input and keeps
     * memory bounded on large fundus-camera exports.
     */
    fun decodeScaled(streamProvider: () -> InputStream?, maxDim: Int = 2048): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        streamProvider()?.use { BitmapFactory.decodeStream(it, null, bounds) }
        check(bounds.outWidth > 0 && bounds.outHeight > 0) { "not a decodable image" }

        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) sample *= 2

        val raw = streamProvider()?.use {
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
        } ?: error("failed to decode image")

        val rotation = runCatching {
            streamProvider()?.use { ExifInterface(it).rotationDegrees }
        }.getOrNull() ?: 0
        if (rotation == 0) return raw
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
    }

    /**
     * Crop the fundus circle. Finds the bounding box of non-black pixels on a
     * downscaled mask, expands it to a square (the fundus is roughly circular),
     * and crops with a small margin. Falls back to a centered square crop when
     * the mask is degenerate (blank/whole-frame images, non-fundus photos).
     */
    fun cropFundus(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        require(w > 0 && h > 0) { "empty bitmap" }

        val scale = min(1f, MASK_MAX_DIM.toFloat() / max(w, h))
        val mw = max(1, (w * scale).roundToInt())
        val mh = max(1, (h * scale).roundToInt())
        val pixels = IntArray(mw * mh)
        (if (scale < 1f) Bitmap.createScaledBitmap(src, mw, mh, true) else src)
            .getPixels(pixels, 0, mw, 0, 0, mw, mh)

        var minX = mw
        var minY = mh
        var maxX = -1
        var maxY = -1
        for (y in 0 until mh) {
            val row = y * mw
            for (x in 0 until mw) {
                val p = pixels[row + x]
                val lum = ((p shr 16 and 0xFF) * 299 + (p shr 8 and 0xFF) * 587 + (p and 0xFF) * 114) / 1000
                if (lum > BRIGHTNESS_THRESHOLD) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        val crop: Bitmap
        if (maxX < 0 || (maxX - minX + 1) * (maxY - minY + 1) < MIN_MASK_COVERAGE * mw * mh) {
            // Degenerate mask -> centered square crop.
            val side = (min(w, h) * FALLBACK_SCALE).roundToInt().coerceIn(1, min(w, h))
            val x = (w - side) / 2
            val y = (h - side) / 2
            crop = Bitmap.createBitmap(src, x, y, side, side)
        } else {
            // Scale bbox back to full resolution and make it square around its center.
            val bx0 = (minX / scale).toInt().coerceIn(0, w - 1)
            val bx1 = ((maxX + 1) / scale).toInt().coerceIn(bx0 + 1, w)
            val by0 = (minY / scale).toInt().coerceIn(0, h - 1)
            val by1 = ((maxY + 1) / scale).toInt().coerceIn(by0 + 1, h)
            val bw = bx1 - bx0
            val bh = by1 - by0
            val side = (max(bw, bh) * CROP_MARGIN).roundToInt()
            val cx = bx0 + bw / 2
            val cy = by0 + bh / 2
            val x = (cx - side / 2).coerceIn(0, w - min(side, w))
            val y = (cy - side / 2).coerceIn(0, h - min(side, h))
            val cw = min(side, w - x)
            val ch = min(side, h - y)
            crop = Bitmap.createBitmap(src, x, y, cw, ch)
        }
        return crop
    }

    /** Bilinear resize to the exact model input size. */
    fun resizeToModel(src: Bitmap, width: Int, height: Int): Bitmap =
        if (src.width == width && src.height == height) {
            src
        } else {
            Bitmap.createScaledBitmap(src, width, height, true)
        }

    /**
     * Fill the (direct) input byte buffer for the interpreter.
     *
     * - float32 graphs: `(v/255 - mean)/std` per channel, written NCHW or NHWC.
     * - uint8 graphs: the same normalized value quantized with the tensor's own
     *   parameters, `q = round(norm/scale) + zeroPoint` — correct regardless of
     *   which value range the conversion calibration captured.
     *
     * Channel order is R,G,B (Bitmap ARGB int layout: R = bits 16..23).
     */
    fun fillInput(
        bmp: Bitmap,
        dst: java.nio.ByteBuffer,
        nchw: Boolean,
        uint8: Boolean,
        qScale: Float,
        qZeroPoint: Int,
        mean: FloatArray = DrContract.IMAGENET_MEAN,
        std: FloatArray = DrContract.IMAGENET_STD,
    ) {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        dst.rewind()

        fun norm(c: Int, ch: Int): Float = (c / 255f - mean[ch]) / std[ch]
        fun quant(c: Int, ch: Int): Byte =
            ((norm(c, ch) / qScale).roundToInt() + qZeroPoint).coerceIn(0, 255).toByte()

        if (uint8) {
            if (nchw) {
                for (ch in 0 until 3) {
                    val shift = 16 - 8 * ch
                    for (p in pixels) dst.put(quant(p shr shift and 0xFF, ch))
                }
            } else {
                for (p in pixels) {
                    dst.put(quant(p shr 16 and 0xFF, 0))
                    dst.put(quant(p shr 8 and 0xFF, 1))
                    dst.put(quant(p and 0xFF, 2))
                }
            }
        } else {
            if (nchw) {
                for (ch in 0 until 3) {
                    val shift = 16 - 8 * ch
                    for (p in pixels) dst.putFloat(norm(p shr shift and 0xFF, ch))
                }
            } else {
                for (p in pixels) {
                    dst.putFloat(norm(p shr 16 and 0xFF, 0))
                    dst.putFloat(norm(p shr 8 and 0xFF, 1))
                    dst.putFloat(norm(p and 0xFF, 2))
                }
            }
        }
        dst.rewind()
    }
}

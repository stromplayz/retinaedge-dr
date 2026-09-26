package com.retinaedge.app.dr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * JVM unit tests for the pure post-processing math that mirrors
 * `retinaedge.models.ordinal_ops` and the Android section of the interface
 * contract.
 */
class DrPostprocessTest {

    private fun assertClose(a: Float, b: Float, eps: Float = 1e-4f) {
        assertTrue("expected $a ≈ $b", abs(a - b) < eps)
    }

    @Test
    fun `dequantize applies scale and zero point`() {
        val out = DrPostprocess.dequantize(byteArrayOf(100, 0, 255), scale = 0.01f, zeroPoint = 0)
        assertClose(1.0f, out[0])
        assertClose(0.0f, out[1])
        assertClose(2.55f, out[2])
    }

    @Test
    fun `dequantize treats bytes as unsigned with offset`() {
        // byte 200 with zeroPoint 128 -> (200 - 128) * scale
        val out = DrPostprocess.dequantize(byteArrayOf(200.toByte()), scale = 0.02f, zeroPoint = 128)
        assertClose(1.44f, out[0])
    }

    @Test
    fun `ordinal logits produce probs matching python ordinal_ops`() {
        // Reference computed with retinaedge.models.ordinal_ops.ordinal_probs
        // for logits g = [2.0, 1.0, -1.0, -3.0]:
        // cum = [1, 0.880797, 0.731059, 0.268941, 0.047426, 0]
        val probs = DrPostprocess.probsFromOrdinalLogits(floatArrayOf(2.0f, 1.0f, -1.0f, -3.0f))
        assertEquals(5, probs.size)
        assertClose(1f - 0.880797f, probs[0])
        assertClose(0.880797f - 0.731059f, probs[1])
        assertClose(0.731059f - 0.268941f, probs[2])
        assertClose(0.268941f - 0.047426f, probs[3])
        assertClose(0.047426f, probs[4])
        assertClose(1f, probs.sum(), 1e-5f)
    }

    @Test
    fun `ordinal probs handle non-monotone heads by clamping and renormalizing`() {
        val probs = DrPostprocess.probsFromOrdinalLogits(floatArrayOf(-5f, 5f, -5f, 5f))
        assertEquals(5, probs.size)
        probs.forEach { assertTrue("prob must be non-negative, got $it", it >= 0f) }
        assertClose(1f, probs.sum(), 1e-5f)
    }

    @Test
    fun `strongly negative logits collapse to grade 0`() {
        val probs = DrPostprocess.probsFromOrdinalLogits(floatArrayOf(-20f, -20f, -20f, -20f))
        assertClose(1f, probs[0], 1e-3f)
        assertEquals(0, DrPostprocess.gradeArgmax(probs))
    }

    @Test
    fun `argmax picks the largest probability`() {
        assertEquals(2, DrPostprocess.gradeArgmax(floatArrayOf(0.1f, 0.1f, 0.5f, 0.2f, 0.1f)))
        assertEquals(0, DrPostprocess.gradeArgmax(floatArrayOf(0.6f, 0.1f, 0.1f, 0.1f, 0.1f)))
        // Ties resolve to the lower grade (conservative for a referral decision).
        assertEquals(1, DrPostprocess.gradeArgmax(floatArrayOf(0.2f, 0.2f, 0.2f, 0.2f, 0.2f)))
    }

    @Test
    fun `referable prob is the sum over grades 2 to 4`() {
        val probs = floatArrayOf(0.1f, 0.2f, 0.3f, 0.25f, 0.15f)
        assertClose(0.7f, DrPostprocess.referableProb(probs))
    }

    @Test
    fun `referable threshold matches the contract at exactly 0_5`() {
        assertTrue(DrPostprocess.isReferable(0.5f))
        assertFalse(DrPostprocess.isReferable(0.4999f))
        assertTrue(DrPostprocess.isReferable(0.83f))
    }

    @Test
    fun `layout detection honours the export contract and onnx2tf graphs`() {
        // Contract: float32 (1,3,H,W) -> NCHW.
        assertTrue(DrPostprocess.isNchw(intArrayOf(1, 3, 224, 224)))
        // onnx2tf-style conversion -> NHWC.
        assertFalse(DrPostprocess.isNchw(intArrayOf(1, 224, 224, 3)))
        // Square smoke-test model (1,3,64,64) and (1,64,64,3).
        assertTrue(DrPostprocess.isNchw(intArrayOf(1, 3, 64, 64)))
        assertFalse(DrPostprocess.isNchw(intArrayOf(1, 64, 64, 3)))
    }

    @Test
    fun `spatial dims are read from the correct axes`() {
        assertEquals(224 to 224, DrPostprocess.spatialDims(intArrayOf(1, 3, 224, 224), nchw = true))
        assertEquals(192 to 256, DrPostprocess.spatialDims(intArrayOf(1, 192, 256, 3), nchw = false))
    }

    @Test
    fun `sigmoid is numerically stable in both tails`() {
        assertClose(1f, DrPostprocess.sigmoid(50f), 1e-6f)
        assertClose(0f, DrPostprocess.sigmoid(-50f), 1e-6f)
        assertClose(0.5f, DrPostprocess.sigmoid(0f))
        // golden value
        assertClose(0.880797f, DrPostprocess.sigmoid(2f))
    }

    @Test
    fun `probs from ordinal logits never exceed unit sum`() {
        repeat(50) { k ->
            val logits = FloatArray(4) { i -> sqrt((i + k).toFloat()) * (if ((i + k) % 2 == 0) 1f else -1f) }
            val probs = DrPostprocess.probsFromOrdinalLogits(logits)
            assertClose(1f, probs.sum(), 1e-5f)
        }
    }
}

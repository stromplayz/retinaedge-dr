package com.retinaedge.app.dr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoSynthesizerTest {

    @Test
    fun `probs are deterministic per seed`() {
        val a = DemoSynthesizer.probs(12345)
        val b = DemoSynthesizer.probs(12345)
        assertTrue(a.contentEquals(b))
    }

    @Test
    fun `probs form a normalized distribution`() {
        repeat(200) { seed ->
            val probs = DemoSynthesizer.probs(seed)
            assertEquals(5, probs.size)
            assertEquals(1f, probs.sum(), 1e-4f)
            probs.forEach { assertTrue("prob $it out of range", it in 0f..1f) }
        }
    }

    @Test
    fun `probs differ across seeds`() {
        val a = DemoSynthesizer.probs(1).toList()
        val b = DemoSynthesizer.probs(2).toList()
        assertTrue(a != b)
    }
}

class DrContractTest {

    @Test
    fun `label list matches the frozen ICDRSS grade scheme`() {
        assertEquals(
            listOf("No DR", "Mild", "Moderate", "Severe", "Proliferative DR"),
            DrContract.DEFAULT_LABELS,
        )
        assertEquals(5, DrContract.NUM_GRADES)
        assertEquals(2, DrContract.REFERABLE_GRADE)
        assertEquals(0.5f, DrContract.REFERABLE_THRESHOLD)
        assertEquals("models/dr_model.tflite", DrContract.MODEL_ASSET_PATH)
    }
}

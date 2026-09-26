package com.retinaedge.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retinaedge.app.dr.DrContract
import com.retinaedge.app.dr.DrPrediction
import com.retinaedge.app.dr.LoadState
import java.util.Locale

private fun pct(v: Float): String = String.format(Locale.US, "%.1f%%", v * 100f)

private fun bytes(size: Long): String = when {
    size >= 1 shl 20 -> String.format(Locale.US, "%.1f MB", size / 1048576f)
    size >= 1 shl 10 -> String.format(Locale.US, "%.0f KB", size / 1024f)
    else -> "$size B"
}

@Composable
fun DemoBanner(reason: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.errorContainer,
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                "DEMO MODE — no model loaded",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                reason,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
        }
    }
}

@Composable
fun ModelStatusCard(
    state: LoadState,
    backendOverride: String? = null,
    useNnapi: Boolean,
    onNnapiChanged: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Model", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            when (state) {
                LoadState.Idle -> Text("Loading ${DrContract.MODEL_ASSET_PATH} …", style = MaterialTheme.typography.bodyMedium)
                is LoadState.Missing -> Text(
                    "Not loaded — running in demo mode.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
                is LoadState.Ready -> {
                    val info = state.info
                    InfoRow("Asset", info.assetPath)
                    InfoRow("Size", bytes(info.sizeBytes))
                    InfoRow(
                        "Input",
                        "${info.inputShape.joinToString("x")} · " +
                            (if (info.nchw) "NCHW" else "NHWC") + " · " +
                            (if (info.inputIsUint8) "uint8 (quantized)" else "float32"),
                    )
                    InfoRow(
                        "Output",
                        "${info.outputShape.joinToString("x")} · " +
                            (if (info.outputIsUint8) "uint8 (dequantized)" else "float32"),
                    )
                    InfoRow("Backend", (backendOverride ?: info.backend) + " · ${info.numThreads} threads")
                    InfoRow("Labels", info.labelsSource)
                }
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("NNAPI hardware accelerator", style = MaterialTheme.typography.bodyMedium)
                Switch(checked = useNnapi, onCheckedChange = onNnapiChanged)
            }
            Text(
                "Takes effect on the next analysis; falls back to CPU when unsupported.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun InfoRow(label: String, value: String, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(
            value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(start = 16.dp),
        )
    }
}

@Composable
fun ResultCard(prediction: DrPrediction, modifier: Modifier = Modifier) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (prediction.demo) {
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Text(
                        "SYNTHETIC DEMO OUTPUT — not a model prediction",
                        Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(
                    color = gradeColor(prediction.grade),
                    contentColor = Color.White,
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            "${prediction.grade}",
                            style = MaterialTheme.typography.headlineLarge,
                            fontWeight = FontWeight.Black,
                        )
                        Text("grade", style = MaterialTheme.typography.labelSmall)
                    }
                }
                Spacer(Modifier.height(0.dp))
                Column(Modifier.padding(start = 14.dp)) {
                    Text(
                        prediction.gradeLabel,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = gradeColor(prediction.grade),
                    )
                    Text(
                        if (prediction.referable) "Referable DR — referral indicated"
                        else "No referral indicated",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (prediction.referable) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.primary,
                        fontWeight = if (prediction.referable) FontWeight.Bold else FontWeight.Normal,
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                val top = prediction.probs.indices.maxByOrNull { prediction.probs[it] } ?: 0
                prediction.probs.forEachIndexed { i, p ->
                    ProbBar(i, p, top == i)
                }
            }

            Text(
                "P(referable DR) = " + pct(prediction.referProb) +
                    "  ·  threshold " + pct(DrContract.REFERABLE_THRESHOLD),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
            )

            prediction.notes.forEach {
                Text(
                    "• $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.tertiary,
                )
            }
        }
    }
}

@Composable
fun ProbBar(grade: Int, prob: Float, isTop: Boolean) {
    Column {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                "$grade · ${DrContract.DEFAULT_LABELS[grade]}",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal,
            )
            Text(pct(prob), style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
        }
        Spacer(Modifier.height(2.dp))
        val trackColor = MaterialTheme.colorScheme.surfaceVariant
        androidx.compose.foundation.layout.Box(
            Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(trackColor),
        ) {
            androidx.compose.foundation.layout.Box(
                Modifier
                    .fillMaxWidth(prob.coerceIn(0f, 1f))
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(gradeColor(grade)),
            )
        }
    }
}

@Composable
fun DisclaimerCard(modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(
                "Not a medical device",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "RetinaEdge-DR is a research prototype for offline diabetic retinopathy " +
                    "screening. Predictions are not a diagnosis and must never be used as the " +
                    "sole basis for a clinical decision. Referable cases require confirmation " +
                    "by a qualified eye-care professional.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

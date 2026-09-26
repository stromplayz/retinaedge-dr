package com.retinaedge.app

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retinaedge.app.dr.DemoSynthesizer
import com.retinaedge.app.dr.DrClassifier
import com.retinaedge.app.dr.DrPrediction
import com.retinaedge.app.dr.FundusPreprocess
import com.retinaedge.app.dr.LoadState
import com.retinaedge.app.ui.DemoBanner
import com.retinaedge.app.ui.DisclaimerCard
import com.retinaedge.app.ui.ModelStatusCard
import com.retinaedge.app.ui.ResultCard
import com.retinaedge.app.ui.RetinaEdgeTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Single-screen screening demo: import a fundus photo via the system Photo
 * Picker, run the TFLite graph fully offline, show ICDRSS grade 0-4,
 * P(referable DR) and per-grade probabilities.
 */
class MainActivity : ComponentActivity() {

    private val classifier: DrClassifier by lazy { DrClassifier(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RetinaEdgeTheme {
                ScreeningScreen(classifier)
            }
        }
    }

    override fun onDestroy() {
        classifier.close()
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ScreeningScreen(classifier: DrClassifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var loadState by remember { mutableStateOf<LoadState>(LoadState.Idle) }
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var prediction by remember { mutableStateOf<DrPrediction?>(null) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var useNnapi by remember { mutableStateOf(false) }
    var backend by remember { mutableStateOf<String?>(null) }
    var rerunTick by remember { mutableIntStateOf(0) }

    // Load the asset model once on first composition.
    LaunchedEffect(Unit) {
        loadState = withContext(Dispatchers.IO) { classifier.load() }
        backend = classifier.info?.backend
    }

    fun analyze() {
        val uri = imageUri ?: return
        if (loadState is LoadState.Idle) return
        scope.launch {
            busy = true
            error = null
            try {
                if (loadState is LoadState.Ready) {
                    withContext(Dispatchers.IO) { classifier.setUseNnapi(useNnapi) }
                    backend = classifier.info?.backend
                }
                val bmp = withContext(Dispatchers.IO) {
                    FundusPreprocess.decodeScaled { context.contentResolver.openInputStream(uri) }
                }
                bitmap = bmp
                prediction = withContext(Dispatchers.Default) {
                    when (val st = loadState) {
                        is LoadState.Ready -> classifier.classify(bmp, useNnapi)
                        is LoadState.Missing -> DemoSynthesizer.predict(bmp, st.reason)
                        LoadState.Idle -> null
                    }
                }
            } catch (t: Throwable) {
                Log.e("ScreeningScreen", "analysis failed", t)
                error = t.message ?: t.javaClass.simpleName
            } finally {
                busy = false
            }
        }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        if (uri != null) {
            imageUri = uri
            bitmap = null
            prediction = null
            error = null
            // Auto-run for the newly picked image once the model has finished loading.
            scope.launch {
                while (loadState is LoadState.Idle) delay(50)
                rerunTick++
            }
        }
    }

    // (Re-)run analysis when a new image arrives, the accelerator toggles,
    // a re-run is requested, or the initial load completes with an image pending.
    LaunchedEffect(imageUri, useNnapi, rerunTick, loadState !is LoadState.Idle) {
        if (imageUri != null && loadState !is LoadState.Idle) analyze()
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("RetinaEdge-DR", fontWeight = FontWeight.Bold)
                        Text(
                            "Offline diabetic retinopathy screening · research prototype",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            )
        },
    ) { inner ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(inner)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            val missing = loadState as? LoadState.Missing
            if (missing != null) DemoBanner(missing.reason)

            ModelStatusCard(
                state = loadState,
                backendOverride = backend,
                useNnapi = useNnapi,
                onNnapiChanged = { on ->
                    useNnapi = on
                    if (bitmap != null) rerunTick++
                },
            )

            Button(
                onClick = {
                    picker.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (imageUri == null) "Import fundus image" else "Import another image")
            }

            bitmap?.let { bmp ->
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Selected fundus image",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 260.dp)
                        .clip(RoundedCornerShape(14.dp)),
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = { rerunTick++ }) { Text("Re-run analysis") }
                }
            }

            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())

            error?.let { err ->
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text(
                            "Analysis failed",
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                        Text(
                            err,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                    }
                }
            }

            prediction?.let { pred ->
                ResultCard(prediction = pred)
                Text(
                    "Latency ${pred.latencyMs} ms · backend ${backend ?: "n/a"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                )
            }

            DisclaimerCard()
            Spacer(Modifier.height(8.dp))
        }
    }
}

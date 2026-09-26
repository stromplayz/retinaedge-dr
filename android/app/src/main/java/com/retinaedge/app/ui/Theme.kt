package com.retinaedge.app.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Brand palette — clinical teal + ICDRSS grade ramp.
private val TealPrimary = Color(0xFF00696D)
private val TealOnPrimary = Color(0xFFFFFFFF)
private val TealContainer = Color(0xFFB4EBEC)
private val TealOnContainer = Color(0xFF002022)
private val DarkTealPrimary = Color(0xFF54DBDE)
private val DarkTealOnPrimary = Color(0xFF003739)
private val DarkTealContainer = Color(0xFF004F52)
private val DarkTealOnContainer = Color(0xFFB4EBEC)

private val LightColors = lightColorScheme(
    primary = TealPrimary,
    onPrimary = TealOnPrimary,
    primaryContainer = TealContainer,
    onPrimaryContainer = TealOnContainer,
    secondary = Color(0xFF4A6365),
    secondaryContainer = Color(0xFFCCE8E9),
    onSecondaryContainer = Color(0xFF051F21),
    background = Color(0xFFFAFDFC),
    surface = Color(0xFFFAFDFC),
    surfaceVariant = Color(0xFFDAE4E5),
    onSurfaceVariant = Color(0xFF3F4949),
    error = Color(0xFFBA1A1A),
)

private val DarkColors = darkColorScheme(
    primary = DarkTealPrimary,
    onPrimary = DarkTealOnPrimary,
    primaryContainer = DarkTealContainer,
    onPrimaryContainer = DarkTealOnContainer,
    secondary = Color(0xFFB1CCCE),
    secondaryContainer = Color(0xFF324B4C),
    onSecondaryContainer = Color(0xFFCCE8E9),
    background = Color(0xFF191C1D),
    surface = Color(0xFF191C1D),
    surfaceVariant = Color(0xFF3F4949),
    onSurfaceVariant = Color(0xFFBEC8C9),
    error = Color(0xFFFFB4AB),
)

/** ICDRSS grade -> colour ramp: green (0) to red (4). */
fun gradeColor(grade: Int): Color = when (grade) {
    0 -> Color(0xFF2E7D32)
    1 -> Color(0xFF9E9D24)
    2 -> Color(0xFFEF6C00)
    3 -> Color(0xFFE64A19)
    else -> Color(0xFFC62828)
}

@Composable
fun RetinaEdgeTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        content = content,
    )
}

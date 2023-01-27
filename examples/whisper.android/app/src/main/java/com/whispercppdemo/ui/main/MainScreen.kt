package com.whispercppdemo.ui.main

import android.widget.Spinner
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.unit.dp
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.whispercppdemo.R
import java.util.*

@Composable
fun MainScreen(viewModel: MainScreenViewModel) {
    MainScreen(
        canTranscribe = viewModel.canTranscribe,
        isRecording = viewModel.isRecording,
        messageLog = viewModel.dataLog,
        languages = viewModel.languages,
        language = viewModel.language,
        onLanguageChange = {
            viewModel.language = it
        },
        onTranscribeSampleTapped = viewModel::transcribeSample,
        onRecordTapped = viewModel::toggleRecord,
        //languages = viewModel.languages,
        )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MainScreen(
    canTranscribe: Boolean,
    isRecording: Boolean,
    messageLog: String,
    languages: List<String>,
    language: String,
    onLanguageChange: (String) -> Unit,
    onTranscribeSampleTapped: () -> Unit,
    onRecordTapped: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) }
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .padding(16.dp)
        ) {
            Row(horizontalArrangement = Arrangement.SpaceBetween) {
                LanguageSelector(false, languages, language,
                            onLanguageChange = onLanguageChange)
                RecordButton(
                         enabled = canTranscribe,
                         isRecording = isRecording,
                         onClick = onRecordTapped
                )
            }
            MessageLog(messageLog)
        }
    }
}

@Composable
private fun MessageLog(log: String) {
    Text(modifier = Modifier.verticalScroll(rememberScrollState()), text = log)
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
private fun RecordButton(enabled: Boolean, isRecording: Boolean, onClick: () -> Unit) {
    val micPermissionState = rememberPermissionState(
        permission = android.Manifest.permission.RECORD_AUDIO,
        onPermissionResult = { granted ->
            if (granted) {
                onClick()
            }
        }
    )
    Button(onClick = {
        if (micPermissionState.status.isGranted) {
            onClick()
        } else {
            micPermissionState.launchPermissionRequest()
        }
     }, enabled = enabled) {
        Text(
            if (isRecording) {
                "Stop recording"
            } else {
                "Start recording"
            }
        )
    }
}

@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun LanguageSelector(expanded: Boolean, items: List<String>, language: String, onLanguageChange: (String) -> Unit) {
    var expanded by remember { mutableStateOf(expanded) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = {
            expanded = !expanded
        }
    ) {
        TextField(
            // The `menuAnchor` modifier must be passed to the text field for correctness.
            modifier = Modifier.menuAnchor(),
            readOnly = true,
            value = language,
            onValueChange = {
                onLanguageChange(it)
            },
            label = { Text("Language") },
            trailingIcon = {
                ExposedDropdownMenuDefaults.TrailingIcon(
                    expanded = expanded
                )
            },
            colors = ExposedDropdownMenuDefaults.textFieldColors()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = {
                expanded = false
            }
        ) {
            items.forEach { code ->
                DropdownMenuItem(
                    text = {
                        Text(text = Locale(code).displayLanguage ?: code)
                    },
                    onClick = {
                        onLanguageChange(code)
                        expanded = false
                    },
                    contentPadding = ExposedDropdownMenuDefaults.ItemContentPadding,
                )
            }
        }
    }
}
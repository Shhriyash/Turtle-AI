/**
 * voice.js — Audio recording (AudioWorklet) and playback (AudioContext)
 *
 * The central bubble is the push-to-talk trigger.
 * Recording uses AudioWorklet for raw PCM16 at 16 kHz.
 */

import AppState from './state.js';
import { setStatus, showToast } from './utils.js';
import { setBubbleState } from './chat.js';

const PROCESSOR_PATH = '/static/audio/pcm-processor.js';
const AUTO_VAD_RMS_THRESHOLD = 350;
const AUTO_VAD_MIN_SPEECH_MS = 180;
const AUTO_VAD_SILENCE_STOP_MS = 850;
const AUTO_VAD_NO_SPEECH_TIMEOUT_MS = 4500;
const AUTO_VAD_MAX_RECORD_MS = 15000;

let autoVadSpeechMs = 0;
let autoVadSilenceMs = 0;
let autoVadTotalMs = 0;
let autoVadSpeechStarted = false;

function resetAutoVadState() {
    autoVadSpeechMs = 0;
    autoVadSilenceMs = 0;
    autoVadTotalMs = 0;
    autoVadSpeechStarted = false;
}

function chunkRms(chunk) {
    if (!chunk || chunk.length === 0) return 0;
    let sumSquares = 0;
    for (let i = 0; i < chunk.length; i++) {
        const s = chunk[i];
        sumSquares += s * s;
    }
    return Math.sqrt(sumSquares / chunk.length);
}

function handleAutoVadChunk(chunk) {
    if (AppState.voiceMode !== 'vad' || !AppState.isRecording) return;

    const chunkMs = (chunk.length / 16000) * 1000;
    autoVadTotalMs += chunkMs;

    const rms = chunkRms(chunk);
    if (rms >= AUTO_VAD_RMS_THRESHOLD) {
        autoVadSpeechMs += chunkMs;
        autoVadSilenceMs = 0;
        if (autoVadSpeechMs >= AUTO_VAD_MIN_SPEECH_MS) {
            autoVadSpeechStarted = true;
        }
    } else if (autoVadSpeechStarted) {
        autoVadSilenceMs += chunkMs;
    }

    if (!autoVadSpeechStarted && autoVadTotalMs >= AUTO_VAD_NO_SPEECH_TIMEOUT_MS) {
        stopRecording({ discard: true });
        showToast('No speech detected', true);
        return;
    }

    if (autoVadSpeechStarted && autoVadSilenceMs >= AUTO_VAD_SILENCE_STOP_MS) {
        stopRecording();
        return;
    }

    if (autoVadTotalMs >= AUTO_VAD_MAX_RECORD_MS) {
        stopRecording();
        showToast('Auto-stopped after 15s');
    }
}

function setVoiceButtonRecordingUi(isRecording) {
    const btn = AppState.dom.btnVoice;
    if (!btn) return;
    btn.classList.toggle('recording', isRecording);
    btn.setAttribute('aria-pressed', isRecording ? 'true' : 'false');
    if (isRecording) {
        btn.title = AppState.voiceMode === 'ptt'
            ? 'Release to send voice input'
            : 'Stop and send voice input';
    } else {
        btn.title = AppState.voiceMode === 'ptt'
            ? 'Hold to talk (or hold Space)'
            : 'Start auto voice detection';
    }
}

export function refreshVoiceButtonUi() {
    setVoiceButtonRecordingUi(AppState.isRecording);
}


function stopTtsPlayback({ resetUi = false } = {}) {
    const source = AppState.ttsSourceNode;
    const ctx = AppState.ttsAudioContext;

    AppState.ttsSourceNode = null;
    AppState.ttsAudioContext = null;
    AppState.isTtsPlaying = false;

    if (source) {
        try {
            source.onended = null;
            source.stop(0);
        } catch (_) {}
    }
    if (ctx) {
        try {
            ctx.close();
        } catch (_) {}
    }

    if (resetUi && !AppState.isRecording) {
        setBubbleState('idle');
        setStatus('ready', 'Ready');
    }
}

/**
 * Start recording from the microphone.
 */
export async function startRecording() {
    if (AppState.isRecording) return;
    if (!AppState.isConnected) {
        showToast('Not connected to server — reconnecting…', true);
        return;
    }

    // Bararge-in behavior: pressing mic while TTS is speaking should interrupt playback.
    if (AppState.isTtsPlaying || AppState.ttsSourceNode || AppState.ttsAudioContext) {
        stopTtsPlayback();
    }

    AppState.isRecording = true;
    AppState.recordedChunks = [];
    resetAutoVadState();
    setBubbleState('listening');
    setStatus('recording', AppState.voiceMode === 'vad' ? 'Listening (Auto VAD)' : 'Recording');
    setVoiceButtonRecordingUi(true);

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            },
        });

        AppState.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000,
        });

        await AppState.audioContext.audioWorklet.addModule(PROCESSOR_PATH);

        const source = AppState.audioContext.createMediaStreamSource(stream);
        AppState.audioWorkletNode = new AudioWorkletNode(AppState.audioContext, 'pcm-processor');

        AppState.audioWorkletNode.port.onmessage = (event) => {
            const chunk = new Int16Array(event.data);
            AppState.recordedChunks.push(chunk);
            handleAutoVadChunk(chunk);
        };

        source.connect(AppState.audioWorkletNode);
        AppState.audioWorkletNode.connect(AppState.audioContext.destination);
        AppState.audioWorkletNode._stream = stream;

    } catch (err) {
        console.error('Recording failed:', err);
        showToast('Microphone access denied', true);
        AppState.isRecording = false;
        setBubbleState('idle');
        setStatus('ready', 'Ready');
        setVoiceButtonRecordingUi(false);
        resetAutoVadState();
    }
}

/**
 * Stop recording, merge chunks, send as binary WebSocket frame.
 */
export function stopRecording(options = {}) {
    if (!AppState.isRecording) return;
    const { discard = false } = options;
    AppState.isRecording = false;
    setVoiceButtonRecordingUi(false);
    resetAutoVadState();

    try {
        if (AppState.audioWorkletNode) {
            if (AppState.audioWorkletNode._stream) {
                AppState.audioWorkletNode._stream.getTracks().forEach(t => t.stop());
            }
            AppState.audioWorkletNode.disconnect();
            AppState.audioWorkletNode = null;
        }
        if (AppState.audioContext) {
            AppState.audioContext.close();
            AppState.audioContext = null;
        }

        if (!discard && AppState.recordedChunks.length > 0) {
            const totalLength = AppState.recordedChunks.reduce((sum, c) => sum + c.length, 0);
            const merged = new Int16Array(totalLength);
            let offset = 0;
            for (const chunk of AppState.recordedChunks) {
                merged.set(chunk, offset);
                offset += chunk.length;
            }

            if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
                AppState.ws.send(merged.buffer);
                setBubbleState('transcribing');
                setStatus('transcribing', 'Transcribing');
            } else {
                setBubbleState('idle');
                setStatus('ready', 'Ready');
            }
        } else {
            setBubbleState('idle');
            setStatus('ready', 'Ready');
        }

        AppState.recordedChunks = [];

    } catch (err) {
        console.error('Stop recording error:', err);
        setBubbleState('idle');
        setStatus('ready', 'Ready');
        setVoiceButtonRecordingUi(false);
    }
}

/**
 * Play a WAV audio blob from the server.
 */
export async function playAudioBlob(blob) {
    // Ignore late TTS blobs while user is already recording.
    if (AppState.isRecording) {
        return;
    }

    // Prevent overlap when back-to-back TTS blobs arrive.
    if (AppState.isTtsPlaying || AppState.ttsSourceNode || AppState.ttsAudioContext) {
        stopTtsPlayback();
    }

    try {
        const arrayBuffer = await blob.arrayBuffer();
        if (AppState.isRecording) return;

        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        if (AppState.isRecording) {
            await ctx.close();
            return;
        }

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);

        AppState.ttsAudioContext = ctx;
        AppState.ttsSourceNode = source;
        AppState.isTtsPlaying = true;

        source.start(0);
        source.onended = () => {
            if (AppState.ttsSourceNode === source) {
                AppState.ttsSourceNode = null;
                AppState.ttsAudioContext = null;
                AppState.isTtsPlaying = false;
                try {
                    ctx.close();
                } catch (_) {}
                if (!AppState.isRecording) {
                    setBubbleState('idle');
                    setStatus('ready', 'Ready');
                }
            }
        };
    } catch (e) {
        console.error('Audio playback error:', e);
        stopTtsPlayback({ resetUi: true });
    }
}

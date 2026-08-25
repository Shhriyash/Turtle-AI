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


/**
 * Get (or lazily create) the single persistent playback AudioContext.
 * Reused across chunks and turns so we pay context-creation cost once, not per
 * blob, and so chunks can be scheduled back-to-back on one clock.
 */
function getPlaybackContext() {
    let ctx = AppState.ttsAudioContext;
    if (!ctx || ctx.state === 'closed') {
        ctx = new (window.AudioContext || window.webkitAudioContext)();
        AppState.ttsAudioContext = ctx;
        AppState.ttsNextStartTime = 0;
    }
    // Browsers may suspend contexts created outside a user gesture; resume is a
    // no-op when already running.
    if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
    }
    return ctx;
}

/**
 * Stop all TTS playback: cancel every scheduled/playing source and flush the
 * decode queue. Used for barge-in (mic press / new recording) and errors.
 *
 * The persistent context is kept alive (just reset) so the next reply starts
 * without re-creating it — unless closeContext is requested.
 */
function stopTtsPlayback({ resetUi = false, closeContext = false } = {}) {
    // Invalidate any in-flight decodes so their .then() schedulers no-op.
    AppState.ttsPlaybackGen++;
    AppState.ttsDecodeChain = Promise.resolve();

    const sources = AppState.ttsSources || [];
    AppState.ttsSources = [];
    for (const source of sources) {
        try {
            source.onended = null;
            source.stop(0);
            source.disconnect();
        } catch (_) {}
    }

    AppState.isTtsPlaying = false;
    AppState.ttsNextStartTime = 0;

    if (closeContext && AppState.ttsAudioContext) {
        try { AppState.ttsAudioContext.close(); } catch (_) {}
        AppState.ttsAudioContext = null;
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

    // Barge-in behavior: pressing mic while TTS is speaking should interrupt
    // playback — cancel every queued/playing chunk and flush pending decodes.
    if (AppState.isTtsPlaying || (AppState.ttsSources && AppState.ttsSources.length)) {
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
 * Enqueue one WAV audio chunk from the server for gapless playback.
 *
 * The server streams one complete, self-describing WAV per sentence. Chunks are
 * decoded and scheduled back-to-back on a single persistent AudioContext so a
 * multi-sentence reply plays as one continuous utterance. Previously each new
 * blob stopped the one before it, truncating every sentence but the last — this
 * queue fixes that.
 *
 * Ordering: decode is async, so blobs are threaded through a promise chain
 * (ttsDecodeChain) to guarantee they schedule in arrival order regardless of how
 * long any single decode takes.
 */
export async function playAudioBlob(blob) {
    // Ignore late TTS blobs while the user is already recording (barge-in).
    if (AppState.isRecording) {
        return;
    }

    // Capture the playback generation this chunk belongs to. A barge-in bumps
    // the generation; any decode that finishes after that is discarded.
    const gen = AppState.ttsPlaybackGen;

    AppState.ttsDecodeChain = AppState.ttsDecodeChain
        .catch(() => {})
        .then(async () => {
            if (AppState.isRecording || gen !== AppState.ttsPlaybackGen) return;

            const arrayBuffer = await blob.arrayBuffer();
            if (AppState.isRecording || gen !== AppState.ttsPlaybackGen) return;

            const ctx = getPlaybackContext();
            // decodeAudioData detaches the ArrayBuffer; slice() keeps callers safe.
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
            if (AppState.isRecording || gen !== AppState.ttsPlaybackGen) return;

            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            // Schedule immediately after whatever is already queued. A small lead
            // (0.04s) absorbs decode jitter so the first chunk never starts in the
            // past. Subsequent chunks chain from ttsNextStartTime for zero gap.
            const now = ctx.currentTime;
            const startAt = Math.max(now + 0.04, AppState.ttsNextStartTime || 0);
            source.start(startAt);
            AppState.ttsNextStartTime = startAt + audioBuffer.duration;

            AppState.ttsSources.push(source);
            AppState.isTtsPlaying = true;
            setBubbleState('speaking');

            source.onended = () => {
                const idx = AppState.ttsSources.indexOf(source);
                if (idx !== -1) AppState.ttsSources.splice(idx, 1);
                try { source.disconnect(); } catch (_) {}
                // When the queue drains and nothing new is scheduled, go idle.
                if (
                    gen === AppState.ttsPlaybackGen &&
                    AppState.ttsSources.length === 0 &&
                    !AppState.isRecording
                ) {
                    AppState.isTtsPlaying = false;
                    AppState.ttsNextStartTime = 0;
                    setBubbleState('idle');
                    setStatus('ready', 'Ready');
                }
            };
        })
        .catch((e) => {
            console.error('Audio playback error:', e);
        });
}

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

/**
 * Start recording from the microphone.
 */
export async function startRecording() {
    if (AppState.isRecording || !AppState.isConnected) return;
    AppState.isRecording = true;
    AppState.recordedChunks = [];
    setBubbleState('listening');
    setStatus('recording', 'Recording');

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
            AppState.recordedChunks.push(new Int16Array(event.data));
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
    }
}

/**
 * Stop recording, merge chunks, send as binary WebSocket frame.
 */
export function stopRecording() {
    if (!AppState.isRecording) return;
    AppState.isRecording = false;

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

        if (AppState.recordedChunks.length > 0) {
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
    }
}

/**
 * Play a WAV audio blob from the server.
 */
export async function playAudioBlob(blob) {
    try {
        const arrayBuffer = await blob.arrayBuffer();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.start(0);
        source.onended = () => {
            ctx.close();
            setBubbleState('idle');
            setStatus('ready', 'Ready');
        };
    } catch (e) {
        console.error('Audio playback error:', e);
        setBubbleState('idle');
        setStatus('ready', 'Ready');
    }
}

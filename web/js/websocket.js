/**
 * websocket.js — WebSocket connection, reconnect, message dispatch
 */

import AppState from './state.js';
import { setStatus, showBanner, hideBanner, showToast } from './utils.js';
import { addMessage, showThinking, hideThinking, setBubbleState } from './chat.js';
import { playAudioBlob } from './voice.js';
import { updateTimings } from './devmode.js';
import { renderConfirmationPrompt } from './memory.js';

/** Connect (or reconnect) to the WebSocket server */
export function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;

    if (AppState.ws) {
        try { AppState.ws.close(); } catch (_) {}
    }

    AppState.ws = new WebSocket(wsUrl);

    AppState.ws.onopen = () => {
        AppState.isConnected = true;
        setStatus('ready', 'Ready');
        setBubbleState('idle');
        hideBanner();
        showToast('Connected to Turtle AI');
    };

    AppState.ws.onclose = () => {
        AppState.isConnected = false;
        setStatus('disconnected', 'Disconnected');
        setBubbleState('idle');
        showBanner();
    };

    AppState.ws.onerror = () => {
        AppState.isConnected = false;
        setStatus('disconnected', 'Connection error');
        showBanner();
    };

    AppState.ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
            playAudioBlob(event.data);
            return;
        }
        try {
            handleServerMessage(JSON.parse(event.data));
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case 'status':
            handleStatusMessage(msg);
            break;
        case 'transcription_partial':
            // Live interim transcript from streaming STT — show it as the
            // thinking caption while the user is still speaking.
            if (msg.text) showThinking(msg.text);
            break;
        case 'transcription':
            addMessage('user', msg.text);
            break;
        case 'done':
            hideThinking();
            addMessage('assistant', msg.content);
            setStatus('ready', 'Ready');
            setBubbleState('idle');
            break;
        case 'timing':
            updateTimings(msg);
            break;
        case 'confirmation_prompt':
            // Server queued an uncertain memory fact behind the gate.
            // Surface it as an inline card the user can confirm/dismiss.
            renderConfirmationPrompt(msg);
            break;
        case 'error':
            hideThinking();
            setStatus('ready', 'Ready');
            setBubbleState('idle');
            showToast(msg.message, true);
            break;
        case 'notice':
            // Non-fatal server notice (e.g. storage_cap: memory writes are
            // failing). Surface as an error-styled toast so the user knows.
            showToast(msg.message, true);
            break;
        case 'routine':
            // A scheduled routine fired (Phase 5 / W2). Informational, not an
            // error — showToast without the error flag = accent-styled toast.
            showToast(msg.message);
            break;
        case 'pong':
            break;
        default:
            console.log('Unknown:', msg);
    }
}

function handleStatusMessage(msg) {
    const labelMap = {
        ready:        'Ready',
        thinking:     'Thinking',
        transcribing: 'Transcribing',
        listening:    'Listening',
        speaking:     'Speaking',
        restored:     'Session restored',
    };

    // The ready frame advertises whether the server has streaming STT enabled.
    if (msg.status === 'ready') {
        AppState.streamSttEnabled = !!msg.stream_stt;
    }

    // 'listening' is a streaming-STT state; map its bubble to the recording look.
    setStatus(msg.status, labelMap[msg.status] || msg.status);
    setBubbleState(msg.status === 'listening' ? 'listening' : msg.status);

    if (msg.status === 'thinking') {
        showThinking('Thinking');
    } else if (msg.status === 'transcribing') {
        showThinking('Transcribing');
    } else if (msg.status === 'speaking') {
        showThinking('Speaking');
    } else if (msg.status === 'ready' || msg.status === 'restored') {
        hideThinking();
    }
}

export function startConnectionWatchdog() {
    setInterval(() => {
        if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
            AppState.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);

    setInterval(() => {
        if (!AppState.isConnected) connectWebSocket();
    }, 5000);
}

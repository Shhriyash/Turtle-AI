/**
 * websocket.js — WebSocket connection, reconnect, message dispatch
 */

import AppState from './state.js';
import { setStatus, showBanner, hideBanner, showToast } from './utils.js';
import { addMessage, showThinking, hideThinking, setBubbleState } from './chat.js';
import { playAudioBlob } from './voice.js';
import { updateTimings } from './devmode.js';

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
        case 'error':
            hideThinking();
            setStatus('ready', 'Ready');
            setBubbleState('idle');
            showToast(msg.message, true);
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
        speaking:     'Speaking',
        restored:     'Session restored',
    };

    setStatus(msg.status, labelMap[msg.status] || msg.status);
    setBubbleState(msg.status);

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

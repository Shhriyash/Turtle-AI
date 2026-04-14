/**
 * websocket.js — WebSocket connection, reconnect, message dispatch
 *
 * Handles connecting, auto-reconnect, keep-alive ping,
 * and dispatching incoming messages to the right handler.
 */

import AppState from './state.js';
import { setStatus, showBanner, hideBanner, showToast } from './utils.js';
import { addMessage, showThinking, hideThinking } from './chat.js';
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
        hideBanner();
        showToast('Connected to Turtle AI');
    };

    AppState.ws.onclose = () => {
        AppState.isConnected = false;
        setStatus('disconnected', 'Disconnected');
        showBanner();
    };

    AppState.ws.onerror = () => {
        AppState.isConnected = false;
        setStatus('disconnected', 'Connection error');
        showBanner();
    };

    AppState.ws.onmessage = (event) => {
        // Binary frame → audio playback
        if (event.data instanceof Blob) {
            playAudioBlob(event.data);
            return;
        }

        try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

/** Dispatch a parsed server JSON message to the right handler */
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
            break;

        case 'timing':
            updateTimings(msg);
            break;

        case 'error':
            hideThinking();
            setStatus('ready', 'Ready');
            showToast(msg.message, true);
            break;

        case 'pong':
            break;

        default:
            console.log('Unknown message type:', msg);
    }
}

/** Map server status values to UI state */
function handleStatusMessage(msg) {
    const labelMap = {
        ready:        'Ready',
        thinking:     'Thinking...',
        transcribing: 'Transcribing...',
        speaking:     'Speaking...',
        restored:     'Session restored',
    };

    setStatus(msg.status, labelMap[msg.status] || msg.status);

    if (msg.status === 'thinking') {
        showThinking('Turtle is thinking...');
    } else if (msg.status === 'transcribing') {
        showThinking('Transcribing speech...');
    } else if (msg.status === 'speaking') {
        showThinking('Generating speech...');
    } else if (msg.status === 'ready' || msg.status === 'restored') {
        hideThinking();
    }
}

/** Start keep-alive ping and auto-reconnect intervals */
export function startConnectionWatchdog() {
    // Keep-alive ping every 30s
    setInterval(() => {
        if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
            AppState.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);

    // Auto-reconnect every 5s when disconnected
    setInterval(() => {
        if (!AppState.isConnected) {
            connectWebSocket();
        }
    }, 5000);
}

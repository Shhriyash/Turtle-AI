/**
 * websocket.js — WebSocket connection, reconnect, message dispatch
 */

import AppState from './state.js';
import { setStatus, showBanner, hideBanner, showToast } from './utils.js';
import { addMessage, showThinking, hideThinking, setBubbleState } from './chat.js';
import { playAudioBlob, handleServerInterrupt } from './voice.js';
import { updateTimings } from './devmode.js';
import { renderConfirmationPrompt } from './memory.js';
import { showHeard } from './ambient.js';

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
        setBubbleState('disconnected');
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
            // thinking caption while the user is still speaking, and echo it
            // under the orb so the user sees themselves being heard.
            if (msg.text) { showThinking(msg.text); showHeard(msg.text, 'heard'); }
            break;
        case 'transcription':
            addMessage('user', msg.text);
            if (msg.text) showHeard(msg.text, 'heard');
            break;
        case 'done':
            hideThinking();
            // WP1.H: msg.tool_urls is the server's allow-list of this turn's
            // tool-sourced URLs. A frame that omits the key entirely (e.g.
            // the budget-refusal "done", which never ran a tool) must render
            // with nothing clickable, not fall back to linkifying everything
            // — addMessage enforces that fail-closed default for the
            // 'assistant' role regardless of what's passed here.
            addMessage('assistant', msg.content, msg.tool_urls);
            setStatus('ready', 'Ready');
            setBubbleState('idle');
            break;
        case 'timing':
            updateTimings(msg);
            break;
        case 'interrupted':
            // Server cancelled the reply (barge-in or explicit interrupt).
            handleServerInterrupt();
            break;
        case 'confirmation_prompt':
            // Server queued an uncertain memory fact behind the gate.
            // Surface it as an inline card the user can confirm/dismiss.
            renderConfirmationPrompt(msg);
            break;
        case 'error':
            hideThinking();
            setStatus('ready', 'Ready');
            setBubbleState('error');
            showToast(msg.message, true);
            setTimeout(() => setBubbleState('idle'), 2400);
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

// Exported (this is otherwise a private dispatch helper) solely so
// test/web_js_wp3c_test.mjs can exercise the "degraded" status branch
// directly without a full DOM/WebSocket harness — see that test file's
// header for why (repo convention: a dependency-free node:assert script,
// no new test framework).
export function handleStatusMessage(msg) {
    const labelMap = {
        ready:        'Ready',
        thinking:     'Thinking',
        transcribing: 'Transcribing',
        listening:    'Listening',
        speaking:     'Speaking',
        restored:     'Session restored',
        // Ledger 3.6(d): the server sends this when it couldn't reach its
        // session store on connect and fell back to a fresh, unsaved
        // in-memory session. Without an entry here this rendered as the
        // literal word "degraded" with none of the special handling below —
        // exactly the "wired at one end only" defect Phase 1 kept shipping.
        degraded:     'Reconnecting memory…',
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
    } else if (msg.status === 'degraded') {
        // The connection is still usable (a fresh session was started), it
        // just won't have prior history/pending drafts this time — tell the
        // user rather than silently losing continuity.
        hideThinking();
        showToast("Couldn't restore your previous session — starting fresh.", true);
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

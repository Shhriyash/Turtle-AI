/**
 * websocket.js — WebSocket connection, reconnect, message dispatch
 */

import AppState from './state.js';
import { setStatus, showBanner, hideBanner, showToast } from './utils.js';
import {
    addMessage,
    showThinking,
    hideThinking,
    setBubbleState,
    setPendingTurn,
    clearPendingTurn,
} from './chat.js';
import { playAudioBlob, handleServerInterrupt } from './voice.js';
import { updateTimings } from './devmode.js';
import { renderConfirmationPrompt } from './memory.js';
import { showHeard } from './ambient.js';

/** Reconnect backoff (ledger 6.8): 1 s, doubling to 30 s, reset on open. */
export const RECONNECT_BASE_MS = 1000;
export const RECONNECT_MAX_MS = 30000;

/** A stable per-tab id. sessionStorage survives a reload and a reconnect but is
 *  NOT shared between tabs, so two tabs are two clients (the server's session
 *  lease then gives the second its own session instead of clobbering the first). */
export function getClientId() {
    if (AppState.clientId) return AppState.clientId;
    let id = null;
    try { id = sessionStorage.getItem('turtle_cid'); } catch (_) {}
    if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
        const c = globalThis.crypto;
        id = (c && c.randomUUID)
            ? c.randomUUID().replace(/-/g, '')
            : Array.from({ length: 24 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
        try { sessionStorage.setItem('turtle_cid', id); } catch (_) {}
    }
    AppState.clientId = id;
    return id;
}

/** Schedule the next reconnect attempt (one timer at a time). */
function scheduleReconnect() {
    if (AppState.reconnectTimer) return;
    const delay = AppState.reconnectDelayMs;
    AppState.reconnectDelayMs = Math.min(delay * 2, RECONNECT_MAX_MS);
    AppState.reconnectTimer = setTimeout(() => {
        AppState.reconnectTimer = null;
        if (!AppState.isConnected) connectWebSocket();
    }, delay);
}

/** Resume protocol: tell the server which session/turn we last saw, then
 *  resend anything it refused to start before the planned cut. */
function sendResumeAndResend(ws) {
    if (AppState.hasConnected && AppState.sessionId) {
        ws.send(JSON.stringify({
            type: 'resume',
            session_id: AppState.sessionId,
            last_turn_id: AppState.lastTurnId,
        }));
    }
    AppState.hasConnected = true;
    const queue = AppState.resendQueue;
    AppState.resendQueue = [];
    for (const item of queue) {
        if (item.kind === 'text') {
            // A typed message already has its bubble; a streamed-mic utterance
            // was never drawn (the server deferred it before `transcription`).
            if (item.source === 'mic') addMessage('user', item.content);
            ws.send(JSON.stringify({ type: 'text', content: item.content }));
        } else if (item.kind === 'audio') {
            ws.send(JSON.stringify({
                type: 'audio', data: item.data, sample_rate: item.sample_rate,
            }));
        }
    }
}

/** Connect (or reconnect) to the WebSocket server */
export function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws?cid=${encodeURIComponent(getClientId())}`;

    if (AppState.reconnectTimer) {
        clearTimeout(AppState.reconnectTimer);
        AppState.reconnectTimer = null;
    }
    if (AppState.ws) {
        try { AppState.ws.close(); } catch (_) {}
    }

    let sock;
    try {
        sock = new WebSocket(wsUrl);
    } catch (_) {
        scheduleReconnect();
        return;
    }
    AppState.ws = sock;

    sock.onopen = () => {
        if (AppState.ws !== sock) return;
        AppState.isConnected = true;
        AppState.reconnectDelayMs = RECONNECT_BASE_MS; // reset on open
        AppState.plannedReconnect = false;
        setStatus('ready', 'Ready');
        setBubbleState('idle');
        hideBanner();
        sendResumeAndResend(sock);
        showToast('Connected to Turtle AI');
    };

    sock.onclose = () => {
        if (AppState.ws !== sock) return; // a superseded socket's late close
        AppState.isConnected = false;
        if (AppState.plannedReconnect) {
            // Deliberate 1012 cut: not an outage, so no banner.
            setStatus('reconnecting', 'Reconnecting');
        } else {
            setStatus('disconnected', 'Disconnected');
            setBubbleState('disconnected');
            showBanner();
        }
        scheduleReconnect();
    };

    sock.onerror = () => {
        if (AppState.ws !== sock) return;
        AppState.isConnected = false;
        if (!AppState.plannedReconnect) {
            setStatus('disconnected', 'Connection error');
            showBanner();
        }
    };

    sock.onmessage = (event) => {
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
            // Only a turn of the CURRENT session is a valid resume cursor.
            if (msg.turn_id && AppState.sessionId
                && msg.turn_id.startsWith(AppState.sessionId + '_turn_')) {
                AppState.lastTurnId = msg.turn_id;
            }
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
        case 'resumed':
            // Answer to our `resume`. If the server could not put us back in
            // the session we were in (e.g. it was finalised after an unclean
            // drop), adopt the new one; its turn numbering starts over.
            if (msg.session_id && msg.session_id !== AppState.sessionId) {
                AppState.sessionId = msg.session_id;
                AppState.lastTurnId = null;
            }
            break;
        case 'turn_queued':
            // Ledger 6.1: a turn is already running, so the server queued this
            // message one deep (and `replaced` means it displaced an earlier
            // queued one). Drive the pending state from the SERVER's frame
            // rather than inferring it client-side: the server owns the queue,
            // so only it knows whether the message was actually accepted.
            setPendingTurn(msg.content);
            break;
        case 'turn_queue_cleared':
            // The queued message will never run (an interrupt discarded it).
            // Clear the pending bubble so it does not sit there implying the
            // answer is still coming.
            clearPendingTurn();
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
        reconnect:    'Reconnecting',
    };

    // Ledger 6.8: the server will not start a turn that cannot finish before
    // its connection ceiling. It sends `reconnect` (carrying any messages it
    // refused), then closes with 1012; we reconnect, send `resume`, and resend.
    if (msg.status === 'reconnect') {
        AppState.plannedReconnect = true;
        AppState.isConnected = false; // sendMessage now keeps text in the box
        if (Array.isArray(msg.unstarted)) {
            AppState.resendQueue.push(...msg.unstarted);
        }
        setStatus('reconnecting', 'Reconnecting');
        return;
    }

    // The ready frame advertises whether the server has streaming STT enabled.
    if (msg.status === 'ready') {
        AppState.streamSttEnabled = !!msg.stream_stt;
        // First connect (no resume sent): the ready frame names the session.
        // On a reconnect the `resumed` frame decides instead.
        if (msg.session_id && !AppState.sessionId) {
            AppState.sessionId = msg.session_id;
            AppState.lastTurnId = null;
        }
    }
    if (msg.status === 'restored' && msg.session_id && !AppState.sessionId) {
        AppState.sessionId = msg.session_id;
        AppState.lastTurnId = null;
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

    // Reconnection is event-driven now (onclose -> scheduleReconnect, 1 s
    // doubling to 30 s), replacing the flat 5 s poll.
}

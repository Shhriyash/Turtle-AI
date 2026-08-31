/**
 * ambient.js — The ambient state machine for the central orb
 *
 * Ported from the "Turtle Ambient States" design canvas. Every visual
 * layer (rings, core, waveform, captions) reads a single [data-state]
 * attribute on .bubble-container, so there is exactly one source of
 * truth and no class-combination drift.
 *
 * The whole thing is attribute swaps + CSS animations — no rAF loop,
 * no timers per frame, nothing on the audio or WebSocket hot path.
 */

import AppState from './state.js';

/** Server/voice status names → canonical ambient states. */
const ALIASES = {
    ready:        'idle',
    idle:         'idle',
    restored:     'idle',
    recording:    'listening',
    listening:    'listening',
    transcribing: 'transcribing',
    thinking:     'thinking',
    speaking:     'speaking',
    error:        'error',
    disconnected: 'disconnected',
};

/** Caption shown beneath the wordmark per state. */
const CAPTIONS = {
    idle:         'Ask me anything',
    listening:    'Listening',
    transcribing: 'Transcribing',
    thinking:     'Thinking',
    speaking:     'Speaking',
    error:        'Something went wrong',
    disconnected: 'Reconnecting',
};

/** States that are part of an active turn (greeting hides, session live). */
const LIVE = new Set(['listening', 'transcribing', 'thinking', 'speaking']);

let current = 'idle';
let heardClearTimer = null;

/** Time-of-day greeting, matching the canvas's "Good evening." */
function greetingText() {
    const h = new Date().getHours();
    if (h < 5)  return 'Still up.';
    if (h < 12) return 'Good morning.';
    if (h < 17) return 'Good afternoon.';
    if (h < 22) return 'Good evening.';
    return 'Good evening.';
}

/** Refresh the greeting line (called once at boot). */
export function initAmbient() {
    const el = document.getElementById('bubble-greeting');
    if (el) el.textContent = greetingText();
    setAmbientState('idle');
}

/**
 * Move the orb to a named state.
 * Unknown names fall back to idle rather than leaving a stale look.
 */
export function setAmbientState(state) {
    const next = ALIASES[state] || 'idle';
    const container = document.getElementById('bubble-container');
    const status = AppState.dom.bubbleStatus || document.getElementById('bubble-status');
    if (!container) return;

    current = next;
    container.setAttribute('data-state', next);

    if (status) {
        status.textContent = CAPTIONS[next] || CAPTIONS.idle;
        status.classList.toggle('active', LIVE.has(next));
    }

    // Leaving a live turn clears whatever transcript line was showing.
    if (!LIVE.has(next)) clearHeard();
}

export function getAmbientState() {
    return current;
}

/**
 * Show a transcript line under the orb.
 * `kind` is 'heard' (what the user said) or 'spoken' (what Turtle replied).
 */
export function showHeard(text, kind = 'heard') {
    const wrap = document.getElementById('bubble-heard');
    if (!wrap || !text) return;

    clearTimeout(heardClearTimer);
    heardClearTimer = null;

    const cls = kind === 'spoken' ? 'spoken-text' : 'heard-text';
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = kind === 'spoken' ? `“${text}”` : text;

    wrap.replaceChildren(span);
    wrap.classList.add('visible');
}

/** Fade the transcript line out (optionally after a delay). */
export function clearHeard(delayMs = 0) {
    const wrap = document.getElementById('bubble-heard');
    if (!wrap) return;

    clearTimeout(heardClearTimer);
    const run = () => {
        wrap.classList.remove('visible');
        heardClearTimer = null;
    };
    if (delayMs > 0) heardClearTimer = setTimeout(run, delayMs);
    else run();
}

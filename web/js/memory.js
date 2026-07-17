/**
 * memory.js — Memory confirmation UI (Phase 2 / W4)
 *
 * Two surfaces, one server contract:
 *
 *   1. Inline confirmation card — rendered in the response panel when the
 *      server pushes a `confirmation_prompt` sidecar frame over the WS.
 *      Yes / No / Later. Yes/No POST /api/memory/confirm for every event_id
 *      folded into the prompt.
 *
 *   2. Pending-memories panel — a header toggle that GETs /api/memory/pending
 *      and lists each queued candidate with per-item Approve / Reject, wired
 *      to the same confirm endpoint.
 *
 * Server contract (read from apps/turtle_server.py, do NOT change here):
 *   GET  /api/memory/pending -> { "pending": [ {event_id, question, topic, key}, ... ] }
 *   POST /api/memory/confirm  body { "event_id": <str>, "accepted": <bool> }
 *                             -> 200 { "status": "ok", "applied": <bool> }
 *                                (400/401/404 with { "error": <str> } otherwise)
 *
 * Dependency-free vanilla JS. Every DOM lookup is guarded so a missing
 * element never throws.
 */

import AppState from './state.js';
import { escapeHtml, showToast } from './utils.js';

const PENDING_URL = '/api/memory/pending';
const CONFIRM_URL = '/api/memory/confirm';

// ── Server calls ─────────────────────────────────────────────

/**
 * Resolve a single candidate. Returns the parsed JSON body on HTTP 200,
 * or throws an Error carrying the server's message on any failure.
 */
async function postConfirm(eventId, accepted) {
    let res;
    try {
        res = await fetch(CONFIRM_URL, {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ event_id: eventId, accepted: !!accepted }),
        });
    } catch (_) {
        throw new Error('Network error');
    }
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
    }
    return data;
}

/**
 * Resolve every event_id in a batch with the same answer. Rejects on the
 * first failure so the caller can keep the buttons live and show the error.
 */
async function confirmBatch(eventIds, accepted) {
    const ids = (eventIds || []).filter(Boolean);
    for (const id of ids) {
        await postConfirm(id, accepted);
    }
}

async function fetchPending() {
    let res;
    try {
        res = await fetch(PENDING_URL, { credentials: 'same-origin' });
    } catch (_) {
        throw new Error('Network error');
    }
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
    }
    return Array.isArray(data.pending) ? data.pending : [];
}

// ── Small builders ───────────────────────────────────────────

/** Subtle "topic · key" line, only when the server provided them. */
function metaLine(topic, key) {
    const parts = [topic, key].filter((p) => typeof p === 'string' && p.length);
    if (!parts.length) return '';
    return `<div class="mem-meta">${parts.map(escapeHtml).join(' · ')}</div>`;
}

// ── Inline confirmation card (from the WS sidecar frame) ──────

/**
 * Render an inline confirmation card in the response panel.
 * @param {{event_ids?: string[], topic?: string, key?: string, message?: string}} msg
 */
export function renderConfirmationPrompt(msg) {
    const container = AppState.dom.responseMessages;
    if (!container) return;

    const eventIds = Array.isArray(msg.event_ids) ? msg.event_ids.filter(Boolean) : [];
    if (!eventIds.length) return;

    // The response panel must be visible for the card to be seen.
    if (AppState.dom.responsePanel) {
        AppState.responsePanelOpen = true;
        AppState.dom.responsePanel.classList.add('open');
    }

    const card = document.createElement('div');
    card.className = 'memory-card';
    card.innerHTML = `
        <div class="memory-card-label">Memory suggestion</div>
        <div class="memory-card-msg">${escapeHtml(msg.message || 'Want me to remember this?')}</div>
        ${metaLine(msg.topic, msg.key)}
        <div class="memory-card-error" hidden></div>
        <div class="mem-actions">
            <button type="button" class="mem-btn mem-btn-yes">Yes</button>
            <button type="button" class="mem-btn mem-btn-no">No</button>
            <button type="button" class="mem-btn mem-btn-later">Later</button>
        </div>
    `;

    const errEl = card.querySelector('.memory-card-error');
    const actions = card.querySelector('.mem-actions');
    const btnYes = card.querySelector('.mem-btn-yes');
    const btnNo = card.querySelector('.mem-btn-no');
    const btnLater = card.querySelector('.mem-btn-later');

    const setBusy = (busy) => {
        [btnYes, btnNo, btnLater].forEach((b) => { if (b) b.disabled = busy; });
    };

    const settle = (noteText) => {
        if (actions) actions.remove();
        if (errEl) errEl.hidden = true;
        const note = document.createElement('div');
        note.className = 'memory-card-note';
        note.textContent = noteText;
        card.appendChild(note);
        card.classList.add('settled');
    };

    const answer = async (accepted) => {
        if (errEl) errEl.hidden = true;
        setBusy(true);
        try {
            await confirmBatch(eventIds, accepted);
            settle(accepted ? 'Saved' : 'Discarded');
            // Keep the standalone panel (if open) in sync.
            refreshPendingPanel();
        } catch (e) {
            if (errEl) {
                errEl.textContent = e.message || 'Could not save. Try again.';
                errEl.hidden = false;
            }
            setBusy(false);
        }
    };

    if (btnYes) btnYes.addEventListener('click', () => answer(true));
    if (btnNo) btnNo.addEventListener('click', () => answer(false));
    if (btnLater) btnLater.addEventListener('click', () => {
        // "Later" leaves the candidate pending on the server; just dismiss.
        card.remove();
    });

    container.appendChild(card);
    requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
}

// ── Pending-memories panel ───────────────────────────────────

function panelEls() {
    return {
        panel: document.getElementById('memory-panel'),
        toggle: document.getElementById('btn-memory-toggle'),
        list: document.getElementById('memory-pending-list'),
    };
}

/** Render one pending candidate row into the panel list. */
function renderPendingItem(item) {
    const row = document.createElement('div');
    row.className = 'mem-item';

    const q = document.createElement('div');
    q.className = 'mem-item-q';
    q.textContent = item.question || 'Remember this?';
    row.appendChild(q);

    const metaHtml = metaLine(item.topic, item.key);
    if (metaHtml) {
        const meta = document.createElement('div');
        meta.innerHTML = metaHtml;
        row.appendChild(meta.firstElementChild);
    }

    const err = document.createElement('div');
    err.className = 'memory-card-error';
    err.hidden = true;
    row.appendChild(err);

    const actions = document.createElement('div');
    actions.className = 'mem-actions';

    const approve = document.createElement('button');
    approve.type = 'button';
    approve.className = 'mem-btn mem-btn-yes';
    approve.textContent = 'Approve';

    const reject = document.createElement('button');
    reject.type = 'button';
    reject.className = 'mem-btn mem-btn-no';
    reject.textContent = 'Reject';

    const act = async (accepted) => {
        if (!item.event_id) return;
        err.hidden = true;
        approve.disabled = true;
        reject.disabled = true;
        try {
            await postConfirm(item.event_id, accepted);
            // Refresh the whole list so the queue reflects the server truth.
            refreshPendingPanel();
        } catch (e) {
            err.textContent = e.message || 'Action failed. Try again.';
            err.hidden = false;
            approve.disabled = false;
            reject.disabled = false;
        }
    };

    approve.addEventListener('click', () => act(true));
    reject.addEventListener('click', () => act(false));
    actions.appendChild(approve);
    actions.appendChild(reject);
    row.appendChild(actions);
    return row;
}

/** Fetch pending candidates and (re)render the panel list. */
export async function refreshPendingPanel() {
    const { list } = panelEls();
    if (!list) return;

    let items;
    try {
        items = await fetchPending();
    } catch (e) {
        list.innerHTML = '';
        const errEl = document.createElement('div');
        errEl.className = 'memory-empty';
        errEl.textContent = e.message || 'Could not load pending memories.';
        list.appendChild(errEl);
        return;
    }

    list.innerHTML = '';
    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'memory-empty';
        empty.textContent = 'Nothing waiting to be confirmed.';
        list.appendChild(empty);
        return;
    }
    for (const item of items) {
        list.appendChild(renderPendingItem(item));
    }
}

/** Open/close the pending-memories panel. */
export function toggleMemoryPanel() {
    const { panel, toggle } = panelEls();
    if (!panel) return;
    const open = !panel.classList.contains('open');
    panel.classList.toggle('open', open);
    if (toggle) toggle.classList.toggle('active', open);
    if (open) refreshPendingPanel();
}

function closeMemoryPanel() {
    const { panel, toggle } = panelEls();
    if (panel) panel.classList.remove('open');
    if (toggle) toggle.classList.remove('active');
}

/** Wire the panel toggle + close button. Safe if elements are absent. */
export function initMemoryUI() {
    const { toggle } = panelEls();
    if (toggle) toggle.addEventListener('click', toggleMemoryPanel);

    const closeBtn = document.getElementById('btn-memory-close');
    if (closeBtn) closeBtn.addEventListener('click', closeMemoryPanel);

    const refreshBtn = document.getElementById('btn-memory-refresh');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshPendingPanel);
}

/**
 * chat.js — Message rendering in the response panel + bubble state
 *
 * Messages appear in the right-side panel. The central bubble
 * changes visual state based on system status.
 */

import AppState from './state.js';
import { escapeHtml, scrollToBottom } from './utils.js';
import { setAmbientState } from './ambient.js';

// ── Bubble state management ──────────────────────────────────

/**
 * Set the bubble to a named visual state.
 *
 * The ambient state machine owns every visual layer now; this stays
 * as the app-wide entry point so existing call sites are unchanged.
 */
export function setBubbleState(state) {
    setAmbientState(state);
}

// ── Thinking indicator in the panel ──────────────────────────

export function showThinking(label) {
    AppState.isThinking = true;
    const { panelThinking, panelThinkingLabel } = AppState.dom;
    if (panelThinkingLabel) panelThinkingLabel.textContent = label || 'Thinking';
    if (panelThinking) panelThinking.classList.add('visible');
    openResponsePanel();
    scrollPanelToBottom();
}

export function hideThinking() {
    AppState.isThinking = false;
    const { panelThinking } = AppState.dom;
    if (panelThinking) panelThinking.classList.remove('visible');
}

// ── Response panel management ────────────────────────────────

export function openResponsePanel() {
    if (AppState.responsePanelOpen) return;
    AppState.responsePanelOpen = true;
    AppState.dom.responsePanel.classList.add('open');
    updateChatToggleUi();
}

export function closeResponsePanel() {
    AppState.responsePanelOpen = false;
    AppState.dom.responsePanel.classList.remove('open');
    updateChatToggleUi();
}

/** Toggle the response panel — lets the user reopen a closed chat to review it. */
export function toggleResponsePanel() {
    if (AppState.responsePanelOpen) {
        closeResponsePanel();
    } else {
        openResponsePanel();
        scrollPanelToBottom();
    }
    updateChatToggleUi();
}

/** Reflect panel state on the floating toggle (hide it while the panel is open). */
export function updateChatToggleUi() {
    const btn = AppState.dom.btnChatToggle;
    if (!btn) return;
    btn.classList.toggle('active', AppState.responsePanelOpen);
    btn.setAttribute('aria-pressed', AppState.responsePanelOpen ? 'true' : 'false');
}

function scrollPanelToBottom() {
    const el = AppState.dom.responseMessages;
    if (el) {
        requestAnimationFrame(() => {
            el.scrollTop = el.scrollHeight;
        });
    }
}

// ── Message rendering ────────────────────────────────────────

/**
 * Add a message to the response panel.
 * @param {'user'|'assistant'} role
 * @param {string} text
 * @param {string[]|undefined} toolUrls - WP1.H: URLs the server confirms came
 *   from a tool result THIS turn (the "done" frame's `tool_urls`). When
 *   provided, only URLs in this list are rendered as clickable anchors — a
 *   URL the model merely wrote in prose, that no tool returned, renders as
 *   plain text. `undefined` (the legacy call shape, still used for the
 *   user's own typed message, which is trusted input — not attacker text)
 *   keeps the old unrestricted linkify behaviour.
 */
export function addMessage(role, text, toolUrls) {
    openResponsePanel();

    const container = AppState.dom.responseMessages;
    if (!container) return;

    const msg = document.createElement('div');
    msg.className = `panel-msg panel-msg-${role}`;

    const label = document.createElement('div');
    label.className = 'panel-msg-label';
    label.textContent = role === 'user' ? 'You' : 'Turtle';

    const content = document.createElement('div');
    content.className = 'panel-msg-content';
    content.innerHTML = formatMessage(text, toolUrls);

    msg.appendChild(label);
    msg.appendChild(content);
    container.appendChild(msg);
    scrollPanelToBottom();
}

/**
 * Best-effort hostname for display, or null if the URL doesn't parse. Never
 * throws — a malformed URL just falls back to "not allowed".
 */
function safeHost(url) {
    try {
        return new URL(url).host || null;
    } catch {
        return null;
    }
}

/**
 * WP1.H: a URL is only ever turned into a clickable anchor when it is
 * EXACTLY one the server reported as tool-sourced this turn (`toolUrls`).
 * Exact-string match rather than same-host match: a tool returning
 * https://reuters.com/page-a should not license the model to turn
 * https://reuters.com/anything-it-invents into a link — same host, but the
 * path wasn't part of any tool result.
 *
 * `toolUrls === undefined` means "no allow-list was supplied" (the legacy
 * call shape) and everything is allowed, preserving old behaviour for the
 * user's own trusted, self-typed message.
 */
function isAllowedUrl(url, toolUrls) {
    if (toolUrls === undefined) return true;
    return Array.isArray(toolUrls) && toolUrls.includes(url);
}

/** Lightweight markdown to HTML */
export function formatMessage(text, toolUrls) {
    let html = escapeHtml(text);
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Markdown links. Only rendered as an anchor when the URL is allow-listed
    // (see isAllowedUrl); the model-supplied label is discarded in favour of
    // the host so a deceptive label (`[paypal.com](https://evil.example)`)
    // can't borrow trust from a URL it doesn't match. When not allow-listed,
    // fall back to the plain (already-escaped) label text — never a raw,
    // clickable URL the model merely wrote in prose.
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_match, label, url) => {
        if (isAllowedUrl(url, toolUrls)) {
            const host = safeHost(url);
            if (host) {
                return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${host}">${escapeHtml(host)}</a>`;
            }
        }
        return label;
    });

    // Raw URLs (using negative lookbehind to avoid replacing URLs inside href
    // attributes). Same allow-list gate and host-only label as above; a
    // non-allow-listed bare URL renders as plain escaped text, not a link.
    html = html.replace(/(?<!href=")(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/g, (match, url) => {
        if (isAllowedUrl(url, toolUrls)) {
            const host = safeHost(url);
            if (host) {
                return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${host}">${escapeHtml(host)}</a>`;
            }
        }
        return match;
    });

    // Bold / Italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ── Text sending ─────────────────────────────────────────────

export function sendMessage() {
    const { chatInput, btnSend } = AppState.dom;
    const text = chatInput.value.trim();
    if (!text || !AppState.isConnected || AppState.isThinking) return;

    addMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    AppState.ws.send(JSON.stringify({ type: 'text', content: text }));
}

export function handleInputKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

export function setupInputAutosize() {
    const { chatInput, btnSend } = AppState.dom;
    if (!chatInput) return;
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        btnSend.disabled = chatInput.value.trim() === '' || !AppState.isConnected;
    });
}

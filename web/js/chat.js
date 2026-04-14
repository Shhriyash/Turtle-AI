/**
 * chat.js — Message rendering in the response panel + bubble state
 *
 * Messages appear in the right-side panel. The central bubble
 * changes visual state based on system status.
 */

import AppState from './state.js';
import { escapeHtml, scrollToBottom } from './utils.js';

// ── Bubble state management ──────────────────────────────────

/** Set the bubble to a named visual state */
export function setBubbleState(state) {
    const { bubbleOrb, bubbleGlow, bubbleStatus } = AppState.dom;
    if (!bubbleOrb) return;

    // Clear all state classes
    bubbleOrb.classList.remove('listening', 'thinking', 'speaking');
    bubbleGlow.classList.remove('active');

    const labels = {
        idle:         'Ready',
        listening:    'Listening',
        recording:    'Listening',
        transcribing: 'Transcribing',
        thinking:     'Thinking',
        speaking:     'Speaking',
        ready:        'Ready',
    };

    bubbleStatus.textContent = labels[state] || 'Ready';
    bubbleStatus.classList.toggle('active', state !== 'idle' && state !== 'ready');

    if (state === 'listening' || state === 'recording') {
        bubbleOrb.classList.add('listening');
        bubbleGlow.classList.add('active');
    } else if (state === 'thinking' || state === 'transcribing') {
        bubbleOrb.classList.add('thinking');
        bubbleGlow.classList.add('active');
    } else if (state === 'speaking') {
        bubbleOrb.classList.add('speaking');
        bubbleGlow.classList.add('active');
    }
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
}

export function closeResponsePanel() {
    AppState.responsePanelOpen = false;
    AppState.dom.responsePanel.classList.remove('open');
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
 */
export function addMessage(role, text) {
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
    content.innerHTML = formatMessage(text);

    msg.appendChild(label);
    msg.appendChild(content);
    container.appendChild(msg);
    scrollPanelToBottom();
}

/** Lightweight markdown to HTML */
export function formatMessage(text) {
    let html = escapeHtml(text);
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
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

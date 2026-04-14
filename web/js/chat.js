/**
 * chat.js — Message rendering and text input handling
 *
 * Manages the chat message list, markdown formatting, thinking
 * indicator, and text sending via WebSocket.
 */

import AppState from './state.js';
import { escapeHtml, scrollToBottom } from './utils.js';

/** Show the thinking indicator with a label */
export function showThinking(label) {
    AppState.isThinking = true;
    const { thinkingEl, thinkingLabel } = AppState.dom;
    if (thinkingLabel) thinkingLabel.textContent = label || 'Turtle is thinking...';
    if (thinkingEl) thinkingEl.classList.add('visible');
    scrollToBottom();
}

/** Hide the thinking indicator */
export function hideThinking() {
    AppState.isThinking = false;
    const { thinkingEl } = AppState.dom;
    if (thinkingEl) thinkingEl.classList.remove('visible');
}

/**
 * Add a message bubble to the chat area.
 * @param {'user'|'assistant'} role
 * @param {string} text
 */
export function addMessage(role, text) {
    const { welcomeScreen, messagesScroll } = AppState.dom;

    // Hide welcome, show messages container
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (messagesScroll) messagesScroll.style.display = 'flex';

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'assistant' ? '\u{1F422}' : '\u{1F464}';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMessage(text);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    messagesScroll.appendChild(msgDiv);
    scrollToBottom();
}

/** Lightweight markdown-ish → HTML conversion */
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

/** Send the current text input value as a chat message */
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

/** Send a suggestion chip's text */
export function sendSuggestion(el) {
    const text = el.textContent.replace(/^[^\s]+\s/, ''); // Strip leading emoji
    const { chatInput, btnSend } = AppState.dom;
    chatInput.value = text;
    btnSend.disabled = false;
    sendMessage();
}

/** Handle Enter key (without Shift) to send */
export function handleInputKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/** Auto-resize the textarea and toggle Send button */
export function setupInputAutosize() {
    const { chatInput, btnSend } = AppState.dom;
    if (!chatInput) return;
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
        btnSend.disabled = chatInput.value.trim() === '' || !AppState.isConnected;
    });
}

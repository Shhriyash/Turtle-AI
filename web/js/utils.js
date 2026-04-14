/**
 * utils.js — Shared UI helpers
 *
 * Toast notifications, status indicator, scroll, HTML escape.
 */

import AppState from './state.js';

let toastTimer = null;

/** Update the header status indicator pill */
export function setStatus(status, text) {
    const { statusIndicator, statusText } = AppState.dom;
    if (statusIndicator) statusIndicator.setAttribute('data-status', status);
    if (statusText) statusText.textContent = text;
}

/** Show the connection-error banner */
export function showBanner() {
    const el = AppState.dom.connectionBanner;
    if (el) el.classList.add('visible');
}

/** Hide the connection-error banner */
export function hideBanner() {
    const el = AppState.dom.connectionBanner;
    if (el) el.classList.remove('visible');
}

/** Show a temporary toast notification */
export function showToast(msg, isError = false) {
    const toast = AppState.dom.toast;
    if (!toast) return;
    toast.textContent = msg;
    toast.style.borderColor = isError ? 'var(--error)' : 'var(--accent)';
    toast.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('visible'), 3000);
}

/** Smooth-scroll the messages area to the bottom */
export function scrollToBottom() {
    const area = AppState.dom.messagesArea;
    if (area) {
        requestAnimationFrame(() => {
            area.scrollTop = area.scrollHeight;
        });
    }
}

/** Escape HTML to prevent XSS in message rendering */
export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

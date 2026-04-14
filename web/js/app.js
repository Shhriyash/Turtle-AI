/**
 * app.js — Application entry point
 *
 * Initializes DOM references, wires event listeners,
 * connects WebSocket, starts connection watchdog.
 */

import AppState from './state.js';
import { connectWebSocket, startConnectionWatchdog } from './websocket.js';
import { sendMessage, sendSuggestion, handleInputKey, setupInputAutosize } from './chat.js';
import { startRecording, stopRecording } from './voice.js';
import { toggleDevPanel, applyDevConfig, resetDevDefaults } from './devmode.js';

/** Initialize DOM element cache */
function initDOM() {
    AppState.dom.messagesScroll   = document.getElementById('messages-scroll');
    AppState.dom.messagesArea     = document.getElementById('messages-area');
    AppState.dom.welcomeScreen    = document.getElementById('welcome-screen');
    AppState.dom.chatInput        = document.getElementById('chat-input');
    AppState.dom.btnSend          = document.getElementById('btn-send');
    AppState.dom.btnMic           = document.getElementById('btn-mic');
    AppState.dom.thinkingEl       = document.getElementById('thinking-indicator');
    AppState.dom.thinkingLabel    = document.getElementById('thinking-label');
    AppState.dom.statusIndicator  = document.getElementById('status-indicator');
    AppState.dom.statusText       = document.getElementById('status-text');
    AppState.dom.connectionBanner = document.getElementById('connection-banner');
    AppState.dom.devSidebar       = document.getElementById('dev-sidebar');
    AppState.dom.btnDevToggle     = document.getElementById('btn-dev-toggle');
    AppState.dom.toast            = document.getElementById('toast');
}

/** Wire up all event listeners */
function initEvents() {
    // Send button
    AppState.dom.btnSend.addEventListener('click', sendMessage);

    // Input keyboard
    AppState.dom.chatInput.addEventListener('keydown', handleInputKey);

    // Mic: push-to-talk (mouse + touch)
    const mic = AppState.dom.btnMic;
    mic.addEventListener('mousedown', startRecording);
    mic.addEventListener('mouseup', stopRecording);
    mic.addEventListener('mouseleave', stopRecording);
    mic.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    mic.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });

    // Dev panel toggle
    AppState.dom.btnDevToggle.addEventListener('click', toggleDevPanel);

    // Suggestion chips
    document.querySelectorAll('.welcome-chip').forEach(chip => {
        chip.addEventListener('click', () => sendSuggestion(chip));
    });

    // Dev actions
    document.getElementById('btn-apply-config')?.addEventListener('click', applyDevConfig);
    document.getElementById('btn-reset-config')?.addEventListener('click', resetDevDefaults);

    // Reconnect button
    document.getElementById('btn-reconnect')?.addEventListener('click', connectWebSocket);

    // Dev sidebar close button
    document.getElementById('btn-dev-close')?.addEventListener('click', toggleDevPanel);

    // Input autosize
    setupInputAutosize();
}

/** Boot the application */
function init() {
    initDOM();
    initEvents();
    connectWebSocket();
    startConnectionWatchdog();
    AppState.dom.chatInput.focus();
}

// Run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

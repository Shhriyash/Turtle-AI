/**
 * app.js — Application entry point
 *
 * Initializes DOM references, wires event listeners,
 * connects WebSocket, starts connection watchdog.
 */

import AppState from './state.js';
import { connectWebSocket, startConnectionWatchdog } from './websocket.js';
import { sendMessage, handleInputKey, setupInputAutosize, closeResponsePanel, setBubbleState } from './chat.js';
import { startRecording, stopRecording } from './voice.js';
import { toggleDevPanel, applyDevConfig, resetDevDefaults } from './devmode.js';

/** Initialize DOM element cache */
function initDOM() {
    AppState.dom.chatInput        = document.getElementById('chat-input');
    AppState.dom.btnSend          = document.getElementById('btn-send');
    AppState.dom.statusIndicator  = document.getElementById('status-indicator');
    AppState.dom.statusText       = document.getElementById('status-text');
    AppState.dom.connectionBanner = document.getElementById('connection-banner');
    AppState.dom.devSidebar       = document.getElementById('dev-sidebar');
    AppState.dom.btnDevToggle     = document.getElementById('btn-dev-toggle');
    AppState.dom.toast            = document.getElementById('toast');

    // Bubble
    AppState.dom.bubbleOrb        = document.getElementById('bubble-orb');
    AppState.dom.bubbleGlow       = document.getElementById('bubble-glow');
    AppState.dom.bubbleStatus     = document.getElementById('bubble-status');

    // Response panel
    AppState.dom.responsePanel    = document.getElementById('response-panel');
    AppState.dom.responseMessages = document.getElementById('response-messages');
    AppState.dom.panelThinking    = document.getElementById('panel-thinking');
    AppState.dom.panelThinkingLabel = document.getElementById('panel-thinking-label');
    AppState.dom.panelTiming      = document.getElementById('panel-timing');
}

/** Wire up all event listeners */
function initEvents() {
    // Send button
    AppState.dom.btnSend.addEventListener('click', sendMessage);

    // Input keyboard
    AppState.dom.chatInput.addEventListener('keydown', handleInputKey);

    // Bubble: push-to-talk (mouse + touch)
    const orb = AppState.dom.bubbleOrb;
    orb.addEventListener('mousedown', (e) => { e.preventDefault(); startRecording(); });
    orb.addEventListener('mouseup',   (e) => { e.preventDefault(); stopRecording(); });
    orb.addEventListener('mouseleave', () => { if (AppState.isRecording) stopRecording(); });
    orb.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    orb.addEventListener('touchend',   (e) => { e.preventDefault(); stopRecording(); });

    // Expose bubble handlers for inline event attrs (fallback)
    window._bubbleDown = (e) => { e.preventDefault(); startRecording(); };
    window._bubbleUp   = (e) => { e.preventDefault(); stopRecording(); };

    // Dev panel toggle
    AppState.dom.btnDevToggle.addEventListener('click', toggleDevPanel);

    // Dev actions
    document.getElementById('btn-apply-config')?.addEventListener('click', applyDevConfig);
    document.getElementById('btn-reset-config')?.addEventListener('click', resetDevDefaults);

    // Reconnect button
    document.getElementById('btn-reconnect')?.addEventListener('click', connectWebSocket);

    // Dev sidebar close button
    document.getElementById('btn-dev-close')?.addEventListener('click', toggleDevPanel);

    // Response panel close button
    document.getElementById('btn-panel-close')?.addEventListener('click', closeResponsePanel);

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

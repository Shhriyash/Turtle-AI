/**
 * app.js — Application entry point
 *
 * Initializes DOM references, wires event listeners,
 * connects WebSocket, starts connection watchdog.
 */

import AppState from './state.js';
import { connectWebSocket, startConnectionWatchdog } from './websocket.js';
import { sendMessage, handleInputKey, setupInputAutosize, closeResponsePanel } from './chat.js';
import { startRecording, stopRecording, refreshVoiceButtonUi } from './voice.js';
import { toggleDevPanel, applyDevConfig, resetDevDefaults } from './devmode.js';

function isTypingTarget(target) {
    const el = target;
    if (!el || !(el instanceof HTMLElement)) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON' || el.isContentEditable;
}

function updateVoiceModeUi() {
    const btn = AppState.dom.btnVoiceMode;
    if (!btn) return;

    const isAuto = AppState.voiceMode === 'vad';
    btn.classList.toggle('auto', isAuto);
    btn.textContent = isAuto ? 'AUTO' : 'PTT';
    btn.title = isAuto
        ? 'Voice mode: Automatic voice detection (click mic to start)'
        : 'Voice mode: Push-to-talk (hold mic or hold space)';

    refreshVoiceButtonUi();
}

/** Initialize DOM element cache */
function initDOM() {
    AppState.dom.chatInput        = document.getElementById('chat-input');
    AppState.dom.btnSend          = document.getElementById('btn-send');
    AppState.dom.btnVoiceMode     = document.getElementById('btn-voice-mode');
    AppState.dom.btnVoice         = document.getElementById('btn-voice');
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

    // Toggle voice mode (PTT <-> Auto VAD)
    AppState.dom.btnVoiceMode?.addEventListener('click', () => {
        if (AppState.isRecording) stopRecording();
        AppState.voiceMode = AppState.voiceMode === 'ptt' ? 'vad' : 'ptt';
        AppState.pttSpaceHeld = false;
        updateVoiceModeUi();
    });

    // Voice input button
    let suppressNextVoiceClick = false;
    const voiceBtn = AppState.dom.btnVoice;

    voiceBtn?.addEventListener('pointerdown', async (e) => {
        if (AppState.voiceMode !== 'ptt') return;
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        e.preventDefault();
        suppressNextVoiceClick = true;
        await startRecording();
    });

    const stopPttPointerRecording = (e) => {
        if (AppState.voiceMode !== 'ptt') return;
        e.preventDefault();
        if (AppState.isRecording) stopRecording();
    };

    voiceBtn?.addEventListener('pointerup', stopPttPointerRecording);
    voiceBtn?.addEventListener('pointercancel', stopPttPointerRecording);
    voiceBtn?.addEventListener('pointerleave', (e) => {
        if (AppState.voiceMode !== 'ptt') return;
        if (e.buttons === 0 && AppState.isRecording) stopRecording();
    });

    // Click-to-toggle when Auto VAD mode is active
    voiceBtn?.addEventListener('click', async () => {
        if (AppState.voiceMode === 'ptt') {
            if (suppressNextVoiceClick) {
                suppressNextVoiceClick = false;
            }
            return;
        }
        if (AppState.isRecording) {
            stopRecording();
            return;
        }
        await startRecording();
    });

    // Hold SPACE to record in PTT mode (like CLI option 1)
    window.addEventListener('keydown', async (e) => {
        if (AppState.voiceMode !== 'ptt') return;
        if (e.code !== 'Space' || e.repeat) return;
        if (isTypingTarget(e.target)) return;
        e.preventDefault();
        AppState.pttSpaceHeld = true;
        await startRecording();
    });

    window.addEventListener('keyup', (e) => {
        if (AppState.voiceMode !== 'ptt') return;
        if (e.code !== 'Space') return;
        if (!AppState.pttSpaceHeld) return;
        e.preventDefault();
        AppState.pttSpaceHeld = false;
        if (AppState.isRecording) stopRecording();
    });

    // Input keyboard
    AppState.dom.chatInput.addEventListener('keydown', handleInputKey);

    // Bubble interaction follows the selected voice mode.
    const orb = AppState.dom.bubbleOrb;

    async function onPointerDown(e) {
        if (e.type === 'mousedown' && e.button !== 0) return;
        e.preventDefault();
        if (AppState.voiceMode !== 'ptt') return;
        if (!AppState.isRecording) await startRecording();
    }

    async function onPointerUp(e) {
        e.preventDefault();
        if (AppState.voiceMode === 'ptt') {
            if (AppState.isRecording) stopRecording();
            return;
        }

        // Auto-VAD mode: tap bubble once to start, tap again to stop.
        if (AppState.isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    orb.addEventListener('mousedown', onPointerDown);
    orb.addEventListener('touchstart', onPointerDown);
    orb.addEventListener('mouseup', onPointerUp);
    orb.addEventListener('touchend', onPointerUp);

    orb.addEventListener('mouseleave', () => {
        if (AppState.voiceMode === 'ptt' && AppState.isRecording) {
            stopRecording();
        }
    });

    // Expose bubble handlers for inline event attrs (fallback)
    window._bubbleDown = onPointerDown;
    window._bubbleUp   = onPointerUp;

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
    updateVoiceModeUi();
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

/**
 * state.js — Global application state singleton
 *
 * All modules import from here instead of using globals.
 * Provides a clean interface for mutation and observation.
 */

const AppState = {
    /** @type {WebSocket|null} */
    ws: null,

    /** Connection flags */
    isConnected: false,
    isThinking: false,
    isRecording: false,

    /** Audio recording state */
    /** @type {AudioContext|null} */
    audioContext: null,
    /** @type {AudioWorkletNode|null} */
    audioWorkletNode: null,
    /** @type {Int16Array[]} */
    recordedChunks: [],

    /** UI state */
    devPanelOpen: false,

    /** DOM element cache (populated in app.js init) */
    dom: {
        messagesScroll: null,
        messagesArea: null,
        welcomeScreen: null,
        chatInput: null,
        btnSend: null,
        btnMic: null,
        thinkingEl: null,
        thinkingLabel: null,
        statusIndicator: null,
        statusText: null,
        connectionBanner: null,
        devSidebar: null,
        btnDevToggle: null,
        toast: null,
    },
};

export default AppState;

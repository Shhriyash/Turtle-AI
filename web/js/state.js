/**
 * state.js — Global application state singleton
 *
 * All modules import from here instead of using globals.
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
    responsePanelOpen: false,

    /** DOM element cache (populated in app.js init) */
    dom: {
        chatInput: null,
        btnSend: null,
        statusIndicator: null,
        statusText: null,
        connectionBanner: null,
        devSidebar: null,
        btnDevToggle: null,
        toast: null,
        // Bubble
        bubbleOrb: null,
        bubbleGlow: null,
        bubbleStatus: null,
        // Response panel
        responsePanel: null,
        responseMessages: null,
        panelThinking: null,
        panelThinkingLabel: null,
        panelTiming: null,
    },
};

export default AppState;

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
    voiceMode: 'ptt',
    pttSpaceHeld: false,

    /** Audio recording state */
    /** @type {AudioContext|null} */
    audioContext: null,
    /** @type {AudioWorkletNode|null} */
    audioWorkletNode: null,
    /** @type {Int16Array[]} */
    recordedChunks: [],

    /** TTS playback state — a gapless scheduled queue over one persistent context */
    /** @type {AudioContext|null} */
    ttsAudioContext: null,
    /** @type {AudioBufferSourceNode[]} currently scheduled/playing sources */
    ttsSources: [],
    /** next absolute context time to schedule the following chunk at */
    ttsNextStartTime: 0,
    /** promise chain that decodes + schedules blobs strictly in arrival order */
    ttsDecodeChain: Promise.resolve(),
    /** generation counter — bumped on barge-in to invalidate in-flight decodes */
    ttsPlaybackGen: 0,
    isTtsPlaying: false,

    /** UI state */
    devPanelOpen: false,
    responsePanelOpen: false,

    /** DOM element cache (populated in app.js init) */
    dom: {
        chatInput: null,
        btnSend: null,
        btnVoiceMode: null,
        btnVoice: null,
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

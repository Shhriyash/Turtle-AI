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
    /**
     * P6-B1 (ledger 6.1): a message sent while a turn is still running. The
     * server queues it one deep (a newer one replaces it), so the client holds
     * exactly one: { text, el }. null when nothing is pending.
     */
    pendingTurn: null,

    /**
     * P6-B2 (ledger 6.8 / 6.4): resume protocol state.
     * clientId   -- per-tab id (sessionStorage), sent as ?cid= so a reconnecting
     *               tab is recognised by the server's session lease.
     * sessionId  -- the server session this tab is in (ready/restored/resumed).
     * lastTurnId -- the last `done.turn_id` of THAT session, sent in `resume`.
     * reconnectDelayMs -- next backoff step: 1 s doubling to 30 s, reset on open.
     * plannedReconnect -- the server announced `status: reconnect` (a deliberate
     *               1012 cut): no "Disconnected" banner for it.
     * resendQueue -- messages the server refused to start (`unstarted`), resent
     *               after the next resume frame.
     */
    clientId: null,
    sessionId: null,
    lastTurnId: null,
    hasConnected: false,
    reconnectDelayMs: 1000,
    reconnectTimer: null,
    plannedReconnect: false,
    resendQueue: [],

    isRecording: false,
    voiceMode: 'ptt',
    pttSpaceHeld: false,

    /** Streaming STT (Deepgram Flux) — advertised by the server's ready frame */
    streamSttEnabled: false,
    /** True while this recording is streaming frames to an open server mic session */
    micStreaming: false,

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
        btnChatToggle: null,
        panelThinking: null,
        panelThinkingLabel: null,
        panelTiming: null,
    },
};

export default AppState;

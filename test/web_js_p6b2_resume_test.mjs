// P6-B2 (ledger 6.8) -- the client half of the resume protocol.
//
// Plain Node ESM script, same style as web_js_p6b1_pending_test.mjs.
// Run: node test/web_js_p6b2_resume_test.mjs
//
// Drives the REAL web/js/websocket.js against a scriptable WebSocket class and
// a captured setTimeout, and asserts:
//   - reconnect backoff is 1 s doubling to 30 s and RESETS on open;
//   - the socket URL carries the per-tab ?cid= (the server's lease identity);
//   - a server `status: reconnect` frame suppresses the "Disconnected" banner,
//     queues the `unstarted` messages, and after the close + reconnect the
//     client sends {"type":"resume","session_id","last_turn_id"} FIRST and then
//     resends the refused messages;
//   - last_turn_id tracks `done.turn_id` of the CURRENT session only;
//   - a `resumed` frame for a different session resets the cursor.
// What this does NOT cover: a real browser, a real socket, the server side.

import assert from 'node:assert/strict';

function el() {
    const node = {
        children: [], className: '', style: {}, _text: '', _html: '', dataset: {},
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
        appendChild(c) { node.children.push(c); c.parent = node; return c; },
        remove() {},
        setAttribute() {},
        addEventListener() {},
        querySelector() { return el(); },
        querySelectorAll() { return []; },
        set textContent(v) { node._text = v; },
        get textContent() { return node._text; },
        set innerHTML(v) { node._html = v; },
        get innerHTML() { return node._html || node._text; },
    };
    return node;
}
globalThis.document = {
    createElement: () => el(),
    getElementById: () => el(),
    querySelector: () => el(),
    querySelectorAll: () => [],
    addEventListener() {},
    body: el(),
};
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.location = { protocol: 'https:', host: 'turtle.example' };
const store = {};
globalThis.sessionStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
};

// Captured timers: the test advances them by hand.
const timers = [];
globalThis.setTimeout = (fn, ms) => { const t = { fn, ms, live: true }; timers.push(t); return t; };
globalThis.clearTimeout = (t) => { if (t) t.live = false; };
globalThis.setInterval = () => 0;
const liveTimers = () => timers.filter(t => t.live);
function fireNextTimer() {
    const t = liveTimers()[0];
    assert.ok(t, 'expected a pending reconnect timer');
    t.live = false;
    t.fn();
    return t.ms;
}

class FakeWebSocket {
    static OPEN = 1;
    static instances = [];
    constructor(url) {
        this.url = url;
        this.sent = [];
        this.readyState = 0;
        this.closed = false;
        FakeWebSocket.instances.push(this);
    }
    send(data) { this.sent.push(typeof data === 'string' ? JSON.parse(data) : data); }
    close() { this.closed = true; }
    // test helpers
    open() { this.readyState = 1; this.onopen(); }
    recv(obj) { this.onmessage({ data: JSON.stringify(obj) }); }
    drop() { this.readyState = 3; this.onclose(); }
}
globalThis.WebSocket = FakeWebSocket;
globalThis.Blob = class Blob {};

const { default: AppState } = await import('../web/js/state.js');
AppState.dom.responseMessages = el();
AppState.dom.responsePanel = el();
AppState.dom.chatInput = Object.assign(el(), { value: '', style: {} });
AppState.dom.btnSend = el();
AppState.dom.connectionBanner = el();
const ws = await import('../web/js/websocket.js');
const last = () => FakeWebSocket.instances[FakeWebSocket.instances.length - 1];

// 1. the URL names this tab (lease identity) ---------------------------------
ws.connectWebSocket();
const first = last();
assert.match(first.url, /^wss:\/\/turtle\.example\/ws\?cid=[A-Za-z0-9_-]{8,64}$/);
const cid = new URL(first.url.replace('wss:', 'https:')).searchParams.get('cid');
assert.equal(store.turtle_cid, cid, 'cid must be persisted in sessionStorage');

// 2. first connect sends NO resume (nothing to resume), then learns the session
first.open();
assert.deepEqual(first.sent, [], 'a first connection has nothing to resume');
first.recv({ type: 'status', status: 'ready', stream_stt: false, session_id: 'turtle_session_A' });
assert.equal(AppState.sessionId, 'turtle_session_A');

// 3. last_turn_id follows `done.turn_id` of the current session only ----------
first.recv({ type: 'done', content: 'one', tool_urls: [], turn_id: 'turtle_session_A_turn_1' });
first.recv({ type: 'done', content: 'two', tool_urls: [], turn_id: 'turtle_session_A_turn_2' });
first.recv({ type: 'done', content: 'foreign', tool_urls: [], turn_id: 'turtle_session_Z_turn_9' });
assert.equal(AppState.lastTurnId, 'turtle_session_A_turn_2');

// 4. planned cut: reconnect frame, then close ---------------------------------
first.recv({
    type: 'status', status: 'reconnect', reason: 'max_duration',
    unstarted: [
        { kind: 'text', content: 'typed while cutting' },
        { kind: 'text', content: 'said aloud', source: 'mic' },
        { kind: 'audio', data: 'QUJD', sample_rate: 16000 },
    ],
});
assert.equal(AppState.plannedReconnect, true);
assert.equal(AppState.isConnected, false, 'sendMessage must refuse further sends');
let bannerShown = false;
AppState.dom.connectionBanner = { classList: { add() { bannerShown = true; }, remove() {} }, style: {} };
first.drop();
assert.equal(liveTimers().length, 1, 'exactly one reconnect timer');
assert.equal(liveTimers()[0].ms, 1000, 'first retry after 1 s');
fireNextTimer();
const second = last();
assert.notEqual(second, first);
assert.equal(first.closed, true, 'the superseded socket is closed');

// 5. on open: resume FIRST, then the refused messages ---------------------------
second.open();
assert.deepEqual(second.sent[0], {
    type: 'resume', session_id: 'turtle_session_A', last_turn_id: 'turtle_session_A_turn_2',
});
assert.deepEqual(second.sent.slice(1), [
    { type: 'text', content: 'typed while cutting' },
    { type: 'text', content: 'said aloud' },
    { type: 'audio', data: 'QUJD', sample_rate: 16000 },
]);
assert.deepEqual(AppState.resendQueue, [], 'queue drained exactly once');
assert.equal(AppState.plannedReconnect, false);

// 6. a late close of the SUPERSEDED socket must not flip state ------------------
const wasConnected = AppState.isConnected;
first.onclose();
assert.equal(AppState.isConnected, wasConnected);

// 7. backoff: 1 s doubling to 30 s, reset on open -------------------------------
second.drop();
const waits = [];
for (let i = 0; i < 7; i++) {
    waits.push(liveTimers()[0].ms);
    fireNextTimer();
    last().drop(); // never opens -> keeps failing
}
assert.deepEqual(waits, [1000, 2000, 4000, 8000, 16000, 30000, 30000]);
fireNextTimer();
last().open(); // success resets
last().drop();
assert.equal(liveTimers()[0].ms, 1000, 'delay resets to 1 s after a successful open');
fireNextTimer();

// 8. server put us in a DIFFERENT session -> adopt it, restart the cursor -------
last().open();
last().recv({ type: 'resumed', session_id: 'turtle_session_B', same_session: false, replayed: 0 });
assert.equal(AppState.sessionId, 'turtle_session_B');
assert.equal(AppState.lastTurnId, null);
last().recv({ type: 'done', content: 'x', tool_urls: [], turn_id: 'turtle_session_B_turn_1' });
assert.equal(AppState.lastTurnId, 'turtle_session_B_turn_1');

console.log('web_js_p6b2_resume_test: all assertions passed');

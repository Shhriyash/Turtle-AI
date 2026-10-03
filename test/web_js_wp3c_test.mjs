// WP3.C (ledger 3.6(d)) — client-side "degraded" status frame handling.
//
// Same convention as test/web_js_wp1h_test.mjs: no test harness/runner in
// this repo, so this is a plain, dependency-free Node ESM script
// (`node:assert/strict` only) that exercises the exported, DOM-adjacent
// piece of web/js/websocket.js directly.
//
// Phase 1's recurring defect was a control computed and shipped
// server-side but never read by the client (the untrusted-content
// allow-list, ISSUE fixed in WP1.H). apps/turtle_server.py now sends a
// {"type": "status", "status": "degraded", ...} frame when the session
// store couldn't be reached on connect (ledger 3.6(d)) — this test proves
// the client actually reads it, rather than falling through to the
// "render the literal word 'degraded'" default every other unmapped
// status would get.
//
// Run: node test/web_js_wp3c_test.mjs (exits non-zero on any failed
// assertion, via node:assert/strict throwing).

import assert from 'node:assert/strict';

// Minimal document stub: handleStatusMessage's setBubbleState() call chains
// into ambient.js's setAmbientState(), which looks up
// document.getElementById('bubble-container') and early-returns when it's
// missing -- exactly the "no real DOM in this connection state" case we
// want here (AppState.dom.bubbleStatus/panelThinking/toast are also left
// null below, exercising every "if (el)" guard along the way).
globalThis.document = {
    getElementById() {
        return null;
    },
};

const AppState = (await import('../web/js/state.js')).default;
const { handleStatusMessage } = await import('../web/js/websocket.js');

// --- labelMap must have a "degraded" entry --------------------------------
// Capture what setStatus was actually told to render by swapping in a fake
// statusText element (setStatus writes textContent on it directly).
const fakeStatusText = { textContent: '' };
AppState.dom.statusText = fakeStatusText;

handleStatusMessage({ status: 'degraded', reason: 'session_store_unavailable' });

assert.notEqual(
    fakeStatusText.textContent,
    'degraded',
    'expected a human label from labelMap, not the raw status literal ' +
        '("degraded" rendering as itself is exactly the "wired at one end ' +
        'only" defect this test guards against)'
);
assert.ok(
    fakeStatusText.textContent.length > 0,
    'expected a non-empty label to have been set'
);

// --- must not throw even with every optional DOM ref left null -----------
// (isThinking is flipped off, exactly like the 'ready'/'restored' branches)
assert.equal(AppState.isThinking, false);

// --- a toast is shown warning the user session state was not restored ----
let toastCalls = 0;
AppState.dom.toast = {
    set textContent(_v) { toastCalls += 1; },
    get textContent() { return ''; },
    style: {},
    classList: { add() {}, remove() {} },
};
handleStatusMessage({ status: 'degraded' });
assert.ok(toastCalls > 0, 'expected a toast to be shown for the degraded status');

console.log('web_js_wp3c_test.mjs: all assertions passed');

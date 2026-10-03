// P6-B1 (ledger 6.1) -- the client shows a message sent mid-turn as PENDING.
//
// Plain Node ESM script, same style as web_js_wp1h_test.mjs (no JS test runner
// exists in this repo). Run: node test/web_js_p6b1_pending_test.mjs
//
// Exercises web/js/chat.js against a minimal DOM stub:
//   - sending while a turn runs puts the message in the pending slot, NOT the
//     transcript, and sends it on the wire (the server queues it);
//   - a second send replaces the pending one (server queue is one deep);
//   - when the running turn's answer lands, the pending message moves into the
//     transcript AFTER that answer;
//   - interrupt clears it.
// What this does NOT cover: the real browser DOM/CSS, and websocket.js's frame
// dispatch (out of this package's write set).

import assert from 'node:assert/strict';

function el() {
    const node = {
        children: [], className: '', style: {}, _text: '', _html: '',
        classList: { add() {}, remove() {}, toggle() {} },
        appendChild(c) { node.children.push(c); c.parent = node; return c; },
        remove() { if (node.parent) node.parent.children = node.parent.children.filter(x => x !== node); },
        setAttribute() {},
        set textContent(v) { node._text = v; },
        get textContent() { return node._text; },
        set innerHTML(v) { node._html = v; },
        get innerHTML() { return node._html || node._text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
    };
    return node;
}
globalThis.document = { createElement: () => el() };
globalThis.requestAnimationFrame = (fn) => fn();

const { default: AppState } = await import('../web/js/state.js');
const chat = await import('../web/js/chat.js');

const sent = [];
AppState.ws = { send: (s) => sent.push(JSON.parse(s)) };
AppState.isConnected = true;
AppState.dom.responseMessages = el();
AppState.dom.responsePanel = el();
AppState.dom.chatInput = Object.assign(el(), { value: '', style: {} });
AppState.dom.btnSend = el();

const transcript = () => AppState.dom.responseMessages.children;
const roles = () => transcript().map(c => c.className);

function type(text) { AppState.dom.chatInput.value = text; chat.sendMessage(); }

// 1. idle: an ordinary send goes straight into the transcript.
type('first');
assert.equal(AppState.pendingTurn, null);
assert.equal(transcript().length, 1);
assert.equal(sent.length, 1);

// 2. the turn is running (server said "thinking"); a second send is PENDING.
chat.showThinking('Thinking');
type('second');
assert.ok(AppState.pendingTurn, 'second message must be pending');
assert.equal(AppState.pendingTurn.text, 'second');
assert.equal(transcript().length, 2, 'pending element sits below the transcript');
assert.ok(transcript()[1].className.includes('panel-msg-pending'));
assert.deepEqual(sent[1], { type: 'text', content: 'second' }, 'it is still sent: the server queues it');

// 3. a third replaces the pending one in place (never two pending).
type('third');
assert.equal(AppState.pendingTurn.text, 'third');
assert.equal(transcript().filter(c => c.className.includes('panel-msg-pending')).length, 1);
assert.equal(transcript()[1].contentEl.textContent, 'third');

// 4. the running turn's answer lands: pending is promoted AFTER the answer.
chat.hideThinking();
chat.addMessage('assistant', 'answer one', []);
assert.equal(AppState.pendingTurn, null);
assert.equal(transcript().length, 3);
assert.deepEqual(
    roles().map(c => c.replace('panel-msg ', '')),
    ['panel-msg-user', 'panel-msg-assistant', 'panel-msg-user'],
);

// 5. interrupt clears the pending message.
chat.showThinking('Thinking');
type('fourth');
assert.ok(AppState.pendingTurn);
const before = transcript().length;
chat.clearPendingTurn();
assert.equal(AppState.pendingTurn, null);
assert.equal(transcript().length, before - 1);

// 6. the running turn FAILED (error frame, no answer): the queued turn starting
//    announces 'Thinking' and promotes the pending message.
chat.hideThinking();
chat.showThinking('Thinking');
type('fifth');
assert.ok(AppState.pendingTurn);
chat.hideThinking();            // websocket.js 'error' case
chat.showThinking('Thinking');  // server started the queued turn
assert.equal(AppState.pendingTurn, null);

// 7. the running turn's own later 'Speaking' status must NOT promote it.
chat.hideThinking();
chat.showThinking('Thinking');
type('sixth');
chat.showThinking('Speaking');
assert.ok(AppState.pendingTurn, "'Speaking' belongs to the running turn");

console.log('web_js_p6b1_pending_test: all assertions passed');

// WP1.H (ledger 1b.2) — client-side allow-list defaulting.
//
// There is no JS test harness/runner in this repo (no package.json test
// script, no Jest/Mocha config) and this WP was told not to stand one up.
// This is a plain, dependency-free Node ESM script — `node:assert/strict`
// only, no framework — that exercises the two exported, DOM-independent
// pieces of web/js/chat.js's WP1.H logic directly:
//
//   - resolveToolUrlsForRole: the role-keyed fail-closed default that
//     decides, for a `done` frame that is missing `tool_urls` entirely
//     (e.g. the budget-refusal frame, which never ran a tool), whether the
//     assistant bubble renders nothing clickable (correct) or falls back to
//     the old unrestricted linkify-everything behaviour (the regression
//     this test exists to catch).
//   - formatMessage: the actual allow-list enforcement, run through the
//     same "missing tool_urls" shape.
//
// Run: node test/web_js_wp1h_test.mjs  (exits non-zero on any failed
// assertion, via node:assert/strict throwing).
//
// `formatMessage` calls the real `escapeHtml` from web/js/utils.js, which
// uses `document.createElement` to escape text — a minimal document stub is
// installed below (browsers only escape &, <, > when round-tripping
// textContent -> innerHTML with no attributes involved, so that's all this
// mimics).

import assert from 'node:assert/strict';

globalThis.document = {
    createElement() {
        let text = '';
        return {
            set textContent(v) { text = v; },
            get textContent() { return text; },
            get innerHTML() {
                return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            },
        };
    },
};

const { resolveToolUrlsForRole, formatMessage } = await import('../web/js/chat.js');

// ---------------------------------------------------------------------------
// resolveToolUrlsForRole — the fail-closed default itself.
// ---------------------------------------------------------------------------

// The exact regression this test guards: a `done` frame with NO `tool_urls`
// key (JS reads this as `undefined`, same as msg.tool_urls on the
// budget-refusal frame) must resolve to an empty allow-list for the
// assistant, not `undefined` (which formatMessage/isAllowedUrl treats as
// "allow everything").
assert.deepEqual(resolveToolUrlsForRole('assistant', undefined), []);
assert.deepEqual(resolveToolUrlsForRole('assistant', null), []);
assert.deepEqual(resolveToolUrlsForRole('assistant', 'not-an-array'), []);
assert.deepEqual(resolveToolUrlsForRole('assistant', ['https://example.com/a']), ['https://example.com/a']);

// The user's own typed/spoken message stays unrestricted regardless of what
// (if anything) is passed as toolUrls — it's trusted input, not tool output.
assert.equal(resolveToolUrlsForRole('user', undefined), undefined);
assert.equal(resolveToolUrlsForRole('user', ['https://example.com/a']), undefined);

// ---------------------------------------------------------------------------
// formatMessage — behaviour when actually rendering a `done` frame missing
// tool_urls (assistant role, allow-list resolved via the function above).
// ---------------------------------------------------------------------------

const modelWrittenUrl = 'https://evil.example/phish';
const assistantTextWithUrl = `click here: ${modelWrittenUrl}`;

// Frame WITHOUT tool_urls (undefined) → resolved to [] for the assistant →
// the bare URL must render as plain (escaped) text, NOT an anchor.
{
    const allowList = resolveToolUrlsForRole('assistant', undefined);
    const html = formatMessage(assistantTextWithUrl, allowList);
    assert.ok(!html.includes('<a '), `expected no anchor, got: ${html}`);
    assert.ok(html.includes(modelWrittenUrl), `expected the plain URL text to survive, got: ${html}`);
}

// Frame WITH tool_urls containing that exact URL → it DOES become an anchor,
// labelled with the host (not the raw URL).
{
    const allowList = resolveToolUrlsForRole('assistant', [modelWrittenUrl]);
    const html = formatMessage(assistantTextWithUrl, allowList);
    assert.ok(html.includes('<a '), `expected an anchor, got: ${html}`);
    assert.ok(html.includes('evil.example'), `expected host-labelled anchor text, got: ${html}`);
    assert.ok(!html.includes(`>${modelWrittenUrl}<`), `anchor text must be the host, not the raw URL: ${html}`);
}

// A markdown-style link the model wrote, pointing at a URL no tool
// returned, must not become clickable even though it has a custom label.
{
    const allowList = resolveToolUrlsForRole('assistant', []);
    const html = formatMessage('[trust me](https://evil.example/x)', allowList);
    assert.ok(!html.includes('<a '), `expected no anchor for a non-allow-listed markdown link, got: ${html}`);
    assert.ok(html.includes('trust me'), `expected the label text to survive as plain text, got: ${html}`);
}

// The user's own message is untouched by any of this — a URL they typed
// themselves still renders as a normal clickable link.
{
    const allowList = resolveToolUrlsForRole('user', undefined);
    const html = formatMessage(`check ${modelWrittenUrl}`, allowList);
    assert.ok(html.includes('<a '), `expected the user's own URL to stay clickable, got: ${html}`);
}

console.log('web_js_wp1h_test.mjs: all assertions passed');

/**
 * calendar.js — Google Calendar connect-button UI module
 *
 * The header calendar icon has three states:
 *   - Not connected: click opens /integrations/google_calendar/connect in a
 *     new tab, which walks the user through Google's consent screen.
 *   - Connected, current scope: clicking the icon should NOT re-run the
 *     OAuth consent screen every time (that used to happen because the
 *     button always linked straight to /connect) — instead it opens the
 *     user's actual Google Calendar, and the icon itself shows a connected
 *     indicator so the state is visible without clicking at all.
 *   - Connected, but scope_stale (WP1.E2 / ledger 1b.3): the stored token
 *     was minted under the old, broader calendar scope. Rather than fail
 *     silently the next time a calendar tool call needs a permission it no
 *     longer has, the button shows a "Reconnect Calendar" prompt and its
 *     click goes back through /connect (re-consent), same as "not connected".
 */

const STATUS_URL = '/integrations/google_calendar/status';
const GOOGLE_CALENDAR_URL = 'https://calendar.google.com/calendar/u/0/r';

let headerBtn = null;
let connected = false;
let scopeStale = false;

function applyConnectedUi() {
    if (!headerBtn) return;
    const needsReconnect = connected && scopeStale;
    headerBtn.classList.toggle('connected', connected && !scopeStale);
    headerBtn.classList.toggle('reconnect-needed', needsReconnect);
    if (needsReconnect) {
        headerBtn.title = 'Reconnect Calendar — Turtle needs you to re-approve with an updated permission scope';
    } else {
        headerBtn.title = connected
            ? 'Google Calendar connected — click to open your calendar'
            : 'Connect Google Calendar';
    }
    headerBtn.setAttribute('aria-label', headerBtn.title);
}

async function refreshStatus() {
    try {
        const resp = await fetch(STATUS_URL, { credentials: 'same-origin' });
        if (!resp.ok) {
            // 401 = not signed in yet; leave the default "not connected" UI as-is.
            return;
        }
        const data = await resp.json();
        connected = !!data.connected;
        scopeStale = !!data.scope_stale;
        applyConnectedUi();
    } catch {
        // Network hiccup checking status — not worth surfacing, the button
        // still works via its default href either way.
    }
}

function onHeaderClick(event) {
    if (!connected || scopeStale) {
        // Not connected, or connected under a now-stale scope: let the
        // anchor's default href do its job (navigates
        // /integrations/google_calendar/connect in a new tab, which
        // re-consents under the current scope either way).
        return;
    }
    // Already connected with the current scope: never re-send the user
    // through Google's consent screen again. Open their actual calendar
    // instead.
    event.preventDefault();
    window.open(GOOGLE_CALENDAR_URL, '_blank', 'noopener,noreferrer');
}

export function initCalendarUI() {
    headerBtn = document.getElementById('btn-calendar-connect');
    if (!headerBtn) return;

    headerBtn.addEventListener('click', onHeaderClick);
    refreshStatus();

    // Re-check when the tab regains focus — covers the common flow of
    // connecting in the new OAuth tab, closing it, and coming back here.
    window.addEventListener('focus', refreshStatus);
}

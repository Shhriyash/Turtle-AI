/**
 * calendar.js — Google Calendar connect-button UI module
 *
 * The header calendar icon has two states:
 *   - Not connected: click opens /integrations/google_calendar/connect in a
 *     new tab, which walks the user through Google's consent screen.
 *   - Connected: clicking the icon should NOT re-run the OAuth consent
 *     screen every time (that used to happen because the button always
 *     linked straight to /connect) — instead it opens the user's actual
 *     Google Calendar, and the icon itself shows a connected indicator so
 *     the state is visible without clicking at all.
 */

const STATUS_URL = '/integrations/google_calendar/status';
const GOOGLE_CALENDAR_URL = 'https://calendar.google.com/calendar/u/0/r';

let headerBtn = null;
let connected = false;

function applyConnectedUi() {
    if (!headerBtn) return;
    headerBtn.classList.toggle('connected', connected);
    headerBtn.title = connected
        ? 'Google Calendar connected — click to open your calendar'
        : 'Connect Google Calendar';
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
        applyConnectedUi();
    } catch {
        // Network hiccup checking status — not worth surfacing, the button
        // still works via its default href either way.
    }
}

function onHeaderClick(event) {
    if (!connected) {
        // Not connected: let the anchor's default href do its job
        // (navigates /integrations/google_calendar/connect in a new tab).
        return;
    }
    // Already connected: never re-send the user through Google's consent
    // screen again. Open their actual calendar instead.
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

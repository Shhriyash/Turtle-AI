# link_account

## Purpose

Start linking the user's CURRENT channel identity (e.g. their Discord account) to their existing Turtle web account, so both surfaces share one memory instead of two separate ones.

## When to USE

When the user asks to connect, link, merge, or sync their accounts — for example:

- "link my account"
- "connect this to my web account"
- "I already use Turtle on the web, can you merge them?"
- "why don't you remember what I told you on the website?"

## When NOT to USE

- On the web surface — the user is already signed in there, so there is nothing to link.
- To look up or change an email address (that is `remember`).
- Do **not** treat the email the user gives you here as proof of anything, and do not tell the user it links their account. It does not. Turtle cannot verify a self-claimed email — it is used only to let the redemption step refuse a code used by the wrong account. The authenticated web sign-in is still the only thing that actually proves ownership. This tool exists precisely so linking never happens on a self-claimed email alone.

## Parameters

- `expected_email` (required): the email address of the Turtle web account the user says they'll sign in with to finish linking. Ask for it if the user hasn't given it — do not guess it or reuse an email from earlier in the conversation without confirming it's the right one.

## Return

A short-lived, single-use claim code plus instructions. Relay the code to the user **exactly as given** and tell them to sign in to Turtle on the web **as the email they gave you** and enter it there. Signing in is the step that proves the web account belongs to them; the email only narrows who is allowed to try.

Do not promise the accounts are linked yet — they are linked only after the user redeems the code on the web, and only if they sign in as the account they named.

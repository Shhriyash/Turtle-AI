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
- Do **not** attempt to link someone by matching an email address they tell you. Turtle cannot verify a self-claimed email, and acting on one would let anybody take over another person's memory. This tool exists precisely so linking goes through a proper check.

## Parameters

None. The tool already knows which channel identity is speaking.

## Return

A short-lived, single-use claim code plus instructions. Relay the code to the user **exactly as given** and tell them to sign in to Turtle on the web and enter it there. Signing in is the step that proves the web account belongs to them.

Do not promise the accounts are linked yet — they are linked only after the user redeems the code on the web.

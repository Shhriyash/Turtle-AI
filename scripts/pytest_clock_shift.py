"""Run the test suite as if it were N days in the future, to surface date rot.

Why this exists
---------------
``core.memory_schema.is_decayed`` drops any non-exempt fact older than
``DECAY_DAYS`` (30). A test fixture pinned to an absolute date therefore passes
on the day it is written and starts failing about a month later, silently, for
a reason that looks nothing like whatever change happened to expose it.
``test/phase2_read_model_test.py`` rotted exactly that way and sat red in CI
for weeks before anyone connected the failure to the calendar.

This plugin makes that class of bug reproducible on demand instead of waiting
for real time to pass.

How it works
------------
It replaces ``datetime.datetime`` with a subclass whose ``now()`` is offset by
``SHIFT_DAYS``. The swap happens at plugin-load time, before pytest imports any
test module or any ``core`` module, so modules doing ``from datetime import
datetime`` pick up the shim as well. That is what makes the simulation
faithful:

* fixtures built from ``datetime.now()`` move with the shifted clock and keep
  passing, exactly as they do as real time passes;
* fixtures pinned to an absolute date stay put and age out.

A failure under this plugin is therefore genuine rot, not an artifact.

The subclass must return itself
-------------------------------
``now()`` deliberately re-wraps its result into the subclass. An earlier
version returned a plain ``datetime`` and every JWT test blew up with
"Object of type datetime is not JSON serializable": PyJWT had bound the shim
as *its* ``datetime`` and its ``isinstance(value, datetime)`` check then failed
for a base-class instance, so it never converted ``exp`` to a timestamp.

The tempting fix, importing PyJWT before the patch so it keeps the real class,
is worse than the problem. It leaves the token minted on the shifted clock and
validated on the real one, so ``test_verify_rejects_expired_token`` fails as a
false positive. Keeping one clock for everyone is what makes the run clean.

Usage
-----
PowerShell (the shell this repo is normally driven from)::

    $env:PYTHONPATH="scripts"; $env:SHIFT_DAYS="400"; pytest test -q -p pytest_clock_shift

bash/zsh::

    PYTHONPATH=scripts SHIFT_DAYS=400 pytest test -q -p pytest_clock_shift

Run it at +0 first as a control: that must be green, since nothing has moved.
Then push it out past ``DECAY_DAYS``.

If something fails, fix the fixture, not this plugin. Three patterns are safe,
and the note beside ``DECAY_DAYS`` in ``core/memory_schema.py`` lists them:
use a decay-exempt topic/source, pin ``reference_time`` alongside the fixed
dates, or build the timestamps relative to now.
"""
from __future__ import annotations

import datetime as _datetime_module
import os
from datetime import timedelta

_REAL = _datetime_module.datetime
SHIFT = timedelta(days=float(os.environ.get("SHIFT_DAYS", "90")))


class _ShiftedDatetime(_REAL):  # type: ignore[misc,valid-type]
    """``datetime`` whose idea of "now" sits SHIFT into the future.

    Every constructor path returns this subclass, so libraries that type-check
    with ``isinstance`` against the name they imported still recognise it.
    """

    @classmethod
    def _wrap(cls, value):
        return cls(
            value.year, value.month, value.day,
            value.hour, value.minute, value.second, value.microsecond,
            value.tzinfo, fold=value.fold,
        )

    @classmethod
    def now(cls, tz=None):
        return cls._wrap(_REAL.now(tz) + SHIFT)

    @classmethod
    def utcnow(cls):
        return cls._wrap(_REAL.utcnow() + SHIFT)

    @classmethod
    def today(cls):
        return cls._wrap(_REAL.today() + SHIFT)


_datetime_module.datetime = _ShiftedDatetime


def pytest_report_header(config):
    days = SHIFT.days
    if not days:
        return "clock-shift: +0 days (control run, everything should pass)"
    return f"clock-shift: +{days} days, hunting fixtures pinned to absolute dates"

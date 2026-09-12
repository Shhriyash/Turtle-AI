"""
main.py
-------
Vercel Python runtime entrypoint (Vercel migration Phase 5).

Vercel's zero-config Python framework detection looks for an entrypoint file
named app.py/index.py/server.py/main.py/wsgi.py/asgi.py (or the same names
under src/ or app/) that defines a top-level `app` — see
https://vercel.com/docs/functions/runtimes/python. The real FastAPI app
lives in apps/turtle_server.py (module path apps.turtle_server:app — the
same path the Dockerfile's gunicorn CMD already uses for local/Docker
deploys), which isn't one of those magic filenames, so this thin re-export
is all Vercel needs to find it. Local dev and Docker are unaffected — they
still run apps/turtle_server.py directly (`python apps/turtle_server.py` or
the gunicorn CMD), never this file.
"""
from apps.turtle_server import app

__all__ = ["app"]

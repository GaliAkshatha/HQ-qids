"""
wsgi.py

Production WSGI entrypoint. Separate from src/api/app.py deliberately:
tests import create_app() from that module directly and construct their
own app instances -- if this file's `app = create_app()` lived in
src/api/app.py itself at module level, every test that imports
create_app would trigger a SECOND real ExperimentService construction
(real ML/quantum artifact loading) as a side effect of the import alone.

Production start command:

    gunicorn wsgi:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT

--threads 4 (not just --workers) matters here: the SSE endpoint
(/api/events) holds a connection open for its whole polling loop, and a
purely worker-based sync gunicorn config would let a handful of open SSE
connections exhaust all workers and stall ordinary requests. Using the
threaded worker model (implied by --threads) avoids that.
"""

from src.api.app import create_app

app = create_app()

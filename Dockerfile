FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies using uv for faster builds
RUN pip install uv
RUN uv pip install --system --no-cache -r requirements.txt

# Ensure new dependencies introduced in Tier 2 are also installed
RUN uv pip install --system --no-cache aiosqlite opentelemetry-sdk opentelemetry-exporter-otlp pydantic-settings pyjwt gunicorn

COPY . .

ENV PYTHONPATH=/app
ENV TURTLE_DEPLOY=cloud
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

VOLUME /app/data

EXPOSE 8000

# Liveness probe against /healthz. python:3.13-slim has no curl, so use the
# stdlib urllib client (exit 0 on HTTP 200, non-zero otherwise).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "apps.turtle_server:app"]

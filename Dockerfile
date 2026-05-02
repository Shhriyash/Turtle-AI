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

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "apps.turtle_server:app"]

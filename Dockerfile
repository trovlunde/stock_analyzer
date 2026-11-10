FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

RUN useradd -m -u 1000 appuser

WORKDIR /app

RUN mkdir -p /app/models && chown -R appuser:appuser /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

VOLUME ["/app/models"]

# Healthchecks live per service in docker-compose.yml — the image runs both the
# web app and the trader, so a baked-in HTTP check would fail for the trader.

CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "stock_analysis.web.app:app"]

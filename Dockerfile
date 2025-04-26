FROM python:3.11.12-slim-bookworm AS base

FROM base AS builder

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

COPY --link pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry install --no-ansi --no-interaction --no-root

FROM base AS runner

WORKDIR /app

COPY --link --from=builder /app/.venv ./.venv
COPY --link ./ml_backend ./ml_backend

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["gunicorn", "-b", ":8000", "ml_backend.main:app"]

EXPOSE 8000

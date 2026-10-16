# Read-only scientific documentation MCP server over stdio.
# Build:  docker build -t omd-mcp .
# Run:    docker run -i --rm --read-only --memory=512m omd-mcp
FROM python:3.12-slim AS build
ARG INCLUDE_PYTORCH=0

# uv resolves the pinned dependency set from uv.lock.
RUN pip install --no-cache-dir uv

WORKDIR /app
ENV DOCS_MCP_MODEL_CACHE=/app/model-cache HF_HUB_DISABLE_TELEMETRY=1

# Dependency layer first: cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --frozen --no-install-project

# Build a fresh index from the catalog and bundled sources. PyTorch docs are
# opt-in; MOM6 and grib2io are fetched from their pinned official sources.
# The weather lock lets fetch_weather skip
# re-acquiring any collection already present in the build context; a clean
# clone has no corpus/mom6, so there the crawl still runs.
COPY corpus.catalog.json weather-lock.json ./
COPY src/ ./src/
COPY corpus/ ./corpus/
RUN if [ "$INCLUDE_PYTORCH" = "1" ]; then python src/fetch_pytorch.py; fi \
    && python src/fetch_weather.py --only mom6,grib2io \
    && python src/ingest.py \
    && python src/build_embeddings.py --cache-dir /app/model-cache

FROM python:3.12-slim AS runtime
RUN pip install --no-cache-dir uv \
    && useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --frozen --no-install-project
COPY --from=build --chown=appuser:appuser /app/data/nuopc.sqlite3 data/nuopc.sqlite3
COPY --from=build --chown=appuser:appuser /app/model-cache /app/model-cache
COPY --chown=appuser:appuser src/server.py src/knowledge.py ./src/
# Compatibility shim: existing MCP client configs launch `python server.py`
# from /app. Keep that path working after the modules moved under src/.
RUN echo 'import sys; sys.path.insert(0, "/app/src")' > server.py \
 && echo 'from server import mcp' >> server.py \
 && echo 'if __name__ == "__main__": mcp.run(transport="stdio")' >> server.py

# DOCS_MCP_DB is already the ROOT/data/nuopc.sqlite3 default in server.py;
# set it explicitly so a bind-mount override is obvious.
ENV DOCS_MCP_DB=/app/data/nuopc.sqlite3 \
    DOCS_MCP_MODEL_CACHE=/app/model-cache \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
ENTRYPOINT ["python", "src/server.py"]

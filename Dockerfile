FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY app.py ./
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN useradd --create-home --uid 10001 vision && mkdir -p /workspace && chown vision /workspace
ENV PATH="/app/.venv/bin:$PATH" VISION_WORKSPACE=/workspace PYTHONUNBUFFERED=1
USER vision
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

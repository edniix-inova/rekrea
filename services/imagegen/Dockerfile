# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# Optional PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu126 for a CUDA 12.6 build.
# Empty (default) uses the default torch wheel from PyPI.
ARG TORCH_INDEX_URL=""

WORKDIR /app
COPY requirements.txt .
# torch and its CUDA libraries are several GB. The cache mount keeps downloaded wheels
# between builds, so a dropped connection does not restart everything from zero.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --retries 10 --timeout 60 torch --index-url "$TORCH_INDEX_URL"; \
    fi && \
    pip install --retries 10 --timeout 60 -r requirements.txt
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

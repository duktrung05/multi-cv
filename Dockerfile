FROM python:3.12-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    APP_HOST=0.0.0.0 \
    APP_PORT=8126 \
    MODEL_DEVICE=cpu

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/outputs /app/ckpts /app/data \
    && chown -R appuser:appuser /app/outputs /app/ckpts /app/data
USER appuser

EXPOSE 8126
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('APP_PORT','8126')+'/health',timeout=4)"
CMD ["python", "main.py"]

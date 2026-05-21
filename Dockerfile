FROM python:3.10-slim

# Jetson (ARM64) / x86 両対応
WORKDIR /app

# 依存パッケージ
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    numpy==1.26.4

# アプリとbankをコピー
COPY api_server.py .
COPY motif_bank.json .

# デフォルトポート
EXPOSE 8000

# 起動
CMD ["python3", "api_server.py", \
     "--bank", "/app/motif_bank.json", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

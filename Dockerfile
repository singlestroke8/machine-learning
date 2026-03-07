# 1. ベースイメージの指定
FROM python:3.13-slim

# 2. コンテナ内の作業ディレクトリ
WORKDIR /workspace

# ★ここを追加！LightGBMを動かすためのシステムライブラリ（libgomp1）をOSにインストール
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 必要なライブラリの設計図を先にコピー
COPY requirements.txt .

# 4. ライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt

# 5. 本番稼働に必要なフォルダだけをコンテナ内にコピー
COPY ./app ./app
COPY ./models ./models
COPY ./src ./src

# 6. ポート開放
EXPOSE 8000

# 7. 起動コマンド
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
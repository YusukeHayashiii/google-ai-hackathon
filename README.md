# Google AI Hackathon Project

## 目次
- [概要](#概要)
- [技術スタック](#技術スタック)
- [主要コンポーネント](#主要コンポーネント)
- [セットアップ](#セットアップ)
  - [環境変数](#環境変数)
  - [依存関係のインストール](#依存関係のインストール)
- [使用方法](#使用方法)
- [主な機能](#主な機能)
- [開発者向け情報](#開発者向け情報)
- [注意事項](#注意事項)
- [補足](#補足)

## 概要
このプロジェクトは、Google AI Hackathonのために作成されたものです。歴史に関するQ&Aチャットボットを実装しています。

## 技術スタック
- Python 3.12
- Streamlit
- Langchain
- Google Cloud AI Platform
- Neo4j
- Docker

## 主要コンポーネント
- `app.py`: メインのStreamlitアプリケーション
- `make_agent.py`: チャットボットエージェントのセットアップ
- `get_wiki_article.py`: 指定された日本の歴史的人物に関するWikipedia記事をダウンロードし、テキストファイルとして保存する
- `make_graph.py`: テキストデータからグラフデータベースを構築し、クエリや検索を行う

## セットアップ

### 環境変数
以下の環境変数を`.env`ファイルに設定してください：
- `COMPOSE_PROJECT_NAME`: プロジェクト名
- `PORTS_NUM_ST1`: Streamlitのポート番号
- `NEO4J_URI`: Neo4jデータベースのURI
- `NEO4J_USERNAME`: Neo4jデータベースのユーザー名
- `NEO4J_PASSWORD`: Neo4jデータベースのパスワード
- `AURA_INSTANCEID`: AuraDBのインスタンスID
- `AURA_INSTANCENAME`: AuraDBのインスタンス名
- `PROJECT_ID`: Google Cloud プロジェクトID
- `REGION`: Google Cloud リージョン
- `OPENAI_API_KEY`: OpenAI APIキー（必要な場合）
- `STAGING_BUCKET`: ステージングバケット名

### Dockerfile

ryeを使用する想定で書いていますが、以下のような構成にするとpipとrequirements.txtを使ってコンテナを作成できます。

```Dockerfile
# ベースイメージの指定
FROM python:3.12 AS builder

# Google Cloud CLIのバージョンを環境変数として定義
ENV GCLOUD_VERSION=477.0.0

# システムパッケージのアップデート
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    git \
    curl \
    fonts-ipaexfont \
    fonts-noto-cjk \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# pythonライブラリのインストール
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Google Cloud CLIをダウンロードしてインストール
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
    tar -xzf google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
    ./google-cloud-sdk/install.sh --quiet

# 実行環境
FROM python:3.12-slim

# 必要なパッケージのみをコピー
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /google-cloud-sdk /google-cloud-sdk

# フォントのコピー
COPY --from=builder /usr/share/fonts/opentype/ipaexfont-gothic /usr/share/fonts/opentype/ipaexfont-gothic
COPY --from=builder /usr/share/fonts/opentype/noto /usr/share/fonts/opentype/noto

# matplotlibの設定ファイルのコピー
COPY matplotlibrc /root/.config/matplotlib/matplotlibrc

# PATH を更新
ENV PATH $PATH:/google-cloud-sdk/bin

# 作業ディレクトリの設定
WORKDIR /workspace

# Open ports for Streamlit
EXPOSE 8501

CMD ["/bin/bash"]
```

### コンテナ作成

vscodeの場合はdevcontainerを使って作成できます。

コマンドの場合は.devcontainerに移動してから以下を実行してください。
```bash
docker compose up -d
```

### 依存関係のインストール

ryeを使用している場合は以下
```bash
rye sync
```
pip installの場合は以下
```bash
pip install -r requirements.lock
```

## 使用方法
1. 環境変数を設定
2. 依存関係をインストール
3. 以下のコマンドでアプリケーションを起動：
```bash
streamlit run app.py
```

## 主な機能
- 織田信長、豊臣秀吉、徳川家康、明智光秀に関する質問に答えるチャットボット
- Neo4jベクトルデータベースを使用した効率的な情報検索
- Google Cloud AI Platformを活用した高度な自然言語処理

## 開発者向け情報
- `pyproject.toml`にプロジェクトの設定が記述されています
- `rye`を使用してパッケージ管理を行っています
- VSCodeの開発コンテナ設定が含まれています

## 注意事項
- このプロジェクトはハッカソン用に開発されたものであり、本番環境での使用には適していない可能性があります
- APIキーや認証情報の取り扱いには十分注意してください

## 補足
- ナレッジグラフはすでに作成されているので、`make_agent.py`と`app.py`でチャットボットを起動できます
- ナレッジグラフを一から作る場合は`make_graph.py`で作成できます
- ナレッジグラフを作成するためのテキストデータは`get_wiki_articles.py`を実行し、`input/`配下に入れてください

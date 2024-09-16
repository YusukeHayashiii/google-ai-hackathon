# # ベースイメージの指定
# FROM python:3.12 AS builder

# # Google Cloud CLIのバージョンを環境変数として定義
# ENV GCLOUD_VERSION=477.0.0

# # システムパッケージのアップデート
# RUN apt-get update -y \
#     && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     fonts-ipaexfont \
#     fonts-noto-cjk \
#     && apt-get autoremove -y \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# # pythonライブラリのインストール
# COPY requirements.txt .
# RUN pip3 install --upgrade pip \
#     && pip3 install --no-cache-dir -r requirements.txt

# # Google Cloud CLIをダウンロードしてインストール
# RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
#     tar -xzf google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
#     ./google-cloud-sdk/install.sh --quiet

# # 実行環境
# FROM python:3.12-slim

# # 必要なパッケージのみをコピー
# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# COPY --from=builder /google-cloud-sdk /google-cloud-sdk

# # フォントのコピー
# COPY --from=builder /usr/share/fonts/opentype/ipaexfont-gothic /usr/share/fonts/opentype/ipaexfont-gothic
# COPY --from=builder /usr/share/fonts/opentype/noto /usr/share/fonts/opentype/noto

# # matplotlibの設定ファイルのコピー
# COPY matplotlibrc /root/.config/matplotlib/matplotlibrc

# # PATH を更新
# ENV PATH $PATH:/google-cloud-sdk/bin

# # 作業ディレクトリの設定
# WORKDIR /workspace

# # Open ports for Streamlit
# EXPOSE 8501


# ベースイメージの指定
FROM python:3.12-slim

# Google Cloud CLIのバージョンとRyeのバージョンを環境変数として定義
ENV GCLOUD_VERSION=477.0.0
ENV RYE_VERSION=0.28.0

# システムパッケージのアップデートと必要なツールのインストール
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    git \
    curl \
    fonts-ipaexfont \
    fonts-noto-cjk \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Ryeのインストール
RUN curl -sSf https://rye.astral.sh/get | RYE_VERSION=${RYE_VERSION} RYE_INSTALL_OPTION="--yes" bash
ENV PATH="/root/.rye/shims:${PATH}"

# Google Cloud CLIをダウンロードしてインストール
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
    tar -xzf google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz && \
    ./google-cloud-sdk/install.sh --quiet && \
    rm google-cloud-cli-${GCLOUD_VERSION}-linux-arm.tar.gz

# matplotlibの設定ファイルのコピー
COPY matplotlibrc /root/.config/matplotlib/matplotlibrc

# pyproject.tomlとrequirements.lockをコピー
# COPY ../pyproject.toml ../requirements.lock ./

# PATH を更新
ENV PATH="${PATH}:/google-cloud-sdk/bin"

# 作業ディレクトリの設定
WORKDIR /workspace

# Open ports for Streamlit
EXPOSE 8501

# Ryeプロジェクトの初期化と依存関係のインストール
# RUN rye sync

# エントリーポイントの設定
CMD ["/bin/bash"]
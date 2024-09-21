import streamlit as st
import subprocess
import os

def run_auth_process():
    # Google Cloud SDKのgcloud authコマンドを実行
    command = ["gcloud", "auth", "application-default", "login", "--no-launch-browser"]
    
    # 環境変数を設定してサブプロセスがStreamlitアプリと同じ端末にアクセスできるようにする
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # プロセスを開始
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    # Streamlitに表示するためのプレースホルダー
    output_placeholder = st.empty()
    
    # 認証URLを格納する変数
    auth_url = None
    
    # 出力を格納する変数
    output_text = ""
    
    # 出力を行ごとに読み取り
    for line in process.stdout:
        # 認証URLを含む行を探す
        if "Go to the following link in your browser" in line:
            # 次の行がURLになるはず
            url_line = next(process.stdout, None)
            if url_line:
                auth_url = url_line.strip()
                st.markdown(f"**認証URL:** [ここをクリックして認証を行ってください]({auth_url})")
        
        # 出力テキストを更新
        output_text += line
        # プレースホルダーの内容を更新
        output_placeholder.text(output_text)
        
        # 検証コードの入力を求めるプロンプトが表示されたら
        if "Enter the authorization code:" in line:
            verification_code = st.text_input("検証コードを入力してください:")
            if verification_code:
                # 検証コードをプロセスに送信
                process.stdin.write(verification_code + '\n')
                process.stdin.flush()
    
    # プロセスが終了するのを待つ
    process.wait()

def main():
    st.title("Google Cloud認証")
    
    if st.button("認証プロセスを開始"):
        run_auth_process()

if __name__ == "__main__":
    main()
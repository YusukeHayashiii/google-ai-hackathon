import sys
from pathlib import Path
import subprocess
# appのディレクトリをパスに追加
current_dir = Path(__file__).absolute()
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st

import src.make_agent as make_agent
import src.auth as auth
import importlib
importlib.reload(make_agent)
importlib.reload(auth)

def main():
    st.title("歴史Q&Aチャットボット")
    st.info("織田信長、豊臣秀吉、徳川家康、明智光秀について詳しいエージェントが質問に答えます！")
    
    # セッションステートの初期化
    if "is_login" not in st.session_state:
        st.session_state["is_login"] = False
        
    # ログインボタン
    if st.button("Google Cloudにログイン"):
        auth.run_auth_process()
        
    if st.sidebar.button("チャット履歴をリセット"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
        
    # 以降はログインしている場合のみ実行
    if st.session_state["is_login"]:
        # チャット履歴のメモリを作成
        chat_history = StreamlitChatMessageHistory(key="chat_messages")

        # エージェントのセットアップ
        @st.cache_resource
        def get_agent():
            return make_agent.setup_agent()
        agent_executor = get_agent()

        # チャット履歴を表示
        for chat in chat_history.messages:
            st.chat_message(chat.type).write(chat.content)

        # チャットの表示と入力
        if prompt := st.chat_input():
            st.chat_message("user").write(prompt) # ユーザーの入力を表示
            with st.chat_message("assistant"):
                # 途中経過の表示
                st_callback = StreamlitCallbackHandler(st.container())
                # Agentの実行
                try:
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": chat_history.messages},
                        {"callbacks": [st_callback]}
                    )
                    # エージェントの回答を表示
                    # st.write(response["output"])
                    output = make_agent.extract_final_answer(response["output"])
                    st.write(output)
                    # チャット履歴を更新
                    chat_history.add_messages(
                        [
                            HumanMessage(content=prompt),
                            AIMessage(content=output),
                        ]
                    )
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    st.error("エージェントの処理中にエラーが発生しました。もう一度お試しください。")
                
if __name__ == "__main__":
    main()
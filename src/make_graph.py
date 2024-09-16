# %%
import os
from dotenv import load_dotenv

from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
import vertexai
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

dotenv_path = '../.devcontainer/.env'
load_dotenv(dotenv_path)


# %%
# テキストの用意
input_dir = "../input"
files = os.listdir(input_dir)
# ファイルの内容を全て読み込み、一つに統合する
combined_text = ""
for file in files:
    file_path = os.path.join(input_dir, file)
    with open(file_path, "r", encoding="utf-8") as f:
        combined_text += f.read() + "\n\n"

# 統合されたテキストを表示
print(combined_text)


# %%
# テキストのチャンク分割 
def split_text(list_text: list, chunk_size=700, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents(list_text)
    return chunks

tgt_chunks = split_text([combined_text])


# %%
# データベース接続
graph = Neo4jGraph()

# グラフ構築用のLLMを定義
vertexai.init(
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("REGION"),
    staging_bucket=os.getenv("STAGING_BUCKET"),
)
llm = ChatVertexAI(
    model_name="gemini-1.5-flash-001",
    temperature=0.0,
)
llm_transformer = LLMGraphTransformer(llm=llm)

# %%
# グラフ構築
graph_documents = llm_transformer.convert_to_graph_documents(tgt_chunks)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")
# %%
# グラフを保存
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True,
)
# %%
# グラフを表示する
default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 100"

def showGraph(cypher: str = default_cypher):
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],  
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph()) 
    widget.node_label_mapping = 'id'
    return widget

# %%
showGraph()


# -------- ここからはいったん書いているだけのコード --------
# %%
# クエリで検索してみる
graph = Neo4jGraph()
llm_query = ChatVertexAI(
    model_name="gemini-1.5-pro-001",
    temperature=0.0,
)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm_query, verbose=True)

result = chain.invoke({"query": "信長と秀吉の関係は？"})
print(result)

# %%
# ベクトルインデックスを作る
vector_index = Neo4jVector.from_existing_graph(
    VertexAIEmbeddings("text-embedding-004"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"], 
    embedding_node_property="embedding"
)  
# %%
# ベクトル検索を試す
# response = vector_index.similarity_search(
#     "信長と秀吉の関係は？"
# )
response = vector_index.similarity_search("明智光秀の晩年はどのような過ごし方ですか？", k=3)
for res in response:
    print(res.page_content)

# %% 
# ベクトル検索を試す2
retriever = vector_index.as_retriever()
retriever.invoke("信長と秀吉の関係は？")
# %%
# RetrievalQAオブジェクトを作成
# vector_qa = RetrievalQA.from_chain_type(
#     llm=ChatVertexAI(
#         model_name="gemini-1.5-pro-001",
#         temperature=0.0,
#     ),
#     chain_type="stuff",
#     retriever=vector_index.as_retriever()
# )

# # %%
# # 質問を実行
# vector_qa.run(
#     "明智光秀の晩年はどのような過ごし方ですか？"
# )
# %%
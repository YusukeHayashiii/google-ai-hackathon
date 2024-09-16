# %%
import os
import re
from dotenv import load_dotenv
from typing import Callable, Sequence

# from langchain import hub
from langchain.vectorstores import Neo4jVector
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import GraphCypherQAChain
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.tools.base import StructuredTool
from langchain_core import prompts
from langchain_community.graphs import Neo4jGraph
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import vertexai

dotenv_path = '/workspace/.devcontainer/.env'
load_dotenv(dotenv_path)
# %%
# クラスを定義する
class VectorIndexTool:
    """ベクトルインデックスのツールのクラス"""    
    def __init__(self, vector_index):
        self.vector_index = vector_index
        self.name = "Vector Index Search"
        self.description = "Use Vector Index to search: you can search for fine text information in the graph database by searching for Document nodes."

    def output_name(self):
        return self.name
    
    def output_description(self):
        return self.description

    def search(self, query: str) -> str:
        """
        Use Vector Index to search: you can search for fine text information in the graph database by searching for Document nodes.
        """
        results = self.vector_index.similarity_search(query, k=3)
        if results:
            results_text = ""
            for res in results:
                results_text += res.page_content
            return results_text
        return "関連する情報が見つかりませんでした。"


class GraphQATool:
    """グラフQAのツールのクラス"""
    def __init__(self, chain):
        self.chain = chain
        self.name = "Graph QA"
        self.description = "Use GraphCypherQAChain to answer questions: you can generate Cypher queries based on natural language input to query a graph database."
    
    def output_name(self):
        return self.name
    
    def output_description(self):
        return self.description

    def query(self, question: str) -> str:
        """
        Use GraphCypherQAChain to answer questions: you can generate Cypher queries based on natural language input to query a graph database.
        """
        result = self.chain(question)
        return result['result']

# ----- いったん使ってない ---------------------------------------------------
class Agent:
    """エージェントのクラス"""
    def __init__(
            self,
            model: str,
            tools: Sequence[Callable],
            project: str,
            location: str,
        ):
        self.model_name = model
        self.tools = tools
        self.project = project
        self.location = location

    def set_up(self):
        """All unpickle-able logic should go here.

        The .set_up() method should not be called for an object that is being
        prepared for deployment.
        """

        prompt = {
            "input": lambda x: x["input"],
            "agent_scratchpad": (
                lambda x: format_to_tool_messages(x["intermediate_steps"])
            ),
        } | prompts.ChatPromptTemplate.from_messages([
            ("user", "{input}"),
            prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatVertexAI(model_name=self.model_name)
        if self.tools:
            llm = llm.bind_tools(tools=self.tools)

        self.agent_executor = AgentExecutor(
            agent=prompt | llm | ToolsAgentOutputParser(),
            tools=[StructuredTool.from_function(tool) for tool in self.tools],
            verbose=True,
        )

    def query(self, input: str):
        """Query the application.

        Args:
            input: The user prompt.

        Returns:
            The output of querying the application with the given input.
        """
        return self.agent_executor.invoke({"input": input})
# ----------------------------------------------------------------------------


# ツールとエージェントのセットアップ
def setup_agent():
    """
    LCEL記法でエージェントを作成する関数
    履歴も保持できるが、エージェントの思考過程の表示がイマイチな時がある
    """
    # Google Cloudの初期化
    vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"))

    # ベクトルインデックスの作成
    vector_index = Neo4jVector.from_existing_graph(
        VertexAIEmbeddings("text-embedding-004"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"], 
        embedding_node_property="embedding"
    )
    # GraphCypherQAChainの作成
    llm_query = ChatVertexAI(
        model_name="gemini-1.5-pro-001",
        temperature=0.0,
        )
    graph = Neo4jGraph()
    graph_chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm_query, verbose=True)

    # ツールの作成
    vector_tool = VectorIndexTool(vector_index)
    graph_tool = GraphQATool(graph_chain)
    tools = [vector_tool.search, graph_tool.query]
    
    # promptの作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたは日本の歴史に詳しいAIアシスタントです。日本語で質問に答えてください。
         
         以下のツールを利用できます：
         Vector Index Search: ベクトルインデックスを利用して、グラフデータベースからテキスト情報を検索できます。
         Graph QA: 自然言語入力からグラフデータベースをクエリできます。
         
         質問に答える際は、以下のフォーマットを使用してください：
         
         Question: 回答しなければならない入力質問
         Thought: 何をすべきかを常に考える必要があります
         Action: 実行するアクション。Vector Index Search、Graph QAのいずれかである必要があります
         Action Input: アクションへの入力
         Observation: アクションの結果
         ... (この考え/アクション/アクション入力/観察は N 回繰り返すことができます)
         Thought: これで最終的な答えがわかりました
         Final Answer: 元の入力質問に対する最終的な答え
         
         すべての思考過程とツールの使用を明示的に表示してください。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # モデルの定義
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-001",
        temperature=0.0,
        )
    llm = llm.bind_tools(tools=tools)
    
    # エージェントの作成
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | ToolsAgentOutputParser()
    )
    # AgentExecutorの作成
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[StructuredTool.from_function(tool) for tool in tools],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor


def setup_react_agent():
    """
    create_react_agentを使用してエージェントを作成する関数
    これだとうまくチャット履歴を保持する方法がわからなかったので未使用
    """
    # Google Cloudの初期化
    vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"))

    # ベクトルインデックスの作成
    vector_index = Neo4jVector.from_existing_graph(
        VertexAIEmbeddings("text-embedding-004"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"], 
        embedding_node_property="embedding"
    )
    # GraphCypherQAChainの作成
    llm_query = ChatVertexAI(
        model_name="gemini-1.5-pro-001",
        temperature=0.0,
        )
    graph = Neo4jGraph()
    graph_chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm_query, verbose=True)

    # ツールの作成
    vector_tool = VectorIndexTool(vector_index)
    graph_tool = GraphQATool(graph_chain)
    tools = [
        Tool(
            name=vector_tool.output_name(),
            func=vector_tool.search,
            description=vector_tool.output_description(),
        ),
        Tool(
            name=graph_tool.output_name(),
            func=graph_tool.query,
            description=graph_tool.output_description(),
        )
    ]
    
    # promptの作成
    prompt_template = """あなたは日本の歴史に詳しいAIアシスタントです。
    日本語で質問に答えてください。以下のツールを使用できます： {tools}
    
    これまでの会話履歴：
    {chat_history}

    新しい質問に答える際は、これまでの会話の文脈を考慮してください。
    
    また、質問に答える際は、以下のフォーマットを使用してください。
    
    Question: 回答しなければならない入力質問
    Thought: 何をすべきかを常に考える必要があります
    Action: 実行するアクション。[{tool_names}] のいずれかである必要があります
    Action Input: アクションへの入力
    Observation: アクションの結果
    ... (この考え/アクション/アクション入力/観察は N 回繰り返すことができます)
    Thought: これで最終的な答えがわかりました
    Final Answer: 元の入力質問に対する最終的な答え
    
    Question: {input}
    Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    # モデルの定義
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-001",
        temperature=0.0,
        streaming=True,
        )
    
    # エージェントの作成
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    # AgentExecutorの作成
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    
    return agent_executor


def extract_final_answer(agent_output: str) -> str:
    """
    エージェントの出力から最終回答を抽出する関数
    """
    match = re.search(r"Final Answer: (.*?)(?:\n|$)", agent_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "最終回答を見つけることができませんでした。"


# %%
# 試す
# Google Cloudの初期化
# vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("REGION"))
# agent = setup_agent()
# result = agent.query("信長と秀吉の関係は？")
# print(result)
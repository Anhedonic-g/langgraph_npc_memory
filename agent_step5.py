import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
# 👇 新增导入
from langchain_community.vectorstores import Chroma

load_dotenv()

# ================= 1. 定义工具 =================
@tool
def search_game_wiki(query: str) -> str:
    """根据关键词查询游戏Wiki，返回相关词条的解释。"""
    wiki_db = {
        "生命值": "角色的生命值，降至0时角色死亡。",
        "攻击力": "影响角色对敌人造成的伤害。",
        "暴击": "攻击时有几率造成额外伤害。"
    }
    return wiki_db.get(query, "未找到相关词条。")

@tool
def calculate_damage(attack: int, defense: int, skill_multiplier: float = 1.0) -> str:
    """根据攻击力、防御力和技能倍率计算伤害值。"""
    damage = (attack * skill_multiplier) - defense
    damage = max(1, int(damage))
    return f"计算出的伤害值为: {damage}"

tools = [search_game_wiki, calculate_damage]

# ================= 2. 配置 LLM =================
llm = ChatOpenAI(model="qwen-plus", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ================= 3. 长期记忆初始化 =================
# ⚠️ 注意：显式传入 API Key 和 Base URL，确保 Embedding 也能连上
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    check_embedding_ctx_length=False,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE") 
)

# 初始化 Chroma 向量库（会持久化保存在本地 ./npc_memory_db 文件夹里）
long_term_memory = Chroma(
    collection_name="npc_memory",
    embedding_function=embeddings,
    persist_directory="./npc_memory_db"
)

def store_interaction(user_input: str, npc_response: str):
    """将一次完整的交互存入长期记忆。"""
    doc = Document(page_content=f"用户: {user_input}\n艾莉亚: {npc_response}")
    long_term_memory.add_documents([doc])

def retrieve_relevant_memory(query: str, k: int = 2):
    """根据用户输入检索相关的长期记忆。"""
    try:
        docs = long_term_memory.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        # 如果数据库是空的，搜索会报错，返回空字符串即可
        return ""

# ================= 4. 构建 Agent 图 =================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = """你是一个名叫“艾莉亚”的精灵族NPC，生活在游戏世界中。
你博学多识，性格温和。请用友好的语气回答玩家的问题。
当玩家询问游戏机制或需要计算时，请主动使用你拥有的工具。"""

def call_model(state: AgentState):
    messages = state["messages"]
    last_user_msg = messages[-1].content if messages else ""
    
    # 👇 检索长期记忆并注入系统提示词
    retrieved_memory = retrieve_relevant_memory(last_user_msg)
    current_system_prompt = SYSTEM_PROMPT
    if retrieved_memory:
        current_system_prompt += f"\n\n以下是你记得的相关历史对话：\n{retrieved_memory}"

    # 重新构建消息列表，确保第一条永远是当前最新的系统提示词
    new_messages = [SystemMessage(content=current_system_prompt)]
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            new_messages.append(msg)

    response = llm_with_tools.invoke(new_messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ================= 5. 测试运行 =================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_session_1"}}
    
    print("--- 第一轮对话 ---")
    user_input1 = "你好，我叫小明。我最喜欢的游戏角色是精灵。"
    result1 = graph.invoke({"messages": [HumanMessage(content=user_input1)]}, config)
    npc_response1 = result1["messages"][-1].content
    print("艾莉亚：", npc_response1)
    # 👇 存入长期记忆
    store_interaction(user_input1, npc_response1)
    
    print("\n--- 第二轮对话（换个话题测试） ---")
    user_input2 = "帮我计算一下攻击力200，防御力30的伤害。"
    result2 = graph.invoke({"messages": [HumanMessage(content=user_input2)]}, config)
    npc_response2 = result2["messages"][-1].content
    print("艾莉亚：", npc_response2)
    store_interaction(user_input2, npc_response2)

    print("\n--- 第三轮对话（测试长期记忆召回） ---")
    # 故意问一个之前提过、但已经被短期记忆挤下去的事情
    user_input3 = "你还记得我最喜欢的游戏角色是什么吗？"
    result3 = graph.invoke({"messages": [HumanMessage(content=user_input3)]}, config)
    print("艾莉亚：", result3["messages"][-1].content)
import os
import uuid
import gradio as gr
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
from langchain_chroma import Chroma

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
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    check_embedding_ctx_length=False,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE")
)

long_term_memory = Chroma(
    collection_name="npc_memory",
    embedding_function=embeddings,
    persist_directory="./npc_memory_db"
)

def store_interaction(user_input: str, npc_response: str):
    doc = Document(page_content=f"用户: {user_input}\n艾莉亚: {npc_response}")
    long_term_memory.add_documents([doc])

def retrieve_relevant_memory(query: str, k: int = 2):
    try:
        docs = long_term_memory.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])
    except Exception:
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
    retrieved_memory = retrieve_relevant_memory(last_user_msg)
    
    current_system_prompt = SYSTEM_PROMPT
    if retrieved_memory:
        current_system_prompt += f"\n\n以下是你记得的相关历史对话：\n{retrieved_memory}"

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

# ================= 5. Gradio 界面 =================
def chat_with_npc(message, history, thread_id):
    """Gradio 的处理函数"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # 调用 LangGraph
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    npc_response = result["messages"][-1].content
    
    # 存入长期记忆
    store_interaction(message, npc_response)
    
    # 更新聊天历史
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": npc_response})
    
    return history, ""  # 返回更新后的历史记录和清空的输入框

with gr.Blocks(title="与艾莉亚对话") as demo:
    gr.Markdown("## 🧝‍♀️ 与NPC“艾莉亚”对话")
    gr.Markdown("试着问问：暴击是什么？或者 帮我计算一下攻击力100，防御力50的伤害...")
    
    # 使用 State 保存当前会话的 thread_id
    thread_id_state = gr.State(str(uuid.uuid4()))
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="你的消息", placeholder="输入你的问题...")
    clear = gr.ClearButton([msg, chatbot])

    # 绑定提交事件
    msg.submit(chat_with_npc, [msg, chatbot, thread_id_state], [chatbot, msg])
    # 回车键提交

if __name__ == "__main__":
    demo.launch()
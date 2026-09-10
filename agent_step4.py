import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# 👇 新增导入
from langgraph.checkpoint.memory import MemorySaver

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

# ================= 3. 构建 Agent 图 =================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = """你是一个名叫“艾莉亚”的精灵族NPC，生活在游戏世界中。
你博学多识，性格温和。请用友好的语气回答玩家的问题。
当玩家询问游戏机制或需要计算时，请主动使用你拥有的工具。"""

def call_model(state: AgentState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
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

# 👇 新增：初始化内存存储
memory = MemorySaver()

# 👇 编译图时，传入 checkpointer
graph = builder.compile(checkpointer=memory)

# ================= 4. 测试运行 =================
if __name__ == "__main__":
    # 指定一个 thread_id，只要 thread_id 不变，Agent 就会记住这次会话的上下文
    config = {"configurable": {"thread_id": "test_session_1"}}
    
    print("--- 第一轮对话 ---")
    result1 = graph.invoke({"messages": [HumanMessage(content="你好，我叫小明。")]}, config)
    print("艾莉亚：", result1["messages"][-1].content)
    
    print("\n--- 第二轮对话（测试记忆） ---")
    result2 = graph.invoke({"messages": [HumanMessage(content="你还记得我叫什么名字吗？")]}, config)
    print("艾莉亚：", result2["messages"][-1].content)
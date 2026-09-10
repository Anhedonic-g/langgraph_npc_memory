import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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
    """调用LLM节点，根据当前状态决定下一步行动。"""
    messages = state["messages"]
    # 在对话最前面加入系统提示词
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """条件边：判断LLM是否要求调用工具。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 构建图
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools)) # 自动执行工具

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent") # 工具执行完毕后，返回给agent继续思考

# 编译图（先不加记忆，只测试图的结构）
graph = builder.compile()

# ================= 4. 测试运行 =================
if __name__ == "__main__":
    print("--- 测试 1：询问游戏机制 ---")
    result1 = graph.invoke({"messages": [HumanMessage(content="暴击是什么意思？")]})
    print("最终回复：", result1["messages"][-1].content)
    print("-" * 30)

    print("--- 测试 2：要求计算伤害 ---")
    result2 = graph.invoke({"messages": [HumanMessage(content="帮我计算攻击力100，防御力50，技能倍率1.5的伤害值。")]})
    print("最终回复：", result2["messages"][-1].content)
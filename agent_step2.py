import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()

# 1. 定义工具（把之前测试的代码复制过来）
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

# 2. 初始化 LLM
llm = ChatOpenAI(model="qwen-plus", temperature=0)

# 3. 将工具绑定到 LLM 上
# 这行代码会让 Qwen 知道，它有两个工具可以调用
llm_with_tools = llm.bind_tools(tools)

# 4. 测试
if __name__ == "__main__":
    print("--- 测试 1：询问游戏机制（应该触发搜索工具） ---")
    response1 = llm_with_tools.invoke([HumanMessage(content="暴击是什么意思？")])
    
    # 打印模型的响应。如果没有直接给出文字，而是给出了 tool_calls，说明它准备调用工具了
    print("模型返回内容:", response1.content)
    print("模型请求的工具调用:", response1.tool_calls)
    print("-" * 30)

    print("--- 测试 2：要求计算伤害（应该触发计算工具） ---")
    response2 = llm_with_tools.invoke([HumanMessage(content="帮我计算一下攻击力100，防御力50，技能倍率1.5的伤害值。")])
    
    print("模型返回内容:", response2.content)
    print("模型请求的工具调用:", response2.tool_calls)
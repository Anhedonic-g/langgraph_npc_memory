from langchain_core.tools import tool

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
    damage = max(1, int(damage)) # 至少造成1点伤害
    return f"计算出的伤害值为: {damage}"

tools = [search_game_wiki, calculate_damage]

# ==============================
# 下面写一段测试代码，直接运行一下，看看工具能不能用
# ==============================
if __name__ == "__main__":
    print("--- 测试 Wiki 查询 ---")
    print(search_game_wiki.invoke("暴击"))
    
    print("\n--- 测试 伤害计算 ---")
    print(calculate_damage.invoke({"attack": 100, "defense": 50, "skill_multiplier": 1.5}))
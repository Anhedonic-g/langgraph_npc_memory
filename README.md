# 🧝‍♀️ 游戏NPC对话Agent (艾莉亚)

基于 **Qwen (通义千问)** 和 **LangGraph** 构建的游戏NPC对话智能体，具备**短期/长期记忆**、**自动工具调用**和**自然的角色扮演**能力。

## ✨ 核心功能
- **🎭 角色扮演**：通过系统提示词设定，拥有“精灵学者艾莉亚”的独特人设和语气。
- **🛠️ 自动工具调用**：根据玩家提问，自动决策并调用 `search_game_wiki`（游戏百科检索）和 `calculate_damage`（伤害数值计算）工具。
- **🧠 短期记忆**：基于 `MemorySaver` 和 `thread_id`，记住当前会话上下文。
- **📚 长期记忆**：基于 `ChromaDB` 向量数据库和 `text-embedding-v2` 模型，将重要交互持久化，即使程序重启也能“召回”记忆。
- **🖥️ 交互界面**：使用 `Gradio` 搭建流畅的网页聊天界面。

## 🛠️ 技术栈
- **LLM**: Qwen (qwen-plus)
- **编排框架**: LangGraph, LangChain
- **向量数据库**: ChromaDB (langchain-chroma)
- **Embedding**: text-embedding-v2
- **前端界面**: Gradio

## 🚀 如何运行

1. **安装依赖**
   ```bash
   pip install langgraph langchain langchain-community langchain-openai langchain-chroma gradio chromadb python-dotenv
2. **配置环境变量**
   在项目根目录创建 `.env` 文件，填入你的阿里云百炼 API Key 和专属 Base URL：
   ```text
   OPENAI_API_KEY=sk-你的Key
   OPENAI_API_BASE=你的专属地址
3. **启动应用**
python agent_step6.py
打开浏览器访问 http://127.0.0.1:7860 即可开始对话。

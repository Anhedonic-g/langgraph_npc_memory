import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv() # 读取你的 .env 文件

llm = ChatOpenAI(model="qwen-plus", temperature=0)
try:
    response = llm.invoke([HumanMessage(content="你好，请回复：测试成功！")])
    print("API 连通成功！返回内容：", response.content)
except Exception as e:
    print("API 报错，请检查配置：", str(e))
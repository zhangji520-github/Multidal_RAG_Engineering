"""
Stream模式对比测试 - values vs updates
重点展示两种模式的核心区别
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_utils import qwen3
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


# ==================== 定义状态 ====================
class AgentState(TypedDict):
    """简单的Agent状态"""
    messages: Annotated[list, add_messages]


# ==================== 定义工具 ====================
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴天，温度15-25℃",
        "上海": "多云，温度18-28℃",
    }
    return weather_data.get(city, f"{city}天气晴朗")


# 绑定工具到模型
tools = [get_weather]
llm_with_tools = qwen3.bind_tools(tools)


# ==================== 定义节点 ====================
def chatbot(state: AgentState) -> dict:
    """调用LLM生成响应"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ==================== 构建图 ====================
def create_agent():
    """创建一个简单的Agent图"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    
    # 添加边
    workflow.add_edge(START, "chatbot")
    
    # 条件边：如果有工具调用就执行工具，否则结束
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
    workflow.add_edge("tools", "chatbot")
    
    return workflow.compile()


# ==================== 对比测试 ====================
def compare_modes():
    """并排对比 values 和 updates 模式"""
    
    print("\n" + "="*100)
    print("【核心区别对比】values vs updates")
    print("="*100)
    
    question = "北京天气怎么样？"
    
    # ==================== VALUES 模式 ====================
    print("\n" + "─"*100)
    print("📦 VALUES 模式 - 返回完整状态（所有消息累积）")
    print("─"*100)
    
    agent1 = create_agent()
    inputs = {"messages": [HumanMessage(content=question)]}
    
    step_count = 0
    for chunk in agent1.stream(inputs, stream_mode="values"):
        step_count += 1
        print(f"\n【第 {step_count} 步输出】")
        print(f"📊 状态中的消息总数: {len(chunk['messages'])}")
        print(f"📋 所有消息列表:")
        
        for i, msg in enumerate(chunk['messages'], 1):
            msg_type = type(msg).__name__
            content = ""
            if hasattr(msg, "content"):
                content = msg.content[:50] if msg.content else "(工具调用)"
            print(f"   {i}. {msg_type}: {content}")
        print("─" * 100)
    
    # ==================== UPDATES 模式 ====================
    print("\n" + "─"*100)
    print("📦 UPDATES 模式 - 返回增量更新（只有新增的消息）")
    print("─"*100)
    
    agent2 = create_agent()
    inputs = {"messages": [HumanMessage(content=question)]}
    
    step_count = 0
    for chunk in agent2.stream(inputs, stream_mode="updates"):
        step_count += 1
        
        for node_name, update_data in chunk.items():
            print(f"\n【第 {step_count} 步输出】")
            print(f"🔧 执行的节点: {node_name}")
            print(f"📊 本次新增的消息数: {len(update_data.get('messages', []))}")
            print(f"📋 新增的消息:")
            
            for i, msg in enumerate(update_data.get('messages', []), 1):
                msg_type = type(msg).__name__
                content = ""
                if hasattr(msg, "content"):
                    content = msg.content[:50] if msg.content else "(工具调用)"
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    content += f" [调用工具: {msg.tool_calls[0]['name']}]"
                print(f"   {i}. {msg_type}: {content}")
            print("─" * 100)


def print_explanation():
    """打印详细说明"""
    print("\n" + "="*100)
    print("【总结】核心区别")
    print("="*100)
    
    explanation = """
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   模式       ┃   返回内容                                                ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│              │                                                            │
│   VALUES     │   返回完整状态 state                                       │
│              │   • 包含从开始到现在的所有消息                              │
│              │   • 消息列表不断累积：[msg1] → [msg1, msg2] → [msg1, msg2, msg3]│
│              │   • 可以看到完整的对话历史                                  │
│              │                                                            │
├──────────────┼────────────────────────────────────────────────────────────┤
│              │                                                            │
│   UPDATES    │   返回增量更新 {节点名: 新增数据}                          │
│              │   • 只包含当前节点新增的消息                                │
│              │   • 每次只返回新增部分：{chatbot: [msg2]} → {tools: [msg3]}│
│              │   • 可以清楚看到每个节点做了什么                            │
│              │                                                            │
└──────────────┴────────────────────────────────────────────────────────────┘

🔍 举例说明（假设执行流程：用户提问 → chatbot调用工具 → tools执行 → chatbot回答）

VALUES 模式输出：
  第1步: {"messages": [用户消息]}                                    ← 1条消息
  第2步: {"messages": [用户消息, AI工具调用]}                        ← 2条消息（累积）
  第3步: {"messages": [用户消息, AI工具调用, 工具结果]}              ← 3条消息（累积）
  第4步: {"messages": [用户消息, AI工具调用, 工具结果, AI最终回答]}  ← 4条消息（累积）

UPDATES 模式输出：
  第1步: {"chatbot": {"messages": [AI工具调用]}}          ← 只有新增的1条
  第2步: {"tools": {"messages": [工具结果]}}              ← 只有新增的1条
  第3步: {"chatbot": {"messages": [AI最终回答]}}          ← 只有新增的1条

💡 使用场景：
  • VALUES  → 需要完整上下文时使用（如保存对话历史、状态快照）
  • UPDATES → 需要追踪每步操作时使用（如调试、显示进度、增量更新UI）
"""
    print(explanation)


# ==================== 主函数 ====================
def main():
    """运行对比测试"""
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + " "*30 + "LangGraph Stream 模式对比测试" + " "*39 + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    
    try:
        compare_modes()
        print_explanation()
        
        print("\n" + "="*100)
        print("测试完成！")
        print("="*100 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

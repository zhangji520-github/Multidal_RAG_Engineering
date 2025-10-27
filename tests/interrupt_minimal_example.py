"""
LangGraph 中断机制 - 最小可运行示例
"""

from typing import Annotated, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# 1. 定义 State
# ============================================================================

def replace_value(old, new):
    return new if new is not None else old

class State(TypedDict):
    user_input: str
    status: Annotated[Optional[str], replace_value]

# ============================================================================
# 2. 定义节点（在这里触发中断）
# ============================================================================

def approval_node(state):
    """人工审核节点"""
    
    # 触发中断
    decision = interrupt({
        "message": "需要人工审核",
        "user_input": state["user_input"]
    })
    
    # 恢复后根据决策更新状态
    if decision and decision.get("approved"):
        return {"status": "approved"}
    else:
        return {"status": "rejected"}

# ============================================================================
# 3. 构建 Workflow（必须有 checkpointer）
# ============================================================================

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

checkpointer = MemorySaver()  # 🔴 关键
graph = builder.compile(checkpointer=checkpointer)

# ============================================================================
# 4. 使用
# ============================================================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-123"}}
    
    # 第一次调用：触发中断
    print("=" * 50)
    print("第一次调用：触发中断")
    print("=" * 50)
    result = graph.invoke({"user_input": "删除数据"}, config=config)
    
    if "__interrupt__" in result:
        print("✅ 工作流已中断")
        print(f"中断信息: {result['__interrupt__']}")
        
        # 第二次调用：恢复执行
        print("\n" + "=" * 50)
        print("第二次调用：用户批准，恢复执行")
        print("=" * 50)
        resumed = graph.invoke(
            Command(resume={"approved": True}),
            config=config  # 🔴 必须相同的 config
        )
        print(f"最终状态: {resumed['status']}")
    else:
        print("❌ 未触发中断")


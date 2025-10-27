"""
LangGraph 中断机制 - 简化版伪代码

核心：在节点中调用 interrupt() -> 检查 __interrupt__ -> 用 Command(resume=...) 恢复
"""

from typing import Annotated, Literal
from langgraph.types import interrupt, Command

# ============================================================================
# 第一步：State 中需要 Annotated 的字段（避免并发更新错误）
# ============================================================================

def replace_value(old, new):
    """新值覆盖旧值"""
    return new if new is not None else old

class State(TypedDict):
    # 普通字段
    field1: str
    
    # 需要在 Command 中更新的字段，必须用 Annotated
    status: Annotated[Optional[str], replace_value]


# ============================================================================
# 第二步：在节点中触发中断
# ============================================================================

def approval_node(state):
    """人工审核节点"""
    
    # 1. 调用 interrupt() 触发中断，传入要展示给用户的数据
    decision = interrupt({
        "message": "需要人工审核",
        "data": state.get("some_data")
    })
    
    # 2. 中断恢复后，decision 是用户通过 Command(resume=...) 传入的数据
    # 3. 根据 decision 决定下一步
    if decision.get("approved"):
        return {"status": "approved"}
    else:
        return {"status": "rejected"}


# ============================================================================
# 第三步：构建 Workflow
# ============================================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

# 🔴 关键：必须使用 checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ============================================================================
# 第四步：使用中断
# ============================================================================

# 配置：必须指定 thread_id
config = {"configurable": {"thread_id": "session-123"}}

# 第一次调用：触发中断
result = graph.invoke({"field1": "value"}, config=config)

# 检查是否中断
if "__interrupt__" in result:
    print("工作流已中断")
    print(result["__interrupt__"])  # 查看中断信息
    
    # 恢复执行：传入用户决策
    resumed = graph.invoke(
        Command(resume={"approved": True}),
        config=config  # 🔴 必须是同一个 config
    )
    print(resumed["status"])


# ============================================================================
# 高级：使用 Command 动态路由
# ============================================================================

def approval_with_routing(state) -> Command[Literal["approved", "rejected"]]:
    """返回 Command 对象，动态决定下一个节点"""
    
    decision = interrupt({"message": "需要审批"})
    
    # 根据决策动态路由
    if decision.get("approved"):
        return Command(goto="approved")  # 跳转到 approved 节点
    else:
        return Command(goto="rejected")  # 跳转到 rejected 节点


# ============================================================================
# 实战：为 RAG 添加人工审核（伪代码）
# ============================================================================

"""
1️⃣ 修改 state.py：
    class MultidalModalRAGState(MessagesState):
        evaluate_score: Annotated[Optional[float], replace_value]
        approval_status: Annotated[Optional[str], replace_value]

2️⃣ 添加人工审核节点 nodes.py：
    def human_approval_node(state):
        decision = interrupt({
            "score": state["evaluate_score"],
            "response": state["messages"][-1].content
        })
        
        if decision.get("approved"):
            return {"approval_status": "approved"}
        else:
            return {"approval_status": "rejected"}

3️⃣ 添加路由函数 routers.py：
    def route_to_approval(state):
        if state["evaluate_score"] < 0.7:
            return "human_approval"
        return END

4️⃣ 修改 workflow.py：
    builder.add_node("human_approval", human_approval_node)
    builder.add_conditional_edges("evaluate_node", route_to_approval, {
        "human_approval": "human_approval",
        END: END
    })
    builder.add_edge("human_approval", END)

5️⃣ 使用：
    result = graph.invoke(input, config=config)
    
    if "__interrupt__" in result:
        # 展示审批界面
        resumed = graph.invoke(
            Command(resume={"approved": True}),
            config=config
        )
"""


# ============================================================================
# 核心要点总结
# ============================================================================

"""
✅ 必须要做的：
1. graph.compile(checkpointer=MemorySaver())  # 必须有 checkpointer
2. 在节点中调用 interrupt({...})
3. 检查 result["__interrupt__"]
4. 使用 Command(resume=...) 恢复
5. 使用相同的 config (thread_id)

⚠️ 避免的坑：
1. 没有 checkpointer -> 不会触发中断
2. thread_id 不一致 -> 恢复失败
3. 字段需要 Annotated -> 避免并发更新错误
4. interrupt() 返回的是用户 resume 时传入的数据
"""


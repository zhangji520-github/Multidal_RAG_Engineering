"""
LangGraph 中断机制详解与实现教程

本文件展示了如何实现 LangGraph 的中断（interrupt）功能，
包括基础用法和在 RAG 项目中的应用示例。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Annotated, Literal, Optional, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


# ============================================================================
# 示例 1: 基础中断实现
# ============================================================================

def replace_reducer(old, new):
    """简单的替换reducer：新值覆盖旧值"""
    return new if new is not None else old


class BasicState(TypedDict):
    """基础状态定义"""
    user_input: str
    result: Annotated[Optional[str], replace_reducer]
    approval_status: Annotated[Optional[str], replace_reducer]


def process_node(state: BasicState):
    """
    处理节点 - 在此节点中触发中断
    
    关键点：
    1. 调用 interrupt() 函数暂停执行
    2. 传入需要展示给用户的信息
    3. interrupt() 返回用户恢复时提供的数据
    """
    user_input = state["user_input"]
    
    # 🔴 关键：调用 interrupt() 触发中断
    approval_decision = interrupt({
        "message": "请审批这个操作",
        "user_input": user_input,
        "timestamp": "2025-10-22 10:00:00"
    })
    
    # ✅ 中断恢复后，继续执行
    # approval_decision 是用户通过 Command(resume=...) 传入的数据
    print(f"用户决策: {approval_decision}")
    
    if approval_decision and approval_decision.get("approved"):
        return {
            "result": f"已处理: {user_input}",
            "approval_status": "approved"
        }
    else:
        return {
            "result": "操作被取消",
            "approval_status": "rejected"
        }


def build_basic_graph():
    """构建基础示例图"""
    builder = StateGraph(BasicState)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    
    # 🔴 关键：必须使用 checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def test_basic_interrupt():
    """测试基础中断功能"""
    print("\n" + "=" * 60)
    print("示例 1: 基础中断实现")
    print("=" * 60)
    
    graph = build_basic_graph()
    config = {"configurable": {"thread_id": "basic-001"}}
    
    # 第一次调用：触发中断
    print("\n1️⃣ 第一次调用 - 触发中断...")
    result = graph.invoke(
        {"user_input": "删除重要数据"},
        config=config
    )
    
    # 检查中断信息
    if "__interrupt__" in result:
        print("✅ 工作流已中断")
        print(f"中断信息: {result['__interrupt__']}")
        
        # 第二次调用：恢复执行
        print("\n2️⃣ 第二次调用 - 用户批准，恢复执行...")
        resumed = graph.invoke(
            Command(resume={"approved": True, "comment": "已审核"}),
            config=config
        )
        print(f"最终结果: {resumed}")
    else:
        print("❌ 未检测到中断")


# ============================================================================
# 示例 2: 使用 Command 进行条件路由
# ============================================================================

class RoutingState(TypedDict):
    """路由状态定义"""
    action: str
    status: Annotated[Optional[str], replace_reducer]


def approval_with_routing(state: RoutingState) -> Command[Literal["approve", "reject"]]:
    """
    带路由的审批节点
    
    关键点：
    1. 返回 Command 对象可以动态控制下一个节点
    2. interrupt() 的返回值可以用于决定路由方向
    """
    action = state["action"]
    
    # 触发中断，等待用户决策
    decision = interrupt({
        "question": f"是否批准操作: {action}?",
        "options": ["approve", "reject"]
    })
    
    # 根据用户决策动态路由
    if decision and decision.get("approved"):
        return Command(goto="approve")
    else:
        return Command(goto="reject")


def approve_node(state: RoutingState):
    """批准节点"""
    return {"status": "approved"}


def reject_node(state: RoutingState):
    """拒绝节点"""
    return {"status": "rejected"}


def build_routing_graph():
    """构建带路由的图"""
    builder = StateGraph(RoutingState)
    builder.add_node("approval", approval_with_routing)
    builder.add_node("approve", approve_node)
    builder.add_node("reject", reject_node)
    
    builder.add_edge(START, "approval")
    # 注意：Command(goto=...) 会自动路由，不需要显式添加条件边
    builder.add_edge("approval", "approve")
    builder.add_edge("approval", "reject")
    builder.add_edge("approve", END)
    builder.add_edge("reject", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def test_routing_interrupt():
    """测试带路由的中断"""
    print("\n" + "=" * 60)
    print("示例 2: 带条件路由的中断")
    print("=" * 60)
    
    graph = build_routing_graph()
    config = {"configurable": {"thread_id": "routing-001"}}
    
    # 第一次调用：触发中断
    print("\n1️⃣ 第一次调用 - 触发中断...")
    result = graph.invoke(
        {"action": "转账 $1000"},
        config=config
    )
    
    if "__interrupt__" in result:
        print("✅ 工作流已中断")
        print(f"中断信息: {result['__interrupt__']}")
        
        # 恢复执行 - 批准
        print("\n2️⃣ 用户批准操作...")
        resumed = graph.invoke(
            Command(resume={"approved": True}),
            config=config
        )
        print(f"最终状态: {resumed['status']}")  # -> "approved"


# ============================================================================
# 示例 3: 在 Tool 中使用中断（类似 send_email 示例）
# ============================================================================

from langchain.tools import tool
from langchain_core.messages import ToolMessage


@tool
def dangerous_operation(operation_name: str, target: str) -> str:
    """
    危险操作工具 - 执行前需要人工审批
    
    这是在 Tool 内部使用 interrupt 的典型场景
    """
    # 在真正执行操作前中断，等待审批
    approval = interrupt({
        "tool": "dangerous_operation",
        "operation": operation_name,
        "target": target,
        "warning": "⚠️ 这是一个危险操作，请仔细审核！"
    })
    
    # 用户审批后继续
    if approval and approval.get("confirmed"):
        result = f"✅ 已执行 {operation_name} on {target}"
        print(result)
        return result
    else:
        return "❌ 操作已取消"


class ToolState(TypedDict):
    """工具状态"""
    messages: Annotated[list, lambda x, y: x + y]


def agent_node(state: ToolState):
    """模拟 Agent 节点"""
    from llm_utils import qwen3_max
    
    llm_with_tools = qwen3_max.bind_tools([dangerous_operation])
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}


class ToolNode:
    """自定义工具执行节点"""
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, state: dict):
        messages = state.get("messages", [])
        last_message = messages[-1]
        
        outputs = []
        for tool_call in last_message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def should_continue_to_tools(state: ToolState):
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def build_tool_graph():
    """构建带工具的图"""
    builder = StateGraph(ToolState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([dangerous_operation]))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue_to_tools, {
        "tools": "tools",
        END: END
    })
    builder.add_edge("tools", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def test_tool_interrupt():
    """测试工具中的中断"""
    print("\n" + "=" * 60)
    print("示例 3: 在 Tool 中使用中断")
    print("=" * 60)
    
    try:
        graph = build_tool_graph()
        config = {"configurable": {"thread_id": "tool-001"}}
        
        print("\n1️⃣ 调用 Agent，LLM 决定调用危险工具...")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Delete the production database"}]},
            config=config
        )
        
        if "__interrupt__" in result:
            print("✅ 工具执行被中断，等待审批")
            print(f"中断信息: {result['__interrupt__']}")
            
            print("\n2️⃣ 管理员审批操作...")
            resumed = graph.invoke(
                Command(resume={"confirmed": True}),
                config=config
            )
            print(f"工具返回结果: {resumed['messages'][-1].content}")
    except Exception as e:
        print(f"⚠️ 此示例需要配置 LLM，跳过测试: {e}")


# ============================================================================
# 核心要点总结
# ============================================================================

def print_key_points():
    """打印关键要点"""
    print("\n" + "=" * 60)
    print("🎯 LangGraph 中断机制 - 核心要点")
    print("=" * 60)
    
    points = """
1️⃣ 必须使用 Checkpointer
   ❌ graph = builder.compile()  # 不会触发中断
   ✅ graph = builder.compile(checkpointer=MemorySaver())

2️⃣ 调用 interrupt() 函数触发中断
   from langgraph.types import interrupt
   
   user_decision = interrupt({
       "message": "需要展示给用户的信息",
       "data": {...}
   })

3️⃣ 检查中断
   result = graph.invoke(input, config=config)
   if "__interrupt__" in result:
       # 工作流已中断
       print(result["__interrupt__"])

4️⃣ 恢复执行
   from langgraph.types import Command
   
   # 方式1: 传递简单值
   graph.invoke(Command(resume=True), config=config)
   
   # 方式2: 传递复杂数据
   graph.invoke(Command(resume={"approved": True, "comment": "OK"}), config=config)

5️⃣ 使用 Command 进行动态路由
   def node(state) -> Command[Literal["next_node_a", "next_node_b"]]:
       decision = interrupt(...)
       return Command(goto="next_node_a" if decision else "next_node_b")

6️⃣ 处理并发更新（使用 Annotated）
   from typing import Annotated
   
   class State(TypedDict):
       # 普通字段：每步只能更新一次
       field1: str
       
       # Annotated 字段：可以在一个步骤中多次更新
       field2: Annotated[str, lambda old, new: new]

7️⃣ 必须使用相同的 thread_id 恢复
   config = {"configurable": {"thread_id": "same-id"}}
   result = graph.invoke(input, config=config)
   # ... 中断 ...
   resumed = graph.invoke(Command(resume=...), config=config)  # 相同 config

8️⃣ 中断可以嵌套在任何地方
   ✅ 直接在节点函数中
   ✅ 在 Tool 函数中
   ✅ 在节点调用的子函数中
"""
    print(points)


# ============================================================================
# 实战示例：为 RAG 项目添加中断
# ============================================================================

def print_rag_implementation():
    """展示如何在 RAG 项目中实现中断"""
    print("\n" + "=" * 60)
    print("🚀 在你的 RAG 项目中实现中断")
    print("=" * 60)
    
    code = '''
# 1️⃣ 修改 state.py - 添加 Annotated 支持
from typing import Annotated, Optional

def replace_value(old, new):
    return new if new is not None else old

class MultidalModalRAGState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    evaluate_score: Annotated[Optional[float], replace_value]
    approval_status: Annotated[Optional[str], replace_value]  # 新增
    approval_comment: Annotated[Optional[str], replace_value]  # 新增

# 2️⃣ 修改 nodes.py - 实现人工审核节点
from langgraph.types import interrupt, Command
from typing import Literal

def human_approval_node(state: MultidalModalRAGState) -> Command[Literal["approved", "rejected"]]:
    """
    人工审核节点 - 当评分低于阈值时触发
    """
    evaluate_score = state.get("evaluate_score", 0)
    last_response = state["messages"][-1].content if state["messages"] else ""
    
    # 触发中断，等待人工审批
    decision = interrupt({
        "type": "quality_review",
        "evaluate_score": evaluate_score,
        "response_preview": last_response[:200],
        "context_retrieved": len(state.get("context_retrieved", [])),
        "message": "⚠️ 响应质量评分较低，需要人工审核"
    })
    
    # 根据审批结果路由
    if decision and decision.get("approved"):
        return Command(
            goto="approved",
            update={"approval_comment": decision.get("comment", "")}
        )
    else:
        return Command(
            goto="rejected",
            update={"approval_comment": decision.get("reason", "未通过审核")}
        )

def approved_node(state: MultidalModalRAGState):
    """审批通过节点"""
    return {"approval_status": "approved"}

def rejected_node(state: MultidalModalRAGState):
    """审批拒绝节点 - 可以重新生成回复"""
    return {"approval_status": "rejected"}

# 3️⃣ 修改 workflow.py - 添加审批流程
builder = StateGraph(MultidalModalRAGState)

# 添加节点
builder.add_node("evaluate_node", evaluate_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("approved", approved_node)
builder.add_node("rejected", rejected_node)

# 添加边
builder.add_conditional_edges(
    "evaluate_node",
    route_human_approval_node,  # 根据评分决定是否需要人工审核
    {
        "human_approval": "human_approval",
        END: END
    }
)
builder.add_edge("human_approval", "approved")   # Command 会动态路由
builder.add_edge("human_approval", "rejected")
builder.add_edge("approved", END)
builder.add_edge("rejected", END)  # 或者路由回 third_chatbot 重新生成

# 4️⃣ 使用示例
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "user-session-123"}}
    
    # 第一次调用
    result = graph.invoke(
        {"messages": [HumanMessage(content="用户问题")]},
        config=config
    )
    
    # 检查是否需要审批
    if "__interrupt__" in result:
        # 展示审批界面给管理员
        interrupt_data = result["__interrupt__"][0].value
        print(f"评分: {interrupt_data['evaluate_score']}")
        print(f"预览: {interrupt_data['response_preview']}")
        
        # 管理员审批后恢复
        resumed = graph.invoke(
            Command(resume={
                "approved": True,
                "comment": "质量可接受，允许返回"
            }),
            config=config
        )
        print(f"最终状态: {resumed['approval_status']}")
'''
    print(code)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 运行所有示例
    test_basic_interrupt()
    test_routing_interrupt()
    test_tool_interrupt()
    
    # 打印要点和实战指南
    print_key_points()
    print_rag_implementation()
    
    print("\n" + "=" * 60)
    print("✅ 教程完成！")
    print("=" * 60)


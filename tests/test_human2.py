import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import TypedDict

from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from llm_utils import qwen3_max
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import ToolMessage
class AgentState(TypedDict):
    messages: list[dict]


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending; payload surfaces in result["__interrupt__"]
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    if response.get("action") == "approve":
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # Actually send the email (your implementation here)
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"

    return "Email cancelled by user"


def agent_node(state: AgentState):
    # LLM may decide to call the tool; interrupt pauses before sending
    qwen3_max_llm = qwen3_max.bind_tools([send_email])
    result = qwen3_max_llm.invoke(state["messages"])
    return {"messages": [result]}

#  自定义是为了替代：由LangGraph框架自带的ToolNode（有大模型动态传参 来调用工具） 这个很好写，主要还是tool的逻辑 
class MyToolNode:
    """自定义类，来真正执行工具 通过对齐上一条AIMessage返回的Toolcall字段的信息来调用对应的工具 tools_by_name[tool_call["name"]].invoke()"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    # inputs 就是这个 自定义 state (自定义schema) 的实例
    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if messages:
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            # 使用LLM推理出的args
            tool_args = tool_call["args"].copy()
            print(f"tool_args: {tool_args}")
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_args)
            
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# 路由函数：判断是否需要调用工具
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息包含 tool_calls，则路由到工具节点
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # 否则结束
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", MyToolNode([send_email]))  # 添加工具执行节点
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", END)  # 工具执行完毕后结束

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "email-workflow"}}
initial = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "Send an email to alice@example.com about the meeting"}
        ]
    },
    config=config,
)

print(initial["__interrupt__"])  # -> [Interrupt(value={'action': 'send_email', ...})]


# 恢复测试 无中断
resumed = graph.invoke(
    Command(resume={"action": "approve", "subject": "Updated subject"}),
    config=config,
)
print(resumed["messages"][-1])  # -> Tool result returned by send_email
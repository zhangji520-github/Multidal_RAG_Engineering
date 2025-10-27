from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


class FormState(TypedDict):
    age: int | None

def get_age_node(state: FormState):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload surfaces in result["__interrupt__"]    这行会暂停，等用户回答，然后 answer = 用户输入

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}' is not a valid age. Please enter a positive number."


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}

# 开始交互循环
result = graph.invoke({"age": None}, config=config)
print(result['__interrupt__'])


# # Provide invalid data; the node re-prompts
# retry = graph.invoke(Command(resume="thirty"), config=config)
# print(retry["__interrupt__"])  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]

# # Provide valid data; loop exits and state updates
# final = graph.invoke(Command(resume=30), config=config)
# print(final["age"])  # -> 30


while True:
    # 检查是否有中断（提示信息）
    if "__interrupt__" in result:
        # 获取提示信息
        prompt = result["__interrupt__"][0].value
        print(f"\n{prompt}")
        
        # 获取用户输入
        user_input = input(">>> ").strip()
        
        # 检查是否退出
        if user_input.lower() == "quit":
            print("退出程序")
            break
        
        # 尝试转换为整数
        try:
            age_value = int(user_input)
            result = graph.invoke(Command(resume=age_value), config=config)
        except ValueError:
            # 如果不是整数，直接传递字符串
            result = graph.invoke(Command(resume=user_input), config=config)
    else:
        # 没有中断，说明流程完成
        print(f"\n完成！年龄是: {result.get('age')}")
        break
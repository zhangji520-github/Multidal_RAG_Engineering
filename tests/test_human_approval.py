"""
人工审核节点测试示例
演示如何正确使用 interrupt 动态中断功能
"""

from langgraph.types import Command
from langchain_core.messages import HumanMessage

# 假设你已经有了编译好的图
# from final_rag.workflow import graph (需要确保workflow.py导出graph)


def test_human_approval_workflow():
    """
    测试人工审核工作流的完整流程
    """
    
    # 配置 thread_id，这是持久化状态的关键
    config = {
        "configurable": {
            "thread_id": "test-approval-thread-001"
        }
    }
    
    # 初始输入
    initial_input = {
        "messages": [
            HumanMessage(content="什么是深度学习？")
        ]
    }
    
    print("=" * 60)
    print("步骤 1: 开始执行工作流")
    print("=" * 60)
    
    # 第一次调用：执行到 interrupt 点
    # result = graph.invoke(initial_input, config=config)
    
    # 检查是否触发了中断
    # if "__interrupt__" in result:
    #     print("\n⏸️  工作流已暂停，等待人工审核...")
    #     print(f"中断信息: {result['__interrupt__']}")
    #     
    #     # 显示需要审核的内容
    #     for interrupt_data in result["__interrupt__"]:
    #         print(f"\n问题: {interrupt_data.value.get('question')}")
    #         print(f"评估分数: {interrupt_data.value.get('score')}")
    #         print(f"响应内容: {interrupt_data.value.get('response')[:100]}...")  # 只显示前100字符
    #     
    #     print("\n" + "=" * 60)
    #     print("步骤 2: 人工审核决策")
    #     print("=" * 60)
    #     
    #     # 模拟人工审核决策
    #     # 选项1: 批准回答
    #     user_decision = True  # True = 批准, False = 拒绝
    #     
    #     print(f"审核决策: {'✅ 批准' if user_decision else '❌ 拒绝'}")
    #     
    #     print("\n" + "=" * 60)
    #     print("步骤 3: 恢复执行工作流")
    #     print("=" * 60)
    #     
    #     # 第二次调用：使用相同的 thread_id 恢复执行
    #     resumed_result = graph.invoke(
    #         Command(resume=user_decision),  # 将决策传递给 interrupt
    #         config=config  # 必须使用相同的 thread_id
    #     )
    #     
    #     print(f"\n最终状态: {resumed_result.get('human_answer')}")
    #     print(f"工作流状态: {'已完成' if '__interrupt__' not in resumed_result else '仍在等待'}")
    #     
    #     return resumed_result
    # else:
    #     print("未触发人工审核（评估分数可能高于阈值）")
    #     return result


def test_rejection_then_web_search():
    """
    测试拒绝审核后触发网络搜索的流程
    """
    
    config = {
        "configurable": {
            "thread_id": "test-rejection-thread-001"
        }
    }
    
    initial_input = {
        "messages": [
            HumanMessage(content="2024年诺贝尔物理学奖得主是谁？")
        ]
    }
    
    print("\n" + "=" * 60)
    print("测试场景: 拒绝审核 → 触发网络搜索")
    print("=" * 60)
    
    # 第一次调用
    # result = graph.invoke(initial_input, config=config)
    
    # if "__interrupt__" in result:
    #     print("\n⏸️  触发人工审核...")
    #     
    #     # 拒绝当前回答
    #     print("决策: ❌ 拒绝回答，启动网络搜索")
    #     
    #     # 恢复并拒绝
    #     resumed_result = graph.invoke(
    #         Command(resume=False),  # False = 拒绝，会路由到 fourth_chatbot
    #         config=config
    #     )
    #     
    #     print(f"\n人工答案状态: {resumed_result.get('human_answer')}")
    #     print("预期: 已触发网络搜索节点...")
    #     
    #     return resumed_result


def demo_best_practices():
    """
    演示使用 interrupt 的最佳实践
    """
    
    print("\n" + "=" * 60)
    print("Interrupt 最佳实践总结")
    print("=" * 60)
    
    best_practices = """
    1. ✅ 使用持久化 checkpointer（生产环境使用 PostgresSaver）
       - 开发环境可以使用 MemorySaver
       - 生产环境必须使用数据库支持的 checkpointer
    
    2. ✅ 必须提供 thread_id
       - 恢复时使用相同的 thread_id
       - thread_id 用于标识和恢复特定的执行状态
    
    3. ✅ interrupt() 只传递 JSON 可序列化的值
       - ✅ 字符串、数字、布尔值、字典、列表
       - ❌ 函数、类实例、复杂对象
    
    4. ✅ interrupt() 之前的代码保持幂等性
       - 恢复时节点会从头重新执行
       - 避免在 interrupt 前执行非幂等操作（如创建数据库记录）
    
    5. ❌ 不要在 try/except 中包裹 interrupt()
       - interrupt 通过抛出异常来暂停执行
       - try/except 会捕获这个异常导致中断失败
    
    6. ❌ 不要在条件分支中改变 interrupt 的顺序
       - 恢复值按索引匹配
       - 保持 interrupt 调用顺序一致
    
    7. ✅ 使用 Command(resume=value) 恢复执行
       - resume 的值会成为 interrupt() 的返回值
       - 可以传递任何 JSON 可序列化的值
    
    8. ✅ 检查 __interrupt__ 字段判断是否暂停
       - 返回结果中有此字段表示已暂停
       - 字段值是 interrupt() 传递的 payload
    """
    
    print(best_practices)
    
    error_examples = """
    
    常见错误示例:
    
    ❌ 错误1: 在 try/except 中使用 interrupt
    def bad_node(state):
        try:
            result = interrupt("请审核")  # 会被 except 捕获
        except Exception as e:
            print(e)
        return state
    
    ✅ 正确做法:
    def good_node(state):
        result = interrupt("请审核")  # 不要包裹
        try:
            risky_operation()  # 只对可能出错的操作使用 try/except
        except Exception as e:
            print(e)
        return {"approved": result}
    
    
    ❌ 错误2: 条件跳过 interrupt
    def bad_node(state):
        name = interrupt("姓名?")
        if state.get("needs_age"):  # 条件可能改变
            age = interrupt("年龄?")  # 索引不一致
        city = interrupt("城市?")
        return state
    
    ✅ 正确做法:
    def good_node(state):
        name = interrupt("姓名?")
        age = interrupt("年龄?")  # 始终保持顺序
        city = interrupt("城市?")
        return {"name": name, "age": age, "city": city}
    
    
    ❌ 错误3: 传递不可序列化的值
    def bad_node(state):
        validator = lambda x: len(x) > 0
        result = interrupt({
            "question": "输入:",
            "validator": validator  # ❌ 函数不能序列化
        })
        return state
    
    ✅ 正确做法:
    def good_node(state):
        result = interrupt({
            "question": "输入:",
            "min_length": 1  # ✅ 使用简单值
        })
        return {"input": result}
    """
    
    print(error_examples)


if __name__ == "__main__":
    print("\n" + "🚀 " * 30)
    print("人工审核节点 - 完整测试示例")
    print("🚀 " * 30)
    
    # 演示最佳实践
    demo_best_practices()
    
    # 注意: 实际运行需要先确保 workflow.py 导出了 graph 对象
    # 并取消上面测试函数中的注释
    
    print("\n" + "=" * 60)
    print("测试说明:")
    print("=" * 60)
    print("""
    要运行实际测试，需要:
    1. 确保 workflow.py 编译并导出 graph 对象
    2. 激活 deep_learning 环境
    3. 取消测试函数中的注释
    4. 运行: python tests/test_human_approval.py
    """)


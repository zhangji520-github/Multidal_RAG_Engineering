"""
人工审核节点交互式演示
展示如何在实际应用中使用 interrupt 动态中断功能
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from langgraph.types import Command


def run_interactive_demo():
    """
    运行交互式人工审核演示
    """
    
    print("\n" + "=" * 70)
    print("🎯 多模态 RAG 人工审核流程演示")
    print("=" * 70)
    
    try:
        # 导入图构建函数
        from src.final_rag.workflow import build_graph
        
        print("\n📦 正在构建工作流图...")
        graph = build_graph()
        print("✅ 工作流图构建成功！")
        
    except Exception as e:
        print(f"\n❌ 构建图失败: {e}")
        print("\n请确保:")
        print("  1. 已激活 deep_learning 环境")
        print("  2. PostgreSQL 数据库正在运行")
        print("  3. 数据库连接配置正确")
        return
    
    # 配置线程ID
    thread_id = input("\n请输入线程ID (直接回车使用默认值 'demo-001'): ").strip()
    if not thread_id:
        thread_id = "demo-001"
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    print(f"\n🔖 使用线程ID: {thread_id}")
    
    # 获取用户问题
    user_question = input("\n请输入您的问题 (直接回车使用默认问题): ").strip()
    if not user_question:
        user_question = "什么是深度学习？请详细介绍一下。"
    
    print(f"\n💬 用户问题: {user_question}")
    
    # 构建初始输入
    initial_input = {
        "messages": [
            HumanMessage(content=user_question)
        ]
    }
    
    print("\n" + "-" * 70)
    print("🚀 开始执行工作流...")
    print("-" * 70)
    
    try:
        # 第一次调用 - 执行到中断点
        result = graph.invoke(initial_input, config=config)
        
        # 检查是否触发了中断
        if "__interrupt__" in result:
            print("\n⏸️  工作流已暂停 - 需要人工审核")
            print("=" * 70)
            
            # 显示中断信息
            for idx, interrupt_data in enumerate(result["__interrupt__"], 1):
                interrupt_value = interrupt_data.value
                
                print(f"\n📋 审核请求 #{idx}:")
                print(f"   问题: {interrupt_value.get('question', 'N/A')}")
                print(f"   评估分数: {interrupt_value.get('score', 'N/A'):.4f}")
                print(f"   用户输入: {interrupt_value.get('user_input', 'N/A')}")
                
                response_text = interrupt_value.get('response', '')
                if len(response_text) > 200:
                    print(f"   响应内容: {response_text[:200]}...")
                else:
                    print(f"   响应内容: {response_text}")
            
            print("\n" + "=" * 70)
            print("请做出审核决策:")
            print("  [y/Y] - 批准回答，结束流程")
            print("  [n/N] - 拒绝回答，触发网络搜索")
            print("=" * 70)
            
            # 获取用户决策
            while True:
                decision = input("\n您的决策 (y/n): ").strip().lower()
                if decision in ['y', 'n']:
                    break
                print("❌ 无效输入，请输入 'y' 或 'n'")
            
            is_approved = (decision == 'y')
            
            print(f"\n{'✅ 批准' if is_approved else '❌ 拒绝'} - 正在恢复执行...")
            print("-" * 70)
            
            # 第二次调用 - 恢复执行
            resumed_result = graph.invoke(
                Command(resume=is_approved),
                config=config
            )
            
            # 显示最终结果
            print("\n" + "=" * 70)
            print("🎉 工作流执行完成")
            print("=" * 70)
            
            print(f"\n审核结果: {resumed_result.get('human_answer', 'N/A')}")
            
            # 显示最终消息
            final_messages = resumed_result.get('messages', [])
            if final_messages:
                last_message = final_messages[-1]
                print(f"\n最终响应:")
                print("-" * 70)
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                else:
                    print(last_message)
            
            print("\n" + "=" * 70)
            
        else:
            # 未触发中断（评估分数可能高于阈值）
            print("\n✅ 工作流正常完成 - 未触发人工审核")
            print("=" * 70)
            print("(评估分数可能高于 0.75 阈值)")
            
            final_messages = result.get('messages', [])
            if final_messages:
                last_message = final_messages[-1]
                print(f"\n最终响应:")
                print("-" * 70)
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                else:
                    print(last_message)
    
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()


def show_usage_guide():
    """显示使用指南"""
    
    guide = """
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    人工审核节点 - 使用指南                             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    📌 核心概念
    ─────────────────────────────────────────────────────────────────────
    
    1. interrupt() 函数
       • 在节点中调用时暂停图的执行
       • 传递的值会在调用者处以 __interrupt__ 字段返回
       • 必须传递 JSON 可序列化的值
    
    2. thread_id
       • 用于唯一标识和恢复特定的执行状态
       • 恢复时必须使用相同的 thread_id
       • 生产环境建议使用用户ID或会话ID
    
    3. checkpointer
       • 持久化保存图的状态
       • 开发环境: MemorySaver
       • 生产环境: PostgresSaver, SqliteSaver 等
    
    4. Command(resume=value)
       • 用于恢复被中断的执行
       • value 会成为 interrupt() 的返回值
       • 可以传递任何 JSON 可序列化的值
    
    
    📋 实际应用场景
    ─────────────────────────────────────────────────────────────────────
    
    • 内容审核: 在发布前审核 AI 生成的内容
    • 风险控制: 高风险操作前请求人工确认
    • 质量把关: 低评分回答触发人工复审
    • 工具调用: 在执行敏感工具前请求批准
    
    
    🔧 代码示例
    ─────────────────────────────────────────────────────────────────────
    
    # 定义节点
    def approval_node(state):
        # 暂停并请求审核
        is_approved = interrupt({
            "question": "是否批准？",
            "details": state["data"]
        })
        
        # 恢复后更新状态
        return {"approved": is_approved}
    
    # 首次执行
    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke(input_data, config=config)
    
    # 检查中断
    if "__interrupt__" in result:
        print(result["__interrupt__"])
        
        # 恢复执行
        final = graph.invoke(
            Command(resume=True),  # 或 False
            config=config
        )
    
    
    ⚠️ 注意事项
    ─────────────────────────────────────────────────────────────────────
    
    ❌ 不要在 try/except 中包裹 interrupt()
    ❌ 不要传递不可序列化的对象（函数、类实例等）
    ❌ 不要在条件分支中改变 interrupt 的调用顺序
    ✅ interrupt 前的代码要保持幂等性
    ✅ 使用相同的 thread_id 恢复执行
    ✅ 生产环境使用持久化 checkpointer
    
    
    📚 更多资源
    ─────────────────────────────────────────────────────────────────────
    
    • LangGraph 文档: https://docs.langchain.com/oss/python/langgraph/interrupts
    • 示例代码: tests/test_human_approval.py
    • 最佳实践: 参见代码中的详细注释
    
    """
    
    print(guide)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="人工审核节点交互式演示")
    parser.add_argument(
        "--guide",
        action="store_true",
        help="显示使用指南"
    )
    
    args = parser.parse_args()
    
    if args.guide:
        show_usage_guide()
    else:
        run_interactive_demo()
        
        # 询问是否查看使用指南
        show_guide = input("\n是否查看完整使用指南? (y/n): ").strip().lower()
        if show_guide == 'y':
            show_usage_guide()


"""
工作流交互式测试脚本
用于测试人工审核中断功能
"""
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src.final_rag.workflow import execute_graph


async def test_simple_question():
    """测试简单问题（可能触发人工审核）"""
    print("\n" + "="*80)
    print("测试 1: 简单问题")
    print("="*80)
    
    result = await execute_graph("什么是深度学习？")
    
    print("\n" + "="*80)
    print("执行结果:")
    print("="*80)
    print(f"状态: {result['status']}")
    print(f"会话ID: {result['session_id']}")
    print(f"回答: {result['answer'][:200]}...")  # 只显示前200字符
    if 'human_approved' in result:
        print(f"人工审核结果: {result['human_approved']}")
    print("="*80)


async def test_image_input():
    """测试图片输入（不会触发人工审核）"""
    print("\n" + "="*80)
    print("测试 2: 图片输入（需要准备一张图片）")
    print("="*80)
    
    # 请替换为实际的图片路径
    image_path = input("请输入图片路径（或按回车跳过）: ").strip()
    
    if image_path and Path(image_path).exists():
        result = await execute_graph(image_path)
        print(f"\n执行结果: {result['status']}")
        print(f"回答: {result['answer'][:200]}...")
    else:
        print("跳过图片测试")


async def test_multimodal_input():
    """测试多模态输入"""
    print("\n" + "="*80)
    print("测试 3: 多模态输入（文本 & 图片）")
    print("="*80)
    
    text = input("请输入文本问题: ").strip()
    image_path = input("请输入图片路径: ").strip()
    
    if text and image_path and Path(image_path).exists():
        user_input = f"{text} & {image_path}"
        result = await execute_graph(user_input)
        print(f"\n执行结果: {result['status']}")
        print(f"回答: {result['answer'][:200]}...")
    else:
        print("输入不完整，跳过多模态测试")


async def main():
    print("\n" + "🚀 "*30)
    print("工作流交互式测试")
    print("🚀 "*30)
    print("\n选择测试场景:")
    print("  1. 简单文本问题（可能触发人工审核）")
    print("  2. 图片输入测试")
    print("  3. 多模态输入测试")
    print("  4. 退出")
    
    while True:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            await test_simple_question()
        elif choice == '2':
            await test_image_input()
        elif choice == '3':
            await test_multimodal_input()
        elif choice == '4':
            print("退出测试")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n测试中断")


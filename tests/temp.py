import sys
from pathlib import Path
import asyncio
import platform

# Windows 平台需要设置事件循环策略
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# 数据库连接字符串
DB_URI = 'postgresql://postgres:200132ji@localhost:5432/multidal_modal_rag'

async def main():
    # 配置 session_id
    session_id = "zhangji_张吉测试摘要"
    config = {"configurable": {"thread_id": session_id}}
    
    # 直接查询 checkpointer，不需要构建 graph
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
        # 使用 aget_tuple 方法获取 checkpoint tuple
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        
        # 打印状态
        print("\n" + "="*80)
        print(f"📋 Session ID: {session_id}")
        print("="*80 + "\n")
        
        if checkpoint_tuple:
            # 我们可以看到所有state信息都在 checkpoint_tuple.checkpoint['channel_values'] 里面
            channel_values = checkpoint_tuple.checkpoint['channel_values']
            
            # 1. 对话摘要
            if 'summary' in channel_values and channel_values['summary']:
                summary = channel_values['summary']
                print("📝 对话摘要:")
                print("-" * 80)
                print(summary)
                print("-" * 80 + "\n")
            else:
                print("⚠️  没有生成摘要\n")
            
            # 2. State 信息汇总
            print("📦 State 状态信息:")
            print("-" * 80)
            print(f"消息总数: {len(channel_values.get('messages', []))} 条")
            print(f"保留消息数: {channel_values.get('message_count', 'N/A')} 条")
            print(f"检索决策: {channel_values.get('retrieve_decision', 'N/A')}")
            print(f"人工审核: {channel_values.get('human_answer', 'N/A')}")
            print(f"摘要长度: {len(str(channel_values.get('summary', ''))) if channel_values.get('summary') else 0} 字符")
            print("-" * 80 + "\n")
            
            # 3. 消息历史
            if 'messages' in channel_values:
                messages = channel_values['messages']
                print(f"💬 消息历史 (共 {len(messages)} 条):")
                print("-" * 80)
                for idx, msg in enumerate(messages, 1):
                    msg_type = msg.__class__.__name__
                    content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    print(f"{idx}. [{msg_type}] {content_preview}\n")
                print("-" * 80)
            
        else:
            print(f"❌ 未找到该 session 的状态")
            print(f"💡 当前 session_id: {session_id}\n")

if __name__ == "__main__":
    asyncio.run(main())


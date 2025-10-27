import sys  # noqa: E402
from pathlib import Path  # noqa: E402
import asyncio  # noqa: E402
import platform  # noqa: E402

# Windows 平台需要设置事件循环策略以支持 psycopg 异步
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from utils.print_messages import pretty_print_messages  # noqa: E402
from utils.embeddings_utils import image_to_base64  # noqa: E402
from src.final_rag.utils.nodes import (  # noqa: E402
    process_input,
    SearchContextToolNode,
    first_agent_decision,
    second_agent_generate,
    retrieve_database,
    third_chatbot,
    evaluate_node,
    human_approval_node,
    fourth_chatbot,
    summarize_if_needed,  # 导入摘要节点
    UserContext,  # 导入 UserContext
)
from src.final_rag.utils.state import MultidalModalRAGState  # noqa: E402
from src.final_rag.utils.tools import context_tools, web_tools  # noqa: E402
from src.final_rag.utils.routers import (  # noqa: E402
    route_llm_or_retrieve_database,
    route_evaluate,
    route_human_answer_node,
    route_after_human_approval,
    route_after_first_agent,  # 新增：first_agent_decision 后的智能路由
)
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402
from utils.save_context import get_milvus_writer  # noqa: E402
import logging  # noqa: E402
logger = logging.getLogger(__name__)

# 在生产环境中，请使用数据库支持的检查点机制：
DB_URI = 'postgresql://postgres:200132ji@localhost:5432/multidal_modal_rag'


def build_graph(checkpointer, store):
    """
    构建并返回编译后的 LangGraph 工作流
    
    Args:
        checkpointer: AsyncPostgresSaver 实例
        store: AsyncPostgresStore 实例
    
    Returns:
        编译后的 CompiledGraph 对象
    """
    # ==================== 创建状态图 ====================
    builder = StateGraph(MultidalModalRAGState)

    # ==================== 添加节点 ====================
    # 节点1: 处理用户输入（提取文本/图片，判断输入类型）
    builder.add_node("process_input", process_input)
    
    # 节点2: 摘要节点 - 当消息数量超过阈值时自动生成摘要并删除旧消息
    builder.add_node("summarize_if_needed", summarize_if_needed)
    
    # 节点3: 第一个Agent - 决策是否需要检索用户历史对话上下文
    # 输入: 用户问题 + 系统提示词
    # 输出: 带/不带 tool_calls 的 AIMessage
    builder.add_node("first_agent_decision", first_agent_decision)
    
    # 节点4: 搜索用户历史对话上下文工具节点（自定义 ToolNode）
    # 功能: 根据第一个Agent的tool_calls，检索用户历史对话记录
    builder.add_node("search_context", SearchContextToolNode(tools=context_tools))

    # 节点5: 第二个Agent - 基于检索到的历史对话上下文生成回复
    # 输入: 用户问题 + 检索到的历史对话上下文（ToolMessage）
    # 输出: 最终回答（AIMessage）
    builder.add_node("second_agent_generate", second_agent_generate)
    
    # 节点6: 检索知识数据库（Milvus向量数据库）
    # 功能: 使用混合检索（密集向量 + 稀疏向量）查找相关文档和图片
    builder.add_node("retrieve_database", retrieve_database)
    
    # 节点7: 第三个Chatbot - 基于知识库检索结果生成回复
    # 输入: 用户问题 + 检索到的文档/图片
    # 输出: Markdown 格式的回答
    builder.add_node("third_chatbot", third_chatbot)
    
    # 节点8: 评估节点 - 使用 RAGAS 评估回答质量
    # 功能: 计算响应相关性分数（ResponseRelevancy）
    builder.add_node("evaluate_node", evaluate_node)
    
    # 节点9: 人工审核节点 - 低分回答触发人工审核
    # 功能: 使用 interrupt() 暂停执行，等待人工决策（批准/拒绝）
    builder.add_node("human_approval_node", human_approval_node)
    
    # 节点10: 第四个Chatbot - 人工拒绝后使用网络搜索提供备选答案
    # 功能: 调用互联网搜索工具，生成基于实时信息的回答
    builder.add_node("fourth_chatbot", fourth_chatbot)
    
    # 节点11: 网络搜索工具节点（官方 ToolNode）
    # 功能: 执行 Tavily 网络搜索工具
    builder.add_node("web_search_node", ToolNode(tools=web_tools))

    # ==================== 添加边（工作流路由） ====================
    
    # 起点 → process_input（所有请求都从这里开始）
    builder.add_edge(START, "process_input")
    
    # 固定边1: process_input → summarize_if_needed
    # 在处理用户输入后，先检查是否需要摘要
    builder.add_edge("process_input", "summarize_if_needed")
    
    # 固定边2: summarize_if_needed → first_agent_decision
    # 摘要处理完成后，进入决策Agent
    # 所有输入（纯文本、纯图片、图文混合）都先由 first_agent_decision 智能判断：
    # - 简单问候/闲聊 → 直接回答 → END
    # - 需要历史上下文 → search_context（检索用户历史对话）
    # - 复杂问题 → 不调用工具，后续路由到 retrieve_database（检索知识库）
    builder.add_edge("summarize_if_needed", "first_agent_decision")
    
    # 路由3: first_agent_decision 后的智能路由
    # 使用自定义路由函数 route_after_first_agent 判断：
    # - 调用 search_context 工具 → search_context 节点（检索历史对话）
    # - 调用 web_search 工具 → web_search_node 节点（网络搜索）
    # - 无 tool_calls 但回答完整 → END（简单问题已回答）
    # - 无 tool_calls 且回答不完整 → retrieve_database（检索知识库）
    builder.add_conditional_edges(
        'first_agent_decision',
        route_after_first_agent,
        {
            "search_context": "search_context",        # 检索历史对话
            "web_search_node": "web_search_node",      # 网络搜索
            "retrieve_database": "retrieve_database",  # 检索知识库
            END: END                                    # 简单问题已回答
        }
    )
    
    # 路由4: search_context 后的分支
    # - 检索到历史对话 → second_agent_generate（基于历史生成回答）
    # - 未检索到 → retrieve_database（改为检索知识库）
    builder.add_conditional_edges(
        "search_context", 
        route_llm_or_retrieve_database,
        {
            "second_agent_generate": "second_agent_generate", 
            "retrieve_database": "retrieve_database"
        }
    )

    # 固定边3: second_agent_generate → evaluate_node
    # 基于历史对话的回答需要评估质量
    builder.add_edge("second_agent_generate", "evaluate_node")
    
    # 固定边4: retrieve_database → third_chatbot
    # 检索知识库后生成回答
    builder.add_edge("retrieve_database", "third_chatbot")
    
    # 路由5: third_chatbot 后的分支
    # - 只有图片输入 → END（RAGAS 不支持多模态评估，直接结束）
    # - 有文本输入 → evaluate_node（进行质量评估）
    builder.add_conditional_edges(
        "third_chatbot", 
        route_evaluate,
        {
            "evaluate_node": "evaluate_node", 
            END: END
        }
    )
    
    # 路由6: evaluate_node 后的分支
    # - 评分 < 0.75 → human_approval_node（触发人工审核）
    # - 评分 ≥ 0.75 → END（质量合格，直接结束）
    builder.add_conditional_edges(
        "evaluate_node", 
        route_human_answer_node,
        {
            "human_approval_node": "human_approval_node", 
            END: END
        }
    )
    
    # 路由7: human_approval_node 后的分支
    # - 人工批准 (approved) → END（结束流程）
    # - 人工拒绝 (rejected) → fourth_chatbot（启动网络搜索备选方案）
    builder.add_conditional_edges(
        "human_approval_node", 
        route_after_human_approval,
        {
            "fourth_chatbot": "fourth_chatbot", 
            END: END
        }
    )

    # 路由8: fourth_chatbot 后的分支（网络搜索工具调用）
    # - LLM 返回 tool_calls → web_search_node（执行网络搜索）
    # - LLM 不调用工具 → END（直接返回回答）
    builder.add_conditional_edges(
        "fourth_chatbot", 
        tools_condition,
        {
            "tools": "web_search_node",  # 需要搜索
            '__end__': END                # 无需搜索
        }
    )
    
    # 固定边5: web_search_node → fourth_chatbot
    # 搜索结果返回给 fourth_chatbot 继续生成回答（形成循环直到不再调用工具）
    builder.add_edge('web_search_node', 'fourth_chatbot')
    
    # 注意：web_search_node 被两个节点使用：
    # 1. first_agent_decision → web_search_node → END (直接结束，因为已获得实时信息)
    # 2. fourth_chatbot → web_search_node → fourth_chatbot (循环直到生成最终答案)
    # 当前配置：web_search_node 总是返回 fourth_chatbot，需要根据来源区分
    # 简化方案：统一让 web_search_node → fourth_chatbot → END
    
    # 编译图并返回
    return builder.compile(checkpointer=checkpointer, store=store)

def draw_graph(graph, output_dir: Path):
    """
    保存工作流图的 Mermaid 代码
    可以在 https://mermaid.live 查看
    """
    # 保存 Mermaid 代码
    mermaid_code = graph.get_graph().draw_mermaid()
    mermaid_file = output_dir / 'graph_rag.mmd'
    with open(mermaid_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    print(f"✅ Mermaid 代码已保存到: {mermaid_file}")
    print("   可以在 https://mermaid.live 粘贴代码查看图形")
    
    # 如果需要 PNG，尝试使用 API（可能失败）
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_file = output_dir / 'graph_rag.png'
        with open(png_file, 'wb') as f:
            f.write(png_bytes)
        print(f"✅ PNG 图片已保存到: {png_file}")
    except Exception as e:
        print(f"⚠️  PNG 生成失败: {str(e)[:100]}...")
        print("   请使用 Mermaid 代码在线查看")

async def execute_graph(user_input: str, session_id: str = None) -> dict:
    """
    执行工作流（支持中断和恢复）
    
    Args:
        user_input: 用户输入（文本/图片路径，或用 & 分隔）
        session_id: 会话ID（可选，用于恢复之前中断的会话）
    
    Returns:
        dict: 包含执行结果的字典
            - status: 'completed' | 'interrupted' | 'error'
            - session_id: 会话ID
            - answer: AI的最终回答（仅在completed时）
            - error: 错误信息（仅在error时）
    # 纯文本
    HumanMessage(content=[
        {"type": "text", "text": "什么是AI?"}
    ])

    # 图文混合
    HumanMessage(content=[
        {"type": "text", "text": "这是什么?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ])
    """
    # 1. 会话管理：如果没有提供session_id，创建新会话
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"创建新会话: {session_id}")
    else:
        logger.info(f"恢复会话: {session_id}")
    
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    # 2. 初始化 checkpointer 和 store，并在整个执行期间保持连接
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        # 设置数据库表（如果不存在则创建）
        await checkpointer.setup()
        await store.setup()
        
        # 构建图（传入 checkpointer 和 store）
        graph = build_graph(checkpointer, store)
        
        # 3. 解析用户输入（文本/图片）
        image_base64 = None
        text = None
        
        if '&' in user_input:
            # 情况1: 图文混合输入，格式 "文本 & 图片路径"
            text = user_input.split('&')[0].strip()  # 提取文本部分
            image = user_input.split('&')[1].strip()  # 提取图片路径
            if image and os.path.isfile(image): # 验证图片文件存在
                # 将图片转换为 base64 编码
                image_base64 = {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image)[0]},
                }
        elif os.path.isfile(user_input):
            # 情况2: 纯图片输入（用户直接输入图片路径）
            image_base64 = {
                "type": "image_url",
                "image_url": {"url": image_to_base64(user_input)[0]},
            }
        else:
            # 情况3: 纯文本输入
            text = user_input

        # 4. 构建消息 
        """
        纯文本消息：
        HumanMessage(content=[
        {"type": "text", "text": "什么是AI?"}
        ])

        图文混合消息：
        HumanMessage(content=[
            {"type": "text", "text": "这是什么?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ])
        """
        message = HumanMessage(content=[])  # 创建空的人类消息
        if text:  
            message.content.append({"type": "text", "text": text})  # 添加文本内容
        if image_base64:
            message.content.append(image_base64)   # 添加图片（base64编码）
        
        # 5. 执行工作流
        try:
            logger.info("开始执行工作流...")
            async for chunk in graph.astream(
                {'messages': [message]},   # 输入：包含用户消息的初始状态
                config,                         # 配置：包含thread_id（会话ID）
                stream_mode='updates',              # 流式输出：只返回本次更新的消息
                context=UserContext(user_name='zhangji')  # 运行时上下文：使用了 Runtime[UserContext]，表示期待 UserContext 类型的上下文
            ):
                # chunk 格式: {node_name: {'messages': [...]}}
                if chunk:
                    pretty_print_messages(chunk)
        except Exception as e:
            # 捕获任何执行错误
            logger.exception("工作流执行错误")  # 使用 exception 显示完整堆栈
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}\n\n完整堆栈:\n{traceback.format_exc()}"
            return {
                'status': 'error',
                'session_id': session_id,
                'error': error_detail
            }
        
        # 6. 检查工作流状态
        current_state = await graph.aget_state(config)
        
        # 6.1 工作流被中断（触发人工审核）
        if current_state.next:   # 如果 next 不为空，说明工作流被中断
            logger.info(f"工作流在节点 {current_state.next} 处中断，等待人工审核...")
            
            # 提取 interrupt 传递的详细信息（从 tasks 中获取）
            interrupt_data = {}
            if current_state.tasks:
                for task in current_state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        # 获取第一个 interrupt 的值
                        interrupt_data = task.interrupts[0].value if task.interrupts else {}
                        break
            
            # 提取中断时的状态信息
            evaluate_score = current_state.values.get('evaluate_score', 0.0)
            user_input = interrupt_data.get('user_input', current_state.values.get('input_text', ''))
            
            # 打印审核信息
            print("\n" + "="*80)
            print("🔔 工作流已暂停，需要人工审核")
            print("="*80)
            print(f"❓ 审核问题: {interrupt_data.get('question', '是否批准此回答？')}")
            print(f"📝 用户提问: {user_input}")
            print(f"📊 评估分数: {evaluate_score:.3f} (阈值: 0.75)")
            if interrupt_data.get('timestamp'):
                print(f"⏰ 时间戳: {interrupt_data.get('timestamp')}")
            print("="*80)

            
            # 等待用户决策（交互式输入）
            while True:
                user_decision = input("\n👉 Do you agree to this answer? (approve/rejected): ").strip().lower()
                
                if user_decision in ['approve', 'approved', 'y', 'yes', 'yep', '是', '批准']:
                    decision_value = True  # 批准
                    print("✅ 已批准，允许生成回答...")
                    break
                elif user_decision in ['reject', 'rejected', 'n', 'no', 'nah', 'nope', '否定', '拒绝']:
                    decision_value = False
                    print("❌ 已拒绝，将使用网络搜索提供备选答案...")
                    break
                else:
                    print(f"⚠️  无效输入 '{user_decision}'，请输入 'approve' 或 'rejected'")
            
            # 恢复工作流
            try:
                from langgraph.types import Command
                logger.info(f"使用决策 {decision_value} 恢复工作流...")
                
                # 恢复执行工作流
                async for chunk in graph.astream(
                    # 关键：使用 Command(resume=decision_value) 恢复执行
                    Command(resume=decision_value),    # 将用户决策传递给 interrupt()
                    config,                          # 使用相同的 thread_id
                    stream_mode='updates'            # 只返回本次更新的消息
                ):
                    # chunk 格式: {node_name: {'messages': [...]}}
                    if chunk:
                        pretty_print_messages(chunk)
            except Exception as e:
                logger.exception("恢复工作流时出错")
                import traceback
                error_detail = f"恢复失败: {type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
                return {
                    'status': 'error',
                    'session_id': session_id,
                    'error': error_detail
                }
            
            # 重新获取最终状态（因为恢复后工作流继续执行了）
            current_state = await graph.aget_state(config)
        
        # 6.2 工作流正常结束
        mess = current_state.values.get('messages', [])   # 从状态中获取所有消息 人工审核之后的最新消息
        final_answer = mess[-1].content if mess and isinstance(mess[-1], AIMessage) else "无回答" # 提取最后一条AI消息作为最终答案
        
        # 获取人工审核状态
        human_answer = current_state.values.get('human_answer')
        
        # 7. 判断是否需要写入 Milvus（只保存有价值的上下文）
        # 写入条件：
        # 1. 知识库检索 + 评分合格（evaluate_score >= 0.75）
        # 2. 知识库检索 + 人工批准（human_answer == 'approved'）
        # 3. 网络搜索返回的答案（检测 messages 中是否有 web_search 的 ToolMessage）
        should_save_to_milvus = False
        save_reason = ""
        
        if mess and isinstance(mess[-1], AIMessage):
            evaluate_score = current_state.values.get('evaluate_score')
            
            # 检查是否使用了网络搜索（查看 messages 中是否有 ToolMessage 且 name == 'web_search'）
            has_web_search = any(
                hasattr(msg, 'name') and msg.name == 'web_search' 
                for msg in mess 
                if msg.__class__.__name__ == 'ToolMessage'
            )
            
            # 情况1: 知识库检索 + 评分合格（未经过人工审核，直接通过）
            if evaluate_score is not None and evaluate_score >= 0.75 and human_answer is None:
                should_save_to_milvus = True
                save_reason = f"知识库检索回答（评分: {evaluate_score:.3f} ≥ 0.75）"
            
            # 情况2: 知识库检索 + 人工批准
            elif human_answer == 'approved':
                should_save_to_milvus = True
                score_str = f"{evaluate_score:.3f}" if evaluate_score is not None else "N/A"
                save_reason = f"知识库检索回答（人工批准，评分: {score_str}）"
            
            # 情况3: 网络搜索返回的答案
            elif has_web_search:
                should_save_to_milvus = True
                if human_answer == 'rejected':
                    save_reason = "网络搜索备选答案（人工拒绝原答案后启用）"
                else:
                    save_reason = "网络搜索回答（实时查询结果）"
            
            # 执行写入
            if should_save_to_milvus:
                logger.info(f"开始写入Milvus... 原因: {save_reason}")
                asyncio.create_task(
                    get_milvus_writer().async_insert(
                        context_text=mess[-1].content,       # 保存最终答案的上下文
                        user=current_state.values.get('user', 'zhangji'),  # 保存用户名
                        message_type="AIMessage"  # 保存消息类型
                    )
                )
            else:
                logger.info("跳过写入Milvus（简单问答，无需保存历史）")
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'answer': final_answer,
            'human_answer': human_answer  # None | 'approved' | 'rejected'
        }


async def main():
    """
    交互式主函数（终端CLI模式）
    支持工作流中断和人工审核
    用户自定义会话ID，简单直接
    """
    print("\n" + "="*80 + "\n")
    print("🤗(●'◡'●) 喵喵喵~欢迎使用多模态RAG系统 - 交互式终端")
    print("\n" + "="*80 + "\n")
    
    # 🔥 让用户输入会话ID
    print("📝 请输入会话ID（用于标识本次对话，下次输入相同ID可继续对话）")
    print("💡 提示: 可以使用中文或英文，例如: 项目讨论、学习笔记、my_project 等")
    print("💡 直接回车将使用默认ID: default\n")
    
    session_input = input("🔖 会话ID: ").strip()
    
    # 如果用户没有输入，使用默认ID
    if not session_input:
        session_input = "default"
        print(f"✅ 使用默认会话ID: {session_input}\n")
    else:
        print(f"✅ 使用会话ID: {session_input}\n")
    
    # 生成完整的 session_id（用户输入 + 用户名前缀，避免多用户冲突）
    user_name = "zhangji"
    session_id = f"{user_name}_{session_input}"
    
    logger.info(f"📝 会话ID: {session_id}")
    
    print("以下是使用说明，你可以按照这些指令进行输入:")
    print("  - 纯文本输入: 直接输入问题")
    print("  - 纯图片输入: 输入图片路径")
    print("  - 图文混合: 文本 & 图片路径")
    print("  - 退出程序: 输入 exit/quit/退出")
    print("\n" + "="*80 + "\n")
    
    while True:
        try:
            user_input = input('💬 大人请告诉我您的问题: ').strip()
        
            if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                print(f"\n 😊 喵喵喵~ 再见！会话 '{session_input}' 已保存~")
                print(f"💡 下次启动输入相同的会话ID可继续本次对话")
                break
            
            if not user_input:
                print("⚠️  uppus！ 输入不能为空，喵喵喵~ 请重新输入")
                continue
            
            # 🔥 执行工作流，传入相同的 session_id 保持会话连续性
            result = await execute_graph(user_input, session_id=session_id)
            
            # 处理执行结果
            if result['status'] == 'completed':
                
                # 只有经过人工审核时才显示审核状态
                human_answer = result.get('human_answer')
                if human_answer == 'approved':
                    print("\n📋 大人的审核结果: ✅ 已批准")
                elif human_answer == 'rejected':
                    print("\n📋 大人的审核结果: ❌ 已拒绝（使用了网络搜索备选方案）")
                # 如果 human_answer 为 None，说明未经过人工审核（简单问题直接回答），不显示审核信息
                
                print("="*80 + "\n")
            
            elif result['status'] == 'error':
                print("\n" + "="*80)
                print("🏖️ 呜呜呜~ 工作流执行失败")
                print("="*80)
                print(f"错误详情:喵喵喵~ 系统出错了，请稍后再试:\n{result['error']}")
                print("="*80 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 检测到中断信号，退出程序...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            logger.exception("主循环异常:")
        

if __name__ == "__main__":
    # 运行交互式终端
    asyncio.run(main())
    
    # 如果需要保存工作流图，取消下面的注释：
    # async def save_graph():
    #     async with (
    #         AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    #         AsyncPostgresStore.from_conn_string(DB_URI) as store,
    #     ):
    #         await checkpointer.setup()
    #         await store.setup()
    #         graph = build_graph(checkpointer, store)
    #         output_dir = project_root / 'static'
    #         draw_graph(graph, output_dir)
    # asyncio.run(save_graph())


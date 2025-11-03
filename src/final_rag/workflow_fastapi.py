"""
FastAPI 专用的工作流执行器
提供两阶段审批机制：
1. execute_graph_for_api: 执行工作流，遇到中断时返回中断信息
2. resume_graph_for_api: 根据审批决策恢复工作流执行
"""

import sys
from pathlib import Path
import asyncio
import platform

# Windows 平台需要设置事件循环策略以支持 psycopg 异步
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
import os
import uuid
import logging

# 导入原 workflow 中的组件
from src.final_rag.workflow import build_graph
from src.final_rag.utils.nodes import UserContext
from utils.embeddings_utils import image_to_base64
from src.config import settings

logger = logging.getLogger(__name__)

# 从配置文件读取 LangGraph 数据库 URI
DB_URI = settings.POSTGRES.LANGGRAPH_DB.URI


async def execute_graph_for_api(user_input: str, session_id: str = None, user_name: str = 'zhangji') -> dict:
    """
    为 FastAPI 执行工作流（支持中断）
    
    与原 execute_graph 的区别：
    - 遇到中断时直接返回中断信息，不等待用户输入
    - 不打印消息到终端（logger 除外）
    - 返回格式为标准的 dict，方便 FastAPI 序列化
    
    Args:
        user_input: 用户输入（文本/图片路径，或用 & 分隔）
        session_id: 会话ID（可选，不提供则创建新会话）
        user_name: 用户名（默认 zhangji）
    
    Returns:
        dict: 包含执行结果的字典
            - status: 'completed' | 'interrupted' | 'error'
            - session_id: 会话ID
            
            当 status='completed':
                - answer: AI的最终回答
                - human_answer: 人工审核结果 (None | 'approved' | 'rejected')
                - evaluate_score: 评估分数
            
            当 status='interrupted':
                - question: 审核问题
                - user_input: 用户输入
                - evaluate_score: 评估分数
                - current_answer: 当前答案预览
            
            当 status='error':
                - error: 错误信息
    """
    # 1. 会话管理
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"创建新会话: {session_id}")
    else:
        logger.info(f"使用会话: {session_id}")
    
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    # 2. 初始化数据库连接
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        
        # 构建图
        graph = build_graph(checkpointer, store)
        
        # 3. 解析用户输入（文本/图片）
        image_base64 = None
        text = None
        
        if '&' in user_input:
            # 图文混合
            text = user_input.split('&')[0].strip()
            image = user_input.split('&')[1].strip()
            if image and os.path.isfile(image):
                image_base64 = {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image)[0]},
                }
        elif os.path.isfile(user_input):
            # 纯图片
            image_base64 = {
                "type": "image_url",
                "image_url": {"url": image_to_base64(user_input)[0]},
            }
        else:
            # 纯文本
            text = user_input
        
        # 4. 构建消息
        message = HumanMessage(content=[])
        if text:
            message.content.append({"type": "text", "text": text})
        if image_base64:
            message.content.append(image_base64)
        
        # 5. 执行工作流
        try:
            logger.info(f"开始执行工作流 - session_id: {session_id}")
            async for chunk in graph.astream(
                {'messages': [message]},
                config,
                stream_mode='updates',
                context=UserContext(user_name=user_name)
            ):
                # FastAPI 模式：不打印到终端，只记录日志
                if chunk:
                    logger.debug(f"工作流更新: {list(chunk.keys())}")
        except Exception as e:
            logger.exception("工作流执行错误")
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            return {
                'status': 'error',
                'session_id': session_id,
                'error': error_detail
            }
        
        # 6. 检查工作流状态
        current_state = await graph.aget_state(config)
        
        # 6.1 工作流被中断（需要人工审批）
        if current_state.next:
            logger.info(f"⏸️  工作流中断 - 等待人工审批 - session_id: {session_id}")
            
            # 提取中断信息
            interrupt_data = {}
            if current_state.tasks:
                for task in current_state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        interrupt_data = task.interrupts[0].value if task.interrupts else {}
                        break
            
            # 提取状态信息
            evaluate_score = current_state.values.get('evaluate_score', 0.0)
            user_input_text = interrupt_data.get('user_input', current_state.values.get('input_text', ''))
            
            # 获取当前答案预览
            messages = current_state.values.get('messages', [])
            current_answer = None
            if messages and isinstance(messages[-1], AIMessage):
                current_answer = messages[-1].content
            
            return {
                'status': 'interrupted',
                'session_id': session_id,
                'question': interrupt_data.get('question', '是否批准此回答？'),
                'user_input': user_input_text,
                'evaluate_score': evaluate_score,
                'current_answer': current_answer
            }
        
        # 6.2 工作流正常结束
        messages = current_state.values.get('messages', [])
        final_answer = messages[-1].content if messages and isinstance(messages[-1], AIMessage) else "无回答"
        human_answer = current_state.values.get('human_answer')
        evaluate_score = current_state.values.get('evaluate_score')
        
        logger.info(f"✅ 工作流完成 - session_id: {session_id}")
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'answer': final_answer,
            'human_answer': human_answer,
            'evaluate_score': evaluate_score
        }


async def resume_graph_for_api(session_id: str, decision: bool) -> dict:
    """
    为 FastAPI 恢复工作流执行（第二阶段）
    
    用于处理人工审批后的工作流恢复
    
    Args:
        session_id: 会话ID（必须是之前中断的会话）
        decision: 审批决策（True=批准, False=拒绝）
    
    Returns:
        dict: 包含执行结果的字典
            - status: 'completed' | 'error'
            - session_id: 会话ID
            - answer: AI的最终回答
            - human_answer: 人工审核结果 ('approved' | 'rejected')
            - evaluate_score: 评估分数
            - error: 错误信息（仅在error时）
    """
    logger.info(f"恢复工作流 - session_id: {session_id}, decision: {decision}")
    
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    # 初始化数据库连接
    async with (
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        
        # 构建图
        graph = build_graph(checkpointer, store)
        
        # 恢复执行
        try:
            logger.info(f"使用决策 {decision} 恢复工作流...")
            
            async for chunk in graph.astream(
                Command(resume=decision),  # 将审批决策传递给 interrupt()
                config,
                stream_mode='updates'
            ):
                # FastAPI 模式：不打印到终端
                if chunk:
                    logger.debug(f"工作流更新: {list(chunk.keys())}")
        except Exception as e:
            logger.exception("恢复工作流时出错")
            import traceback
            error_detail = f"恢复失败: {type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            return {
                'status': 'error',
                'session_id': session_id,
                'error': error_detail
            }
        
        # 获取最终状态
        current_state = await graph.aget_state(config)
        
        # 提取结果
        messages = current_state.values.get('messages', [])
        final_answer = messages[-1].content if messages and isinstance(messages[-1], AIMessage) else "无回答"
        human_answer = current_state.values.get('human_answer')
        evaluate_score = current_state.values.get('evaluate_score')
        
        logger.info(f"✅ 工作流恢复完成 - session_id: {session_id}")
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'answer': final_answer,
            'human_answer': human_answer,
            'evaluate_score': evaluate_score
        }


# 为了保持向后兼容，导出别名
execute_graph = execute_graph_for_api
resume_graph = resume_graph_for_api


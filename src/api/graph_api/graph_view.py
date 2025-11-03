# FastAPI 路由：多模态 RAG 聊天接口
from fastapi import APIRouter, HTTPException
from typing import Union
from src.api.graph_api.graph_schema import (
    ChatRequest, 
    ChatResponse, 
    InterruptResponse,
    ApprovalRequest
)
from src.final_rag.workflow_fastapi import execute_graph_for_api, resume_graph_for_api
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/graph', tags=['多模态RAG'])


@router.post('/chat', response_model=Union[ChatResponse, InterruptResponse])
async def chat(request: ChatRequest):
    """
    多模态 RAG 聊天接口（完整版，支持中断）
    
    流程：
    1. 接收前端请求（text/image_path/session_id/user_name）
    2. 构建 user_input 字符串（兼容原 execute_graph 的格式）
    3. 调用 execute_graph() 执行工作流
    4. 根据执行结果返回：
       - ChatResponse: 正常完成
       - InterruptResponse: 需要人工审批
    
    Args:
        request: ChatRequest
            - text: 文本输入（可选）
            - image_path: 图片路径（可选）
            - session_id: 会话ID（可选，不传则创建新会话）
            - user_name: 用户名（默认 zhangji）
    
    Returns:
        ChatResponse | InterruptResponse:
            - ChatResponse: 执行完成时返回
            - InterruptResponse: 触发人工审批时返回
    """
    try:
        # 1. 验证输入（至少要有 text 或 image_path）
        if not request.text and not request.image_path:
            raise HTTPException(
                status_code=400, 
                detail="请提供 text 或 image_path 中的至少一个"
            )
        
        # 2. 构建 user_input（兼容原 execute_graph 的字符串格式）
        user_input = ""
        if request.text and request.image_path:
            user_input = f"{request.text} & {request.image_path}"
        elif request.text:
            user_input = request.text
        else:
            user_input = request.image_path
        
        logger.info(f"📝 收到聊天请求 - user_input: {user_input[:100]}...")
        
        # 3. 生成或使用 session_id
        session_id = request.session_id or f"{request.user_name}_{str(uuid.uuid4())[:8]}"
        logger.info(f"🔖 会话ID: {session_id}")
        
        # 4. 调用工作流（FastAPI 专用版本）
        result = await execute_graph_for_api(
            user_input=user_input,
            session_id=session_id,
            user_name=request.user_name
        )
        
        # 5. 处理错误状态
        if result['status'] == 'error':
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', '工作流执行失败')
            )
        
        # 6. 处理中断状态（需要人工审批）
        if result['status'] == 'interrupted':
            logger.info(f"⏸️  工作流中断，等待人工审批 - session_id: {session_id}")
            return InterruptResponse(
                status='interrupted',
                session_id=result['session_id'],
                question=result.get('question', '是否批准此回答？'),
                user_input=result.get('user_input', user_input),
                evaluate_score=result.get('evaluate_score', 0.0),
                current_answer=result.get('current_answer')
            )
        
        # 7. 处理完成状态
        logger.info(f"✅ 工作流执行完成 - session_id: {session_id}")
        return ChatResponse(
            status='completed',
            session_id=result['session_id'],
            answer=result.get('answer'),
            human_answer=result.get('human_answer'),
            error=None,
            evaluate_score=result.get('evaluate_score')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ 聊天接口异常")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post('/approval', response_model=ChatResponse)
async def approval(request: ApprovalRequest):
    """
    人工审批接口（第二阶段）
    
    当 /chat 接口返回 InterruptResponse 时，前端调用此接口提交审批决策
    
    流程：
    1. 接收审批请求（session_id + decision）
    2. 调用 resume_graph() 恢复工作流执行
    3. 返回最终结果
    
    Args:
        request: ApprovalRequest
            - session_id: 会话ID（必须与之前中断的会话ID一致）
            - decision: 审批决策（approve/reject）
    
    Returns:
        ChatResponse: 恢复执行后的最终结果
    """
    try:
        logger.info(f"📋 收到审批请求 - session_id: {request.session_id}, decision: {request.decision}")
        
        # 1. 将决策转换为布尔值
        decision_value = (request.decision == "approve")
        
        # 2. 恢复工作流执行（FastAPI 专用版本）
        result = await resume_graph_for_api(
            session_id=request.session_id,
            decision=decision_value
        )
        
        # 3. 处理错误状态
        if result['status'] == 'error':
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', '恢复工作流失败')
            )
        
        # 4. 返回最终结果
        logger.info(f"✅ 工作流恢复完成 - session_id: {request.session_id}")
        return ChatResponse(
            status='completed',
            session_id=result['session_id'],
            answer=result.get('answer'),
            human_answer=result.get('human_answer'),
            error=None,
            evaluate_score=result.get('evaluate_score')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ 审批接口异常")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


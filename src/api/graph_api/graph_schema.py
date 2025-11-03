# 声明在调用api接口的请求或者响应数据的格式
from pydantic import BaseModel, Field
from typing import Optional, Literal


class ChatRequest(BaseModel):
    """聊天请求"""
    text: Optional[str] = Field(None, description='用户输入的文本')
    image_path: Optional[str] = Field(None, description='用户输入的图片路径')
    session_id: Optional[str] = Field(None, description='会话ID，用于恢复历史对话')
    user_name: str = Field(default="zhangji", description="用户名")
    
    class Config:        # 为 API 文档提供示例
        json_schema_extra = {
            "example": {
                "text": "什么是多智能体系统？",
                "image_path": None,
                "session_id": "zhangji_项目讨论",
                "user_name": "zhangji"
            }
        }


class ChatResponse(BaseModel):
    """聊天响应"""
    status: Literal["completed", "interrupted", "error"] = Field(..., description="执行状态")
    session_id: str = Field(..., description="会话ID")
    answer: Optional[str] = Field(None, description="AI的最终回答")
    human_answer: Optional[Literal["approved", "rejected"]] = Field(None, description="人工审核结果")
    error: Optional[str] = Field(None, description="错误信息")
    evaluate_score: Optional[float] = Field(None, description="RAGAS评分")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "session_id": "zhangji_项目讨论",
                "answer": "多智能体系统是指由多个智能体组成的分布式系统...",
                "human_answer": None,
                "error": None,
                "evaluate_score": 0.85
            }
        }


class InterruptResponse(BaseModel):
    """中断响应（需要人工审批）"""
    status: Literal["interrupted"]
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="审核问题")
    user_input: str = Field(..., description="用户提问")
    evaluate_score: float = Field(..., description="评估分数")
    current_answer: Optional[str] = Field(None, description="当前答案预览")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "interrupted",
                "session_id": "zhangji_项目讨论",
                "question": "是否批准此回答？",
                "user_input": "什么是容错控制？",
                "evaluate_score": 0.65,
                "current_answer": "容错控制是一种..."
            }
        }


class ApprovalRequest(BaseModel):
    """人工审批请求"""
    session_id: str = Field(..., description="会话ID")
    decision: Literal["approve", "reject"] = Field(..., description="审批决策")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "zhangji_项目讨论",
                "decision": "approve"
            }
        }
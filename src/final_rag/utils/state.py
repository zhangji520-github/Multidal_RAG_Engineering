from typing import Literal
from langgraph.graph import MessagesState
from typing import Optional, List, Dict

class MultidalModalRAGState(MessagesState):
    """状态数据结构类schema"""

    input_type: Literal["has_text", "only_image"]      # 用户输入的类型
    context_retrieved: Optional[List[Dict[str, str]]] = None  # 从向量数据库检索到的上下文
    images_retrieved: Optional[List[str]]                     # 从向量数据库检索到的图片路径

    needs_retrieval: Optional[bool] = False              # 是否需要从向量数据库检索上下文
    evaluate_score: Optional[float] = None               # 评估分数
    final_response: Optional[str]                 # 最终的响应

    input_image: Optional[str]                    # 用户输入的图片，里面存储的是base64编码的图片
    input_text: Optional[str]                    # 用户输入的文本
    user: str = "zhangjishuaige"                      # 用户名

    human_answer: Optional[str] = None # 人工审核结果: None(未审核) | 'approved'(批准) | 'rejected'(拒绝)
    
    # 摘要相关字段
    summary: Optional[str] = None                 # 对话历史摘要
    message_count: int = 0                        # 当前消息数量（用于判断是否需要摘要）

# 自定义异常类
class InvalidInputError(Exception):
    """自定义异常，用于表示无效输入"""
    def __init__(self, message: str, error_code: int = 400):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
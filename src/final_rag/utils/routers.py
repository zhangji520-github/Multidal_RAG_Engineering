from langgraph.graph import END
from src.final_rag.utils.state import MultidalModalRAGState
from langchain_core.messages import AIMessage


def route_after_first_agent(state: MultidalModalRAGState):
    """
    first_agent_decision 之后的路由
    
    判断逻辑：
    1. 如果调用了 search_context 工具 → search_context 节点
    2. 如果调用了 web_search 工具 → web_search_node 节点
    3. 如果没有 tool_calls：
       - 检查用户原始输入是否包含显式检索关键词 → retrieve_database
       - LLM 给出了实质性回答 → END（简单问题已回答）
       - LLM 回答很短或像是拒绝回答 → retrieve_database（检索知识库）
    """
    messages = state.get("messages", [])
    if not messages:
        return "retrieve_database"
    
    last_message = messages[-1]
    
    # 检查是否调用了工具
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_calls = last_message.tool_calls
        if tool_calls:
            tool_name = tool_calls[0].get("name", "")
            # 根据工具名称路由到不同节点
            if tool_name == "search_context":
                return "search_context"
            elif tool_name == "web_search":
                return "web_search_node"  # 使用现有的 web_search_node
            else:
                return "search_context"  # 默认路由
    
    # 获取用户原始输入（检查是否有显式检索意图）
    user_input = (state.get("input_text") or "").lower()
    
    # 【优先级1】明确的知识库检索关键词（用户明确说要检索知识库）
    knowledge_base_keywords = [
        "检索知识库", "检索数据库", "search database", "search knowledge base",
        "查询知识库", "查询数据库", "搜索知识库", "check database"
    ]
    
    # 【优先级2】技术/专业内容关键词（暗示需要查询知识库）
    domain_keywords = [
        "gpt-4", "gpt4", "技术报告", "technical report", "rlhf", "reinforcement learning",
        "exam benchmark", "capability", "appendix", "实验", "benchmark",
        "论文", "paper", "研究", "research"
    ]
    
    # ✅ 优先级1：用户明确要求检索知识库 → 直接路由到 retrieve_database
    if any(keyword in user_input for keyword in knowledge_base_keywords):
        return "retrieve_database"
    
    # ✅ 优先级2：专业/技术内容 → 路由到 retrieve_database
    if any(keyword in user_input for keyword in domain_keywords):
        return "retrieve_database"
    
    # 没有调用工具，检查回答质量
    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content.strip()
        
        # 判断是否是实质性回答
        # 简单启发式：如果回答较长且包含句号或问号，认为是完整回答
        if len(content) > 20 and ('.' in content or '。' in content or '?' in content or '！' in content):
            return END  # 简单问题已回答
        else:
            return "retrieve_database"  # 可能需要更多信息，检索知识库
    
    return "retrieve_database"

def route_llm_or_retrieve_database(state: MultidalModalRAGState):
    """
    search_context 之后的路由
    检查是否检索到历史对话上下文
    """
    messages = state.get("messages", [])
    if messages:
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")
    
    if not tool_message.content or tool_message.content == "no context found":
        return "retrieve_database"
    else:
        return "second_agent_generate"

def route_evaluate(state: MultidalModalRAGState):
    """
    路由评价 如果用户仅仅输入图片，则不进行评估(目前RAGAS不支持多模态评估) 其他情况下进入评估节点
    """
    if state.get("input_type") == "only_image":
        return END
    else:
        return "evaluate_node"    # 评估节点

def route_human_answer_node(state: MultidalModalRAGState):
    """
    路由人工审核节点
    
    规则：评分 < 0.75 → 人工审核，评分 >= 0.75 → 直接通过
    （适用于所有答案类型：知识库检索、网络搜索等）
    """
    if state.get("evaluate_score", 0) < 0.75:
        return "human_approval_node"
    else:
        return END

def route_after_human_approval(state: MultidalModalRAGState):
    """
    人工审核后的路由逻辑
    - 如果批准(approved)，则结束流程
    - 如果拒绝(rejected)，检查是否已使用过网络搜索
      - 首次拒绝 → 调用网络搜索（fourth_chatbot）
      - 二次拒绝（网络搜索结果也不满意）→ 直接结束（避免无限循环）
    """
    human_answer = state.get("human_answer", "rejected")
    if human_answer == "approved":
        return END
    else:
        # 检查消息历史中是否已经有网络搜索的痕迹
        # 判断依据：是否有 web_search 工具的 ToolMessage
        messages = state.get("messages", [])
        has_web_search = any(
            hasattr(msg, 'name') and msg.name == 'web_search' 
            for msg in messages
        )
        
        if has_web_search:
            # 已经用过网络搜索了，不再重试，直接结束
            return END
        else:
            # 首次拒绝，尝试网络搜索
            return "fourth_chatbot"


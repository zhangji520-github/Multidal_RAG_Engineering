import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataclasses import dataclass
import logging
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import RunnableConfig
from langgraph.runtime import Runtime

from src.final_rag.utils.state import InvalidInputError, MultidalModalRAGState
from src.final_rag.utils.prompt import CONTEXT_SYSTEM_PROMPT, ANSWER_GENERATION_PROMPT, RETRIEVER_GENERATE_SYSTEM_PROMPT
from src.final_rag.utils.tools import  web_tools
from llm_utils import qwen3_vl_plus, qwen3_max
from langchain_core.messages import SystemMessage, AIMessage
from env_utils import COLLECTION_NAME, MILVUS_URI
from milvus_db.milvus_retrieve import MilvusRetriever
from pymilvus import MilvusClient
from utils.embeddings_utils import call_dashscope_once
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy
from langgraph.types import interrupt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


m_retriever = MilvusRetriever(collection_name=COLLECTION_NAME, milvus_client=MilvusClient(uri=MILVUS_URI, user='root', password='Milvus'))
@dataclass
class UserContext:
    user_name: str

def process_input(state: MultidalModalRAGState, config: RunnableConfig, runtime:Runtime[UserContext]):
    """处理用户输入
    config: RunnableConfig 包含配置信息（如 thread_id ）和追踪信息（如 tags ）的 RunnableConfig 对象 config["configurable"]["thread_id"]
    runtime: Runtime[UserContext] 包含运行时 Runtime 及其他信息（如 context 和 store ）的对象 runtime.context.user_name
    """
    user_name = runtime.context.user_name  # UserContext是dataclass，直接访问属性
    last_message = state["messages"][-1]
    
    input_type = 'has_text'
    text_context = None
    image_url = None

    # 检查输入类型
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):       # 多模态消息是列表 比如 input = [{'image': 'xxxx', 'text': 'yyyy'}]
            content = last_message.content
            for item in content:
                # 提取文本内容
                if item.get("type") == "text":
                    text_context = item.get("text", None)
                
                # 提取图片URL
                elif item.get("type") == "image_url":
                    url = item.get("image_url", "").get('url')
                    if url:              # 图片的base64格式的字符串 
                        image_url = url
    
    # 打印简化的用户输入信息（不包含 base64 数据）
    if text_context and image_url:
        logger.info(f"   文本内容: {text_context[:50]}..." if len(text_context) > 50 else f"   文本内容: {text_context}")
    elif text_context:
        logger.info(f"🤗主人 {user_name} 发送消息: {text_context}")
    elif image_url:
        logger.info(f"🤗主人 {user_name} 发送消息: [纯图片]")
    
    else:
        raise InvalidInputError(f"Invalid input type: {type(last_message)}")

    # 判断输入类型
    if text_context and image_url:
        input_type = 'has_text'  # 图文混合，也算有文本
    elif text_context and not image_url:
        input_type = 'has_text'  # 纯文本
    elif image_url and not text_context:
        input_type = 'only_image'  # 纯图片
    else:
        # 既没有文本也没有图片，这是错误情况
        raise InvalidInputError("Input must contain either text or image")
                
    # 如果想把什么样的数据更新到自己定义的状态中，请返回一个字典，按照你自己定义的schema来
    return {
        "input_type": input_type,
        "input_text": text_context,  # 修改为 input_text，与 state 定义一致
        "input_image": image_url,    # 修改为 input_image，与 state 定义一致
        "user": user_name,
    }

#  自定义是为了替代：由LangGraph框架自带的ToolNode（有大模型动态传参 来调用工具） 这个很好写，主要还是tool的逻辑 
class SearchContextToolNode:
    """自定义类，来真正执行搜索上下文工具 通过对齐上一条AIMessage返回的Toolcall字段的信息来调用对应的工具 tools_by_name[tool_call["name"]].invoke()"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    # inputs 就是这个 自定义 state (自定义schema) 的实例
    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if messages:
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            # 使用LLM推理出的args
            tool_args = tool_call["args"].copy()
            
            # 只补充LLM无法知道的user_name（从runtime context注入到state中）
            if "user_name" not in tool_args or tool_args["user_name"] is None:
                tool_args["user_name"] = inputs.get("user")
            
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_args)
            
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# 检索数据库节点
def retrieve_database(state: MultidalModalRAGState):
    """
    检索数据库节点
    Args:
        state: MultidalModalRAGState 状态
    """
    if state.get("input_type") == "has_text":
        # 构建文本输入数据
        input_data = [{'text': state.get("input_text")}]
        ok, dense_embedding, status, retry_after = call_dashscope_once(input_data)
        results = m_retriever.hybrid_search(dense_embedding, state.get("input_text"), sparse_weight=0.8, dense_weight=1, limit=3)

    else:
        # 构建图像输入数据
        input_data = [{'image': state.get("input_image")}]
        ok, dense_embedding, status, retry_after = call_dashscope_once(input_data)      # 图像仅支持密集向量检索的方式
        results = m_retriever.dense_search(dense_embedding, limit=3)
    
    # logger.info(f"从知识数据库检索到的结果为: {results}")

    # 返回文档内容
    images = []           # 图片的实际路径+图片的摘要
    docs = []
    # print(results) 
    for hit in results:
        if hit.get('category') == 'image':        # 根据数据库的category字段判断是图片还是文本
            images.append({
                'image_path': hit.get('image_path'),
                'image_summary': hit.get('text'),           # 数据库的 'category' 的'text' 字段是我结合llm生成的图片的摘要
                'category': hit.get('category'),
            })
        else:
            docs.append({
                'text': hit.get('text'),
                'category': hit.get('category'),
                'filename': hit.get('filename'),
                'filetype': hit.get('filetype'),
                'image_path': hit.get('image_path'),
                'title': hit.get('title'),
                })
    # 根据自定义的state，放入对应的信息  要啥我就返回啥 很easy
    return {"context_retrieved": docs, "images_retrieved": images}


# 第一个agent决策节点
def first_agent_decision(state: MultidalModalRAGState):
    """
    第一个agent决策节点 
    
    功能：
    - 可以调用 search_context（检索历史对话）
    - 可以调用 web_search（网络搜索实时信息）
    - 可以直接回答简单问题
    
    Args:
        state: MultidalModalRAGState 状态
    Returns:
        如果llm决定调用工具 返回带有tool_call字段的AIMessage
        如果llm决定不调用工具 返回不带有tool_call字段的AIMessage
    """
    # 检查用户是否明确要求检索上下文
    user_input = (state.get("input_text") or "").lower()
    explicit_context_keywords = ["检索上下文", "检索历史", "search context", "check history", "search my context"]
    
    # 如果用户明确要求检索上下文，强制调用 search_context 工具
    if any(keyword in user_input for keyword in explicit_context_keywords):
        from langchain_core.messages import AIMessage
        # 提取查询内容（去掉"检索上下文"等关键词后的内容）
        query = user_input
        for keyword in explicit_context_keywords:
            query = query.replace(keyword, "").strip().strip("，,")
          
        # 构造强制的 tool_call
        return {
            'messages': [AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_context",
                    "args": {"query": query},
                    "id": f"forced_search_context_{hash(query)}"
                }]
            )]
        }
    
    # 1.绑定所有工具给llm（历史上下文 + 网络搜索）
    from src.final_rag.utils.tools import all_tools
    llm_with_tools = qwen3_vl_plus.bind_tools(all_tools)
    return {
        'messages': llm_with_tools.invoke([
            SystemMessage(content=CONTEXT_SYSTEM_PROMPT),
        ] + state["messages"])
    }

# 第二次生成回复（基于检索历史上下文 生成回复, 检索到的历史上下文在ToolMessage里面）
def second_agent_generate(state: MultidalModalRAGState):
    
    """
    第二次生成回复（基于检索用户历史上下文 生成回复, 检索到的用户历史上下文在SearchContextToolNode工具节点实现的ToolMessage里面）
    Args:
        state: MultidalModalRAGState 状态
    Returns:
        str: 第二次生成回复
    """
    # 添加系统提示词，指导模型如何基于检索到的上下文生成回复
    messages_with_prompt = [SystemMessage(content=ANSWER_GENERATION_PROMPT)] + state["messages"]
    return {'messages': [qwen3_vl_plus.invoke(messages_with_prompt)]}

# 第三次回复 (基于从知识库的上下文 进行回复 markdown格式输出，因为既有图片也有文字，图片用markdown语法展示 检索到的结果在状态里面)
def third_chatbot(state: MultidalModalRAGState):
    """
    处理多模态请求并返回Markdown格式的结果
    """
    context_retrieved = state.get("context_retrieved", [])
    images_retrieved = state.get("images_retrieved", [])

    # 格式化处理文本内容的上下文
    count = 0
    context_pieces = []
    for hit in context_retrieved:
        count += 1
        context_pieces.append(f"\n上下文{count}:\n {hit.get('text')} \n 资料来源: {hit.get('filename')}")
    context = "\n".join(context_pieces) if context_pieces else "no context found"         # 构建检索到的最终文本上下文

    # 格式化处理图片内容的上下文
    image_count = 0
    image_pieces = []
    for image in images_retrieved:
        image_count += 1
        image_pieces.append(f"\n图片{image_count}:\n {image.get('image_summary')} \n 资料来源: {image.get('image_path')}")
    images = "\n".join(image_pieces) if image_pieces else "no image found"         # 构建检索到的最终图片上下文

    input_text = state.get("input_text", "")
    input_image = state.get("input_image", "")

    # 构建用户消息
    user_content = []
    if input_text:
        user_content.append({'type': 'text', 'text': input_text})
    if input_image:
        # input_image 已经是完整的 base64 URL 字符串，需要包装成正确的格式
        user_content.append({'type': 'image_url', 'image_url': {'url': input_image}})

    # 提示词的撰写需要参考你格式化之后传入的上下文信息，用好这些信息达到你的目的
    prompt = ChatPromptTemplate.from_messages([
        ('system', RETRIEVER_GENERATE_SYSTEM_PROMPT),
        ('user', user_content),
    ])

    chain = prompt | qwen3_vl_plus

    response = chain.invoke({'context': context, 'images': images})   # 把格式化好的文本以及图片上下文传入到提示词中

    return {'messages': [response]}

# 评估节点
async def evaluate_node(state: MultidalModalRAGState):
    """评估大模型的响应和用户输入之间的相关性"""
    context_retrieved = state.get("context_retrieved", [])
    input_text = state.get("input_text", "")
    last_message = state["messages"][-1]      # 大模型的响应
    if isinstance(last_message, AIMessage):
        answer = last_message.content         # 拿到检索生成节点生成的文字回答
    
    # 1.创建评估样本SingleTurnSample
    sample = SingleTurnSample(
        user_input=input_text,          # 用户输入的问题
        retrieved_contexts=[f"上下文{i+1}: {context['text']}" for i, context in enumerate(context_retrieved)],    # 检索到的上下文 text 字段是我们需要的
        response=answer,            # RAG模型生成的答案
    )
    # 2.创建评估指标
    # 响应相关性评估指标 - 需要同时提供 LLM 和 embeddings
    from llm_utils import qwen_embeddings  # 导入embeddings
    response_relevancy = ResponseRelevancy(llm=qwen3_max, embeddings=qwen_embeddings)
    # 检索内容 上下文准确度评估指标
    # retrieved_context_precision = LLMContextPrecision(llm=qwen3_max)
            
    # 3.评估
    context_precision_score = await response_relevancy.single_turn_ascore(sample)
    score_value = float(context_precision_score)
    
    # 输出评估结果（带阈值对比）
    threshold = 0.75
    if score_value >= threshold:
        logger.info(f"✅ 评估完成 - 分数: {score_value:.3f} (>= 阈值 {threshold}) - 质量合格")
    else:
        logger.info(f"⚠️  评估完成 - 分数: {score_value:.3f} (< 阈值 {threshold}) - 需要人工审核")
    
    return {'evaluate_score': score_value}

# 人工审核节点
def human_approval_node(state: MultidalModalRAGState):
    """
    人工审核节点
    当评估分数低于阈值时，暂停执行并请求人工审核
    
    使用方式：
    1. 首次执行时触发 interrupt，暂停并等待人工审核
    2. 恢复执行时，传入的 Command(resume=decision) 中的 decision 会成为 interrupt 的返回值
    3. 根据审核结果更新状态
    
    注意：
    - interrupt() 必须返回 JSON 可序列化的值
    - 恢复时节点会从头重新执行，interrupt() 前的代码会再次运行（需保证幂等性）
    - 不要在 try/except 中包裹 interrupt() 调用
    """
    # 提取当前响应内容
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    response_content = last_message.content if last_message else "No response available"
    
    # 暂停执行，请求人工审核
    # 这里传递的字典会在调用者端以 __interrupt__ 字段返回
    is_approved = interrupt({
        "question": "是否批准此回答？",     # 告诉调用者：我在问什么
        "score": state.get("evaluate_score"),  # 提供决策依据：评分是多少
        "response": response_content[:50],          # 提供决策依据：回答内容是什么
        "user_input": state.get("input_text"), # 提供上下文：用户问的是什么
        "timestamp": "evaluation_pending"      # 其他元信息
    })
    # 当图恢复执行时，is_approved 会是 Command(resume=xxx) 中传入的值
    # 更新状态中的审核结果
    logger.info(f"人工审核结果: {'批准' if is_approved else '拒绝'}")
    # 更新人工审核结果，后续路由会使用到
    return {
        "human_answer": "approved" if is_approved else "rejected"
    }

# 第四次回复节点
def fourth_chatbot(state: MultidalModalRAGState):
    """
    网络搜索工具绑定的大模型，第四次回复节点
    
    逻辑：
    1. 首次调用：调用 web_search 工具获取信息
    2. 二次调用：基于搜索结果生成最终回答（不再调用工具）
    """
    messages = state.get("messages", [])
    llm_tools = qwen3_vl_plus.bind_tools(web_tools)
    input_text = state.get("input_text")
    
    # 检查是否已经有工具调用结果（ToolMessage）
    has_tool_results = any(msg for msg in messages if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage')
    
    if has_tool_results:
        # 已经搜索过了，生成最终回答（不再调用工具）
        system_message = SystemMessage(content='''你是一个智能助手。上面的 ToolMessage 中已经包含了网络搜索的结果，请基于这些搜索结果直接回答用户的问题。

重要要求：
1. **完全信任搜索结果**：ToolMessage 中的内容是真实可靠的网络搜索结果，请直接使用
2. **不要质疑搜索结果**：不要说"没有公布"、"信息不准确"等，搜索结果已经是最新信息
3. **直接整理呈现**：提取搜索结果中的关键信息，组织成清晰的回答
4. **友好自然**：保持对话风格，直接回答用户问题

不要再调用工具。''')
        # 使用不绑定工具的 LLM，避免再次调用
        return {"messages": [qwen3_vl_plus.invoke(messages + [system_message])]}
    else:
        # 首次调用，需要搜索
        system_message = SystemMessage(content='你是一个智能助手。请调用 web_search 工具搜索用户问题的最新信息。')
        message = HumanMessage(content=[{"type": "text", "text": input_text}])
        return {"messages": [llm_tools.invoke([system_message, message])]}


# 摘要节点
async def summarize_if_needed(state: MultidalModalRAGState):
    """
    条件摘要节点：当消息数量超过阈值时自动生成摘要
    
    功能：
    1. 检查消息数量是否超过阈值（默认25条，适应多Agent复杂工作流）
    2. 如果超过，调用LLM生成或更新摘要
    3. 智能保留策略：保留从最后一个用户消息开始的完整对话轮次
    4. 将摘要信息持久化到state中
    
    摘要策略：
    - 如果已有摘要，则基于旧摘要 + 最近消息生成增量摘要
    - 如果没有摘要，则对所有消息生成新摘要
    
    保留策略：
    - 智能模式：从最后一个 HumanMessage 开始保留所有后续消息（保证完整对话轮次）
    - 降级模式：如果找不到 HumanMessage，保留最近8条消息
    
    Args:
        state: MultidalModalRAGState 状态对象
        
    Returns:
        dict: 包含更新后的 summary、messages（删除指令）和 message_count
    """
    from langchain_core.messages import RemoveMessage
    
    messages = state.get("messages", [])
    current_count = len(messages)
    threshold = 20  # 消息数量阈值（可根据实际情况调整）
    
    logger.info(f"📊 摘要检查 - 当前消息数: {current_count}, 阈值: {threshold}")
    
    # 如果消息数量未超过阈值，跳过摘要生成
    if current_count <= threshold:
        logger.info(f"✅ 消息数量未超过阈值，跳过摘要生成")
        return {"message_count": current_count}
    
    logger.info(f"⚠️ 消息数量超过阈值，开始生成摘要...")
    
    # 获取现有摘要
    existing_summary = state.get("summary", "")
    
    # 构建摘要提示词
    if existing_summary:
        # 已有摘要，生成增量摘要（基于旧摘要 + 最近5条消息） 
        recent_messages = messages[-5:]
        recent_messages_text = "\n".join([
            f"- {msg.__class__.__name__}: {msg.content[:400]}..." 
            if len(str(msg.content)) > 400 else f"- {msg.__class__.__name__}: {msg.content}"
            for msg in recent_messages   # 遍历 recent_messages 中的每一条消息 msg，对它做处理，生成一个字符串，最后组成一个列表。
        ])
        
        summary_prompt = f"""你是一个对话摘要助手。请更新以下对话摘要。

【之前的摘要】
{existing_summary}

【最新的对话（最近5条消息）】
{recent_messages_text}

【要求】
1. 保留之前摘要中的关键信息
2. 整合最新对话的重要内容
3. 保持摘要简洁（不超过500字）
4. 突出用户的问题和系统的关键回答
5. 只返回摘要内容，不要添加任何额外说明

请生成更新后的摘要："""
    else:
        # 首次生成摘要，基于所有消息
        all_messages_text = "\n".join([
            f"- {msg.__class__.__name__}: {msg.content[:500]}..." 
            if len(str(msg.content)) > 500 else f"- {msg.__class__.__name__}: {msg.content}"
            for msg in messages
        ])
        
        summary_prompt = f"""你是一个对话摘要助手。请为以下对话生成简洁的摘要。

【完整对话历史】
{all_messages_text}

【要求】
1. 提取对话的核心主题和关键信息
2. 保留用户的主要问题和系统的关键回答
3. 保持摘要简洁（不超过500字）
4. 只返回摘要内容，不要添加任何额外说明

请生成摘要："""
    
    # 调用LLM生成摘要（使用 qwen3_max 获得更好的摘要质量）
    try:
        summary_message = HumanMessage(content=summary_prompt)
        summary_response = await qwen3_max.ainvoke([summary_message])
        new_summary = summary_response.content
        
        logger.info(f"✅ 摘要生成成功 - 长度: {len(new_summary)} 字符")
        logger.info(f"📝 摘要内容: {new_summary[:100]}..." if len(new_summary) > 100 else f"📝 摘要内容: {new_summary}")
        
    except Exception as e:
        logger.error(f"❌ 摘要生成失败: {e}")
        # 如果摘要生成失败，保留原有状态，不删除消息
        return {"message_count": current_count}
    
    # 🔥 智能保留策略：找到最后一个 HumanMessage 的位置
    # 目标：保留从最后一个用户提问开始的完整对话轮次
    last_human_index = None
    for i in range(len(messages) - 1, -1, -1):  # 从后往前遍历
        if isinstance(messages[i], HumanMessage):
            last_human_index = i
            break  # 找到第一个（从后往前）就停止
    
    if last_human_index is not None and last_human_index > 0:
        # 找到了用户消息，保留从该位置开始的所有消息（完整对话轮次）
        messages_to_keep_count = len(messages) - last_human_index
        messages_to_remove = messages[:last_human_index]
        
        logger.info(f"🎯 找到最后的用户消息位置: 索引 {last_human_index}")
        logger.info(f"📦 保留完整对话轮次: 从索引 {last_human_index} 到 {len(messages)-1}，共 {messages_to_keep_count} 条消息")
    else:
        # 降级方案：如果找不到 HumanMessage（理论上不应该发生），保留最近8条
        messages_to_keep_count = min(8, current_count)
        messages_to_remove = messages[:-messages_to_keep_count] if messages_to_keep_count > 0 else []
        
        logger.info(f"⚠️ 未找到用户消息（或用户消息在首位），使用降级策略保留最近 {messages_to_keep_count} 条")
    
    # 生成删除指令
    remove_message_objects = [
        RemoveMessage(id=msg.id) for msg in messages_to_remove 
        if hasattr(msg, 'id') and msg.id
    ]
    
    
    # 返回更新后的状态
    return {
        "summary": new_summary,                    # 更新摘要
        "messages": remove_message_objects,        # 删除旧消息的指令
        "message_count": messages_to_keep_count    # 更新消息计数
    }



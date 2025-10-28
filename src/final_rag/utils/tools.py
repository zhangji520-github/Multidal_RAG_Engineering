import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils.embeddings_utils import call_dashscope_once
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker, MilvusClient, Function, FunctionType
from env_utils import CONTEXT_COLLECTION_NAME, MILVUS_URI
from llm_utils import qwen3_max

logger = logging.getLogger(__name__)

client=MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet")

@tool('web_search', args_schema=WebSearchInput, description='Search the internet for real-time public information')
def web_search(query: str) -> str:
    """
    Search the internet for real-time information. Use when you need current events or up-to-date facts.

    Args:
        query: The search query

    Returns:
        str: Search results from the internet
    """
    logger.info(f"Web search for: {query}")
    response = qwen3_max.invoke(f"Search web and answer: {query}")
    return response.content
    

@tool('search_context', parse_docstring=True)
def search_context(query: str=None, user_name: str=None) -> str:
    """
    根据用户的输入，从上下文数据库检索查询相关的历史上下文信息，并给出正确的回答

    Args:
        query: 用户的输入
        user_name: 当前的用户名

    Returns:
        str: 查询到的历史上下文信息
    """
    # 1.构造符合 DashScope（通义千问 API）文本嵌入接口 要求的输入格式 DashScope 的 text-embedding API 通常要求输入是 List[Dict]
    input_data = [{'text': query}]
    # 2.向 DashScope 发送请求，获取 query 的 稠密向量嵌入（context_embedding）。
    ok, context_embedding, status, retry_after = call_dashscope_once(input_data)
    filter_expr = None
    if user_name:
        filter_expr = f"user == '{user_name}'"      # 过滤查询指定用户 根据schema中自定义的schema决定

    # 混合检索
    # 每个 AnnSearchRequest 代表针对特定向量字段的基础 ANN 搜索请求 不管是图片还是文本，我们都可以统一转化为dense向量
    dense_search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    dense_req = AnnSearchRequest(
        data = [context_embedding],
        anns_field = "context_dense",
        limit = 5,
        param = dense_search_params,
        expr=filter_expr # 过滤用户名进行检索
    )

    sparse_search_params = {"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}
    sparse_req = AnnSearchRequest(
        data = [query],
        anns_field = "context_sparse",         # collection 中存储稀疏向量（如 SPLADE 或 BM25 生成的）的字段
        limit = 5,
        param = sparse_search_params,
        expr=filter_expr             # 过滤用户名进行检索
    )

    # 在混合搜索中，重排序是一个关键步骤，它整合了来自多个向量搜索的结果，以确保最终输出是最相关和最准确的
    
    # 方案1: RRF Ranker (Reciprocal Rank Fusion) - 自动归一化不同类型的分数
    # ranker = RRFRanker(k=60)  # k值越大，排名靠后的结果影响越小
    
    # 方案2: Weighted Ranker + 归一化 (当前使用)
    # 使用 arctan 归一化确保不同度量标准（COSINE 和 BM25）的分数可比
    ranker = Function(
        name="weighted_ranker",
        input_field_names=[],  # 重排序函数必须为空列表
        function_type=FunctionType.RERANK,
        params={
            "reranker": "weighted",
            "weights": [0.8, 1],  # [dense权重, sparse权重]，dense检索权重更高
            "norm_score": True  # 启用归一化，使用arctan函数将分数归一化到相近范围
        }
    )
    
    logger.info(f"💬 开始检索历史对话上下文... (user={user_name}, query={query[:30]}...)")

    res = client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs = [dense_req, sparse_req],
        ranker = ranker,
        limit = 10,  # 先获取更多候选结果，再通过阈值过滤
        output_fields = ["context_text"],
    )                  # res : [[hit1, hit2, hit3]] 

    # 应用层过滤：只保留分数 >= min_score 的结果
    # 由于启用了 norm_score=True，distance 已归一化到 [0, ~1.57] 范围 (arctan(∞) ≈ π/2)
    # 阈值设为 0.7：考虑到Markdown格式、emoji、空格等因素可能降低相似度
    # 对于历史对话检索，适当放宽阈值可以提高召回率
    filtered_results = [item for item in res[0] if item.distance >= 0.65]
    logger.info(f"✅ 历史对话检索完成：找到 {len(filtered_results)} 条相关记录 (阈值: 0.65, 归一化后)")

    # 处理结果 你想要模型看到什么 context_pieces 就拿 hit 的哪个字段
    context_pieces = []
    for hit in filtered_results:
        context_pieces.append(f"{hit.get('context_text')}")        # 只需要拿到每个hit的context_text字段

    return "\n".join(context_pieces) if context_pieces else "no context found"  # 返回拼接后的上下文信息 作为后续模型回答参考的上下文



# 工具列表
context_tools = [search_context]  # 上下文检索工具
web_tools = [web_search]  # 网络搜索工具
all_tools = [search_context, web_search]  # 所有工具
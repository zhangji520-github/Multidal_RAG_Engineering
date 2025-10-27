import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from pymilvus import MilvusClient

from env_utils import CONTEXT_COLLECTION_NAME, MILVUS_URI
from llm_utils import qwen_embeddings
from utils.log_utils import log

client=MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')
# 全局线程池用于异步操作        更新用户的上下文数据库 如果生成了最终的回答 存入
thread_pool = ThreadPoolExecutor(max_workers=5) # 创建一个线程池

class OptimizedMilvusAsyncWriter:
    def __init__(self,
                 client: MilvusClient,
                 collection_name: str = CONTEXT_COLLECTION_NAME):

        self.client = client
        self.collection_name = collection_name

    def _get_dense_vector(self, text: str):
        """异步生成稠密向量"""
        try:

            dense_vector = qwen_embeddings.embed_query(text)
            return dense_vector

        except Exception as e:
            log.exception(f"向量生成失败: {e}")
            return None


    def _sync_insert(self, data: Dict[str, Any]):
        """同步插入数据到Milvus"""
        try:
            # 插入数据
            result = self.client.insert(collection_name=self.collection_name, data=data)
            log.info(f"[Milvus] 成功插入 {result['insert_count']} 条记录。IDs 示例: {result['ids'][:5]}")

        except Exception as e:
            log.exception(f"插入数据到Milvus失败: {e}")


    async def async_insert(self, context_text: str, user: str, message_type: str = "AIMessage"):
        """异步插入数据"""
        # 准备数据
        dense_vector = self._get_dense_vector(context_text)
        data = {
            "context_text": context_text,
            "user": user,
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "message_type": message_type,
            "context_dense": dense_vector
        }

        # 打印简洁的日志（不包含向量数据，避免终端混乱）
        log.info(f"准备异步插入上下文: user={user}, text_preview={context_text[:50]}..., vector_dim={len(dense_vector) if dense_vector else 0}")
        
        # 使用线程池异步执行插入操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, self._sync_insert, data)


# 全局写入器实例（单例模式）
_milvus_writer_instance = None

def get_milvus_writer() -> OptimizedMilvusAsyncWriter:
    """获取全局Milvus写入器实例（单例）"""
    global _milvus_writer_instance
    if _milvus_writer_instance is None:
        _milvus_writer_instance = OptimizedMilvusAsyncWriter(
            client=client,
            collection_name=CONTEXT_COLLECTION_NAME
        )
    return _milvus_writer_instance
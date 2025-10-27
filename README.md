# Multi-Modal RAG Engineering

基于 LangGraph 的多模态检索增强生成（RAG）系统，集成混合检索、上下文记忆和智能路由功能。

## 项目特性

- 🔍 **混合检索**：结合密集向量（Dense）和稀疏向量（Sparse BM25）检索
- 🎯 **智能路由**：根据查询类型自动选择最佳处理路径
- 💾 **上下文记忆**：支持多用户历史对话检索和管理
- 🖼️ **多模态支持**：处理文本和图片的向量化检索
- 🔄 **人机协同**：关键决策节点支持人工审核
- 🌐 **实时搜索**：集成网络搜索工具获取最新信息
- 📊 **归一化重排序**：使用 WeightedRanker 解决不同度量标准的分数归一化问题

## 系统架构

![工作流程图](static/graph_rag.png)

### 工作流程说明

1. **输入处理（process_input）**
   - 接收用户输入并进行预处理
   - 提取关键信息和上下文

2. **智能摘要（summarize_if_needed）**
   - 对长文本进行自动摘要
   - 优化后续检索效率

3. **路由决策（first_agent_decision）**
   - 分析查询意图
   - 智能路由至以下路径之一：
     - 📚 知识库检索（retrieve_database）
     - 💬 历史对话检索（search_context）
     - 🌐 实时网络搜索（web_search_node）
     - ❌ 直接结束（无需检索）

4. **知识库检索（retrieve_database）**
   - 使用混合检索策略（Dense + Sparse）
   - WeightedRanker 归一化重排序
   - 支持文本和图片多模态查询

5. **上下文检索（search_context）**
   - 检索用户历史对话记录
   - 支持多用户隔离
   - 可选择继续知识库检索或直接生成答案

6. **智能生成**
   - **second_agent_generate**：基于历史上下文生成答案
   - **third_chatbot**：基于知识库内容生成答案
   - **fourth_chatbot**：基于网络搜索结果生成答案

7. **答案评估（evaluate_node）**
   - 自动评估生成答案的质量
   - 决定是否需要人工审核

8. **人工审核（human_approval_node）**
   - 关键决策点的人工介入
   - 支持审批通过或触发网络搜索补充

## 技术栈

- **框架**: LangGraph, LangChain
- **向量数据库**: Milvus (支持混合检索)
- **LLM**: 通义千问 (Qwen)
- **嵌入模型**: DashScope Text Embedding
- **OCR**: DoTS OCR (文档解析)
- **语言**: Python 3.x

## 项目结构

```
Multidal_RAG_Engineering/
├── src/
│   └── final_rag/
│       ├── workflow.py          # 主工作流定义
│       └── utils/
│           ├── nodes.py         # 工作流节点实现
│           ├── routers.py       # 路由决策逻辑
│           ├── tools.py         # 工具函数（检索、搜索）
│           ├── state.py         # 状态管理
│           └── prompt.py        # 提示词模板
├── milvus_db/
│   ├── milvus_db_with_schema.py # Milvus 数据库操作
│   ├── milvus_retrieve.py       # 混合检索实现
│   └── collections_operator.py  # 集合管理
├── dots_ocr/                    # OCR 文档解析模块
├── splitters/                   # 文档分割器
├── utils/
│   ├── embeddings_utils.py      # 向量化工具
│   ├── common_utils.py          # 通用工具函数
│   └── log_utils.py             # 日志工具
├── tests/                       # 测试文件
├── static/
│   └── graph_rag.png            # 工作流程图
├── env_utils.py                 # 环境配置
├── llm_utils.py                 # LLM 调用封装
└── .gitignore
```

## 核心功能

### 1. 混合检索与归一化重排序

```python
# 使用 WeightedRanker + 归一化解决 COSINE 和 BM25 分数不可比问题
ranker = Function(
    name="weighted_ranker",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted",
        "weights": [1.0, 0.5],  # dense 和 sparse 权重
        "norm_score": True      # 启用 arctan 归一化
    }
)

# 混合检索
res = client.hybrid_search(
    collection_name=COLLECTION_NAME,
    reqs=[dense_req, sparse_req],
    ranker=ranker,
    limit=10
)

# 应用阈值过滤（归一化后分数范围 [0, ~1.57]）
filtered_results = [item for item in res[0] if item.distance >= 0.9]
```

### 2. 智能路由决策

系统根据查询内容自动选择处理路径：
- 专业知识 → 知识库检索
- 个人问题 → 历史对话检索
- 实时信息 → 网络搜索
- 简单问候 → 直接响应

### 3. 多用户上下文管理

```python
# 支持用户级别的历史对话隔离
filter_expr = f"user == '{user_name}'"
res = client.hybrid_search(
    collection_name=CONTEXT_COLLECTION_NAME,
    reqs=[dense_req, sparse_req],
    ranker=ranker,
    limit=5,
    output_fields=["context_text"],
)
```

## 快速开始

### 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量（`.env` 文件）：
```env
DASHSCOPE_API_KEY=your_api_key
MILVUS_URI=your_milvus_uri
COLLECTION_NAME=your_collection
CONTEXT_COLLECTION_NAME=your_context_collection
```

3. 初始化 Milvus 数据库：
```bash
python milvus_db/milvus_db_with_schema.py
```

### 运行示例

```bash
# 运行主工作流
python src/final_rag/workflow.py

# 运行测试
python tests/test_workflow_interactive.py
```

## 配置说明

### 混合检索参数

- **Dense 检索**: COSINE 相似度，适合语义匹配
- **Sparse 检索**: BM25 算法，适合关键词匹配
- **权重比例**: dense=1.0, sparse=0.5（可调整）
- **归一化阈值**: 0.9（归一化后，范围 [0, 1.57]）

### Ranker 选择

支持两种重排序策略（可在 `tools.py` 中切换）：

**方案1: RRF Ranker**（自动归一化）
```python
ranker = RRFRanker(k=60)
```

**方案2: Weighted Ranker**（当前使用，手动归一化）
```python
ranker = Function(..., params={"norm_score": True})
```

## 性能优化

1. **向量维度**: 1536 (DashScope)
2. **检索数量**: 初始 10 条，阈值过滤后通常 3-5 条
3. **归一化方法**: arctan 函数，解决不同度量标准的分数归一化
4. **缓存策略**: 支持历史对话缓存，减少重复检索

## 开发计划

- [ ] 支持更多 LLM 模型
- [ ] 添加流式输出功能
- [ ] 优化多轮对话管理
- [ ] 增强图片理解能力
- [ ] 支持更多向量数据库

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- 作者: zhangji520-github
- 邮箱: 2944405449@shu.edu.cn

---

⭐ 如果这个项目对你有帮助，请给个 Star！


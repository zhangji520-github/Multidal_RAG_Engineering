# Multi-Modal RAG Engineering

基于 LangGraph 的多模态检索增强生成（RAG）系统，集成混合检索、分层记忆架构、智能路由和人机协同审批机制，提供企业级 FastAPI 后端服务。

## 🌟 项目特性

### 核心功能
- 🔍 **混合检索**：结合密集向量（Dense）和稀疏向量（Sparse BM25）检索，使用 WeightedRanker 归一化重排序
- 🎯 **智能路由**：根据查询类型自动选择最佳处理路径（知识库/历史对话/网络搜索）
- 🖼️ **多模态支持**：处理文本和图片的向量化检索，支持 OCR 文档解析
- 🌐 **实时搜索**：集成网络搜索工具获取最新信息
- 📊 **质量评估**：基于 RAGAS 框架自动评估生成答案质量

### 记忆系统（核心亮点）
- 🧠 **三层记忆架构**：
  - **会话级记忆**：PostgreSQL + LangGraph Checkpointer（状态持久化、中断恢复）
  - **用户级长期记忆**：Milvus 上下文库（高质量对话语义检索）
  - **知识级记忆**：Milvus 知识库（文档和图片多模态检索）
- 🔒 **多用户隔离**：基于 user_name 的记忆隔离和个性化服务
- ✅ **智能筛选**：根据 RAGAS 评分和人工审批自动过滤低质量对话

### 人机协同
- 🔄 **两阶段审批机制**：质量评分低时触发人工审核，支持批准/拒绝决策
- 🛑 **工作流中断**：基于 LangGraph interrupt() 实现的可恢复工作流
- 👤 **人工介入**：关键决策节点支持人工审核和补充信息

### 企业级后端
- 🚀 **FastAPI 服务**：RESTful API、OpenAPI 文档、异步处理
- 🔐 **用户认证**：JWT Token + OAuth2 + PassLib 密码加密
- 🗄️ **数据库管理**：PostgreSQL + SQLAlchemy ORM + DAO 模式
- ⚙️ **配置管理**：Dynaconf 多环境配置（开发/生产）
- 📝 **日志系统**：结构化日志记录和监控

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

## 🛠️ 技术栈

### AI 框架
- **LangGraph**：多智能体工作流编排、状态持久化、中断恢复
- **LangChain**：LLM 工具链、提示词模板、消息管理

### 后端框架
- **FastAPI**：现代 Python Web 框架、自动 API 文档生成、异步支持
- **Pydantic**：数据验证和序列化、类型安全

### 数据库
- **PostgreSQL**：
  - 用户管理数据库（user_db）
  - LangGraph 状态持久化数据库（multidal_modal_rag）
- **Milvus**：向量数据库、混合检索（Dense + Sparse）、多模态支持
  - 知识库集合（multimodal_rag）
  - 上下文库集合（multimodal_rag_context）

### ORM 与数据访问
- **SQLAlchemy**：Python ORM 框架、数据库迁移管理
- **DAO 模式**：数据访问对象模式、业务逻辑分离

### 认证与安全
- **JWT (JSON Web Token)**：无状态 Token 认证
- **OAuth2**：授权框架、Swagger UI 集成
- **PassLib + Bcrypt**：密码哈希和验证

### 配置管理
- **Dynaconf**：多环境配置管理（development/production）、环境变量加载

### LLM 与嵌入
- **通义千问 (Qwen)**：大语言模型
- **DashScope Text Embedding**：文本向量化（1536 维）

### 文档处理
- **DoTS OCR**：文档光学字符识别和解析
- **自定义分割器**：文档分块和预处理

### 评估框架
- **RAGAS**：RAG 答案质量自动评估

### 开发工具
- **Python 3.10+**
- **Uvicorn**：ASGI 服务器
- **日志系统**：结构化日志记录

## 📁 项目结构

```
Multidal_RAG_Engineering/
├── src/                         # 主源码目录
│   ├── main.py                  # FastAPI 应用入口
│   ├── config/                  # 配置管理
│   │   ├── __init__.py         # Dynaconf 初始化
│   │   ├── log_config.py       # 日志配置
│   │   ├── development.yml     # 开发环境配置
│   │   └── production.yml      # 生产环境配置
│   ├── api/                     # API 路由
│   │   ├── routers.py          # 路由注册
│   │   ├── system_mgt/         # 用户管理模块
│   │   │   ├── user_views.py  # 用户接口
│   │   │   └── user_schemas.py # 用户数据模型
│   │   └── graph_api/          # RAG 工作流模块
│   │       ├── graph_view.py   # RAG 接口（聊天/审批）
│   │       └── graph_schema.py # RAG 数据模型
│   ├── api_utils/               # API 工具
│   │   ├── handler_error.py    # 错误处理
│   │   ├── cors.py             # CORS 配置
│   │   ├── middlewares.py      # 中间件
│   │   └── docs_oauth2.py      # OAuth2 认证
│   ├── db/                      # 数据库层
│   │   ├── __init__.py         # SQLAlchemy 初始化
│   │   ├── base_dao.py         # 基础 DAO 类
│   │   └── system_mgt/
│   │       ├── models.py       # 用户模型
│   │       └── user_dao.py     # 用户 DAO
│   └── final_rag/               # RAG 工作流
│       ├── workflow.py          # CLI 版本（测试用）
│       ├── workflow_fastapi.py # FastAPI 版本
│       └── utils/
│           ├── nodes.py         # 工作流节点
│           ├── routers.py       # 智能路由
│           ├── tools.py         # 检索工具
│           ├── state.py         # 状态管理
│           └── prompt.py        # 提示词模板
├── milvus_db/                   # Milvus 向量数据库
│   ├── milvus_db_with_schema.py # 数据库操作
│   ├── milvus_retrieve.py       # 混合检索
│   └── collections_operator.py  # 集合管理
├── dots_ocr/                    # OCR 文档解析
│   ├── inference.py             # OCR 推理
│   ├── parser.py                # 结果解析
│   └── utils/                   # OCR 工具
├── splitters/                   # 文档分割器
│   └── splitter_md.py           # Markdown 分割
├── utils/                       # 通用工具
│   ├── embeddings_utils.py      # 向量化工具
│   ├── common_utils.py          # 通用函数
│   └── log_utils.py             # 日志工具
├── tests/                       # 测试文件
│   ├── test_workflow_interactive.py # 工作流测试
│   └── test_human_approval.py  # 审批机制测试
├── static/                      # 静态文件
│   ├── graph_rag.png            # 工作流程图
│   └── graph_rag.mmd            # Mermaid 图源码
├── init_db.py                   # 数据库表初始化脚本
├── view_users.py                # 查看用户数据脚本
├── env_utils.py                 # 环境配置
├── llm_utils.py                 # LLM 调用封装
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明
```

## 💡 核心功能详解

### 1. 三层记忆架构（项目亮点）

#### 设计背景
传统 RAG 系统的对话历史管理面临两大挑战：
1. **工作流状态持久化**：多步骤 AI 工作流需要支持中断和恢复
2. **语义检索局限**：关系型数据库的 SQL 查询无法理解语义相似性（如"容错控制" ≈ "fault-tolerant system"）

#### 解决方案
采用"向量数据库 + 关系型数据库"的混合架构，实现三层记忆分离：

**🔹 第一层：会话级记忆（PostgreSQL + LangGraph Checkpointer）**
- 存储工作流状态（消息历史、节点状态、工具调用结果）
- 使用 JSONB 格式存储复杂状态数据
- 通过 thread_id 实现会话隔离
- 支持工作流中断后的精确恢复

```python
# 配置 LangGraph Checkpointer
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    graph = build_graph(checkpointer, store)
    config = {"configurable": {"thread_id": session_id}}
```

**🔹 第二层：用户级长期记忆（Milvus 上下文库）**
- 存储高质量对话（RAGAS 评分 ≥ 0.75 或人工审批通过）
- 向量化后支持语义搜索（理解同义词、多语言）
- 基于 user_name 的多用户隔离
- 自动过滤低质量对话，避免知识库污染

```python
# 智能筛选：只保存优质对话
if (evaluate_score >= 0.75) or (human_answer == 'approved'):
    save_context_to_milvus(
        user_name=user_name,
        context_text=f"Q: {user_input}\nA: {final_answer}",
        session_id=session_id
    )
```

**🔹 第三层：知识级记忆（Milvus 知识库）**
- 存储文档和图片的向量表示
- 支持多模态检索（文本 + 图片）
- 混合检索策略（Dense + Sparse）

#### 技术亮点
✅ **语义 vs SQL**：向量检索可以匹配"多智能体系统" ≈ "multi-agent architecture"，而 SQL LIKE 无法实现  
✅ **分层存储**：短期状态用 PostgreSQL（完整记录），长期记忆用 Milvus（精选内容），降低向量库成本  
✅ **多用户隔离**：每个用户的历史对话独立存储和检索，确保隐私和个性化  
✅ **智能过滤**：基于 RAGAS 评分和人工审批自动筛选，保证知识库质量

---

### 2. 混合检索与归一化重排序

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

---

### 3. 两阶段审批机制

#### 设计挑战
FastAPI 的无状态 HTTP 协议与 LangGraph 的有状态工作流存在矛盾：
- **问题**：传统方式在工作流中 `input()` 会阻塞整个 HTTP 线程
- **需求**：需要将"等待审批"和"提交决策"拆分为两个独立的 API 调用

#### 解决方案

**第一阶段：工作流执行与中断**
```python
@router.post('/chat', response_model=Union[ChatResponse, InterruptResponse])
async def chat(request: ChatRequest):
    result = await execute_graph_for_api(
        user_input=request.text,
        session_id=request.session_id,
        user_name=request.user_name
    )
    
    # 根据评分决定是否中断
    if result['status'] == 'interrupted':
        return InterruptResponse(
            status='interrupted',
            session_id=result['session_id'],
            question='是否批准此回答？',
            evaluate_score=result['evaluate_score'],
            current_answer=result['current_answer']
        )
    
    return ChatResponse(status='completed', answer=result['answer'])
```

**第二阶段：人工决策与恢复**
```python
@router.post('/approval', response_model=ChatResponse)
async def approval(request: ApprovalRequest):
    # 用户提交审批决策（approve/reject）
    result = await resume_graph_for_api(
        session_id=request.session_id,
        decision=(request.decision == "approve")
    )
    
    return ChatResponse(
        status='completed',
        answer=result['answer'],
        human_answer=result['human_answer']
    )
```

#### 工作流程
1. 前端调用 `POST /api/graph/chat`，系统执行 RAG 工作流
2. 如果 RAGAS 评分 < 0.75，工作流中断并返回 `InterruptResponse`
3. 前端展示待审批内容，用户做出决策（批准/拒绝）
4. 前端调用 `POST /api/graph/approval`，传入 `session_id` 和 `decision`
5. 系统通过 `Command(resume=decision)` 恢复工作流并返回最终结果

#### 技术亮点
✅ **状态持久化**：LangGraph Checkpointer 保存中断状态，支持跨请求恢复  
✅ **灵活决策**：批准直接返回答案，拒绝则触发网络搜索补充信息  
✅ **前后端分离**：符合 RESTful 设计，前端可自由实现审批 UI

---

### 4. 智能路由决策

系统根据查询内容自动选择处理路径：
- 📚 专业知识 → 知识库检索（retrieve_database）
- 💬 个人问题 → 历史对话检索（search_context）
- 🌐 实时信息 → 网络搜索（web_search_node）
- 👋 简单问候 → 直接响应（END）

```python
# 使用 LLM 进行智能路由
router_result = llm_with_router.invoke(messages)
if router_result.name == "ToDatabase":
    return "retrieve_database"
elif router_result.name == "ToSearchContext":
    return "search_context"
# ...
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- PostgreSQL 12+
- Milvus 2.3+
- 通义千问 API Key

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/zhangji520-github/Multidal_RAG_Engineering.git
cd Multidal_RAG_Engineering

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 PostgreSQL 驱动
pip install psycopg2-binary
```

### 3. 配置文件

**开发环境配置：`src/config/development.yml`**

```yaml
# 服务配置
LOG_LEVEL: INFO
HOST: 127.0.0.1
PORT: 8001
ORIGINS: ['http://localhost:8080', 'http://127.0.0.1:8080']

# PostgreSQL 配置
POSTGRES:
  USER_DB:  # 用户管理数据库
    DRIVER: postgresql
    NAME: user_db
    HOST: localhost
    PORT: 5432
    USERNAME: postgres
    PASSWORD: your_password
  LANGGRAPH_DB:  # LangGraph 状态数据库
    DRIVER: postgresql
    NAME: multidal_modal_rag
    HOST: localhost
    PORT: 5432
    USERNAME: postgres
    PASSWORD: your_password
    URI: postgresql://postgres:your_password@localhost:5432/multidal_modal_rag

# Milvus 配置
MILVUS:
  URI: http://localhost:19530
  USERNAME: root
  PASSWORD: Milvus
  COLLECTIONS:
    KNOWLEDGE: multimodal_rag
    CONTEXT: multimodal_rag_context

# JWT 认证配置
JWT_SECRET_KEY: your_secret_key_here
ALGORITHM: HS256
ACCESS_TOKEN_EXPIRE_MINUTES: 30

# 白名单路由（无需认证）
WHITE_LIST: ['/api/login', '/api/register', '/docs', '/openapi']

# 默认密码
DEFAULT_PASSWORD: 123123
```

**环境变量：`.env` 文件**

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
EMP_ENV=development  # 或 production
```

### 4. 初始化数据库

```bash
# 1. 创建 PostgreSQL 数据库
createdb user_db
createdb multidal_modal_rag

# 2. 创建数据库表
python init_db.py

# 3. 初始化 Milvus 集合
python milvus_db/milvus_db_with_schema.py
```

### 5. 启动服务

```bash
# 方式1：直接启动（推荐用于开发）
python -m src.main

# 方式2：使用 Uvicorn（推荐用于生产）
uvicorn src.main:Server().app --host 0.0.0.0 --port 8001 --reload

# 方式3：切换到生产环境
export EMP_ENV=production
python -m src.main
```

### 6. 访问 API 文档

启动后访问：
- Swagger UI: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc

### 7. 测试 API

**注册用户**
```bash
curl -X POST "http://127.0.0.1:8001/api/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "zhangji",
    "password": "123456",
    "phone": "13800138000",
    "email": "zhangji@example.com"
  }'
```

**登录获取 Token**
```bash
curl -X POST "http://127.0.0.1:8001/api/auth/" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=zhangji&password=123456"
```

**调用 RAG 接口**
```bash
curl -X POST "http://127.0.0.1:8001/api/graph/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "什么是多智能体系统？",
    "session_id": "zhangji_test_001",
    "user_name": "zhangji"
  }'
```

**处理审批（如果触发中断）**
```bash
curl -X POST "http://127.0.0.1:8001/api/graph/approval" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "zhangji_test_001",
    "decision": "approve"
  }'
```

### 8. CLI 测试模式

```bash
# 测试 RAG 工作流（无需启动 FastAPI）
python src/final_rag/workflow.py

# 测试人工审批机制
python tests/test_human_approval.py
```

## ⚙️ 配置说明

### 混合检索参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **Dense 检索** | COSINE 相似度 | 适合语义匹配（同义词、多语言） |
| **Sparse 检索** | BM25 算法 | 适合关键词精确匹配 |
| **权重比例** | dense=1.0, sparse=0.5 | 可在 `tools.py` 中调整 |
| **归一化阈值** | 0.9 | 归一化后分数范围 [0, 1.57] |
| **向量维度** | 1536 | DashScope Text Embedding |
| **检索数量** | 初始 10 条 | 阈值过滤后通常 3-5 条 |

### 环境切换

通过环境变量 `EMP_ENV` 切换配置文件：

```bash
# 开发环境（默认）
export EMP_ENV=development

# 生产环境
export EMP_ENV=production
```

Dynaconf 会自动加载对应的 `development.yml` 或 `production.yml`。

### JWT 配置

- **JWT_SECRET_KEY**: 用于签名 Token，建议使用 `secrets.token_hex(32)` 生成
- **ALGORITHM**: 默认 HS256
- **ACCESS_TOKEN_EXPIRE_MINUTES**: Token 过期时间（分钟）

### 白名单路由

`WHITE_LIST` 中的路由无需 JWT 认证：
- `/api/login`, `/api/register`: 登录注册接口
- `/docs`, `/swagger`, `/openapi`: API 文档
- `/static`: 静态资源

## 📊 性能优化

1. **向量检索优化**
   - 使用 Milvus IVF_FLAT 索引，查询延迟 < 100ms
   - 混合检索并行执行，总耗时约为单一检索的 1.2 倍

2. **数据库连接池**
   - SQLAlchemy pool_size=10，避免频繁建立连接
   - 异步 PostgreSQL Saver，支持高并发

3. **缓存策略**
   - LangGraph Checkpointer 自动缓存工作流状态
   - 历史对话缓存减少重复向量化

4. **归一化方法**
   - 使用 arctan 函数解决 COSINE 和 BM25 分数不可比问题
   - 自动将异构分数映射到统一区间 [0, 1.57]

## 🔍 API 接口文档

### 用户管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/register` | POST | 用户注册 |
| `/api/login` | POST | 用户登录（返回用户信息） |
| `/api/auth/` | POST | OAuth2 登录（返回 JWT Token） |
| `/api/users` | GET | 获取用户列表（需认证） |
| `/api/users/{id}` | GET | 获取用户详情（需认证） |

### RAG 工作流

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/graph/chat` | POST | 发起对话（可能返回中断状态） |
| `/api/graph/approval` | POST | 提交审批决策（恢复中断工作流） |

**请求示例（/api/graph/chat）：**
```json
{
  "text": "什么是多智能体系统？",
  "image_path": null,
  "session_id": "zhangji_项目讨论",
  "user_name": "zhangji"
}
```

**正常响应（ChatResponse）：**
```json
{
  "status": "completed",
  "session_id": "zhangji_项目讨论",
  "answer": "多智能体系统是指...",
  "human_answer": null,
  "error": null,
  "evaluate_score": 0.85
}
```

**中断响应（InterruptResponse）：**
```json
{
  "status": "interrupted",
  "session_id": "zhangji_项目讨论",
  "question": "是否批准此回答？",
  "user_input": "什么是容错控制？",
  "evaluate_score": 0.65,
  "current_answer": "容错控制是一种..."
}
```

## 🛠️ 开发计划

- [ ] 支持流式输出（Server-Sent Events）
- [ ] 前端 Vue 3 管理界面
- [ ] 支持更多 LLM 模型（DeepSeek、GPT-4）
- [ ] 多轮对话上下文窗口优化
- [ ] 增强图片理解能力（支持图生图）
- [ ] 支持 Pinecone、Qdrant 等更多向量数据库
- [ ] 添加 Prometheus 监控指标
- [ ] Docker Compose 一键部署

## 📖 项目文档

- [工作流程图](static/graph_rag.mmd)
- [Mermaid 源码](static/graph_rag.mmd)
- 更多文档请查看 `Resume/` 目录

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

**开发流程：**
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- **作者**: zhangji520-github
- **邮箱**: 2944405449@shu.edu.cn
- **GitHub**: [https://github.com/zhangji520-github](https://github.com/zhangji520-github)

---

## 🌟 致谢

感谢以下开源项目：
- [LangGraph](https://github.com/langchain-ai/langgraph) - AI 工作流编排
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [Milvus](https://milvus.io/) - 向量数据库
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG 评估框架

---

⭐ **如果这个项目对你有帮助，请给个 Star！** ⭐


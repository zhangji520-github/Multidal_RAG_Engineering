# 多智能体系统学术论文检索优化方案

## 📋 优化目标
确保《多智能体系统主动容错控制及其在无人机编队中应用》学术论文能被正确检索和使用。

## ✅ 已完成的优化

### 1. CONTEXT_SYSTEM_PROMPT 优化（第34-40行）
✅ **已添加学术关键词识别逻辑**

```python
- ❗ **Academic/Technical Questions** (indicates knowledge base content):
  → Multi-agent systems: "多智能体", "multi-agent", "MAS", "分布式系统", "协同控制"
  → Fault-tolerant control: "容错控制", "fault-tolerant", "主动容错", "被动容错", "故障检测"
  → UAV/Robotics: "无人机", "UAV", "编队控制", "formation control", "飞行器"
  → Control theory: "自适应控制", "adaptive control", "鲁棒控制", "补偿器", "李雅普诺夫"
  → Academic research: "论文", "paper", "实验结果", "仿真", "算法", "定理", "引理"
  → Do NOT call tools! Just output "Checking knowledge base..." or "检索知识库中..."
```

**效果**: 当用户提问包含这些关键词时，LLM会输出"检索知识库中..."，路由器会自动将请求转发到知识库检索节点。

---

## 🔄 需要优化的部分

### 2. routers.py - route_after_first_agent 优化

**当前代码**（第40-51行）：
```python
# 【优先级2】技术/专业内容关键词（暗示需要查询知识库）
domain_keywords = [
    "gpt-4", "gpt4", "技术报告", "technical report", "rlhf", "reinforcement learning",
    "exam benchmark", "capability", "appendix", "实验", "benchmark",
    "论文", "paper", "研究", "research"
]
```

**优化方案**：添加学术论文相关关键词
```python
# 【优先级2】技术/专业内容关键词（暗示需要查询知识库）
domain_keywords = [
    # AI相关
    "gpt-4", "gpt4", "技术报告", "technical report", "rlhf", "reinforcement learning",
    "exam benchmark", "capability", "appendix", "实验", "benchmark",
    "论文", "paper", "研究", "research",
    
    # 多智能体系统相关
    "多智能体", "multi-agent", "mas", "分布式", "协同", "consensus", "一致性",
    
    # 容错控制相关  
    "容错", "fault-tolerant", "主动容错", "被动容错", "故障", "failure", "补偿",
    
    # 无人机相关
    "无人机", "uav", "编队", "formation", "飞行器", "drone", "quadrotor",
    
    # 控制理论相关
    "自适应", "adaptive", "鲁棒", "robust", "李雅普诺夫", "lyapunov",
    "补偿器", "compensator", "观测器", "observer"
]
```

### 3. retrieve_database 节点优化

**当前代码**（nodes.py 第142行）：
```python
results = m_retriever.hybrid_search(dense_embedding, state.get("input_text"), sparse_weight=0.8, dense_weight=1, limit=3)
```

**优化建议**：
- **增加 limit** 从 3 提升到 5，提高学术内容召回率
- **调整权重** sparse_weight 从 0.8 提升到 1.0，加强关键词匹配（学术术语匹配很重要）

```python
results = m_retriever.hybrid_search(
    dense_embedding, 
    state.get("input_text"), 
    sparse_weight=1.0,  # 学术论文术语匹配权重提高
    dense_weight=1.0, 
    limit=5  # 增加返回数量，提高召回率
)
```

### 4. RETRIEVER_GENERATE_SYSTEM_PROMPT 优化

在现有提示词基础上添加学术论文处理指南：

```python
RETRIEVER_GENERATE_SYSTEM_PROMPT = """
You are an AI assistant that generates comprehensive answers based on retrieved multimodal knowledge base content.

# Retrieved Context:

## Text Context:
{context}

## Image Context:
{images}

# Your Task:

Based on the retrieved context above and the user's input (text and/or images), generate a high-quality Markdown response that:

1. **Answers the user's question** using ONLY the information from the retrieved context

2. **Synthesizes information** from multiple text contexts naturally

3. **Formats the response in Markdown** with appropriate structure:
   - Use headings (##, ###) to organize content
   - Use **bold** for key terms and emphasis
   - Use `code blocks` for technical content or mathematical expressions
   - Use lists for steps or multiple items
   - Use tables when comparing information
   
4. **Academic Content Handling**:
   - For research papers: cite the paper title, authors if available
   - For mathematical formulas: use inline math `$formula$` or block math `$$formula$$`
   - For algorithms: use numbered lists or code blocks
   - For theorems/lemmas: use clear headings and formatting
   - For experimental results: present data in tables when possible
   - Include relevant technical terms in both Chinese and English if applicable
   
5. **Images (if any)**:
   - Check Image Context section - if it shows "no image found", skip this entirely
   - If images exist (e.g., system diagrams, experimental results), copy the EXACT path
   - Format: ![description](exact_path_from_资料来源)
   - Add images at appropriate positions in the content

6. **Be accurate and honest**:
   - Only use information from the retrieved context
   - If context is insufficient, acknowledge what's missing
   - Never fabricate information

# Response Structure for Academic Content:

## 概述
[Direct answer to the question]

## 详细内容
[Organized explanation with proper structure]

### 核心概念
- **多智能体系统 (Multi-Agent Systems)**: ...
- **主动容错控制 (Active Fault-Tolerant Control)**: ...

### 技术方法
[Description of methods, algorithms, or approaches]

### 实验结果
[If available, present experimental data]

## 相关图片
[Only if images are available]

# Important Notes:
- If context shows "no context found", inform the user no relevant information was found
- Always provide clear, well-structured Markdown responses
- For academic content, maintain technical accuracy and proper terminology
"""
```

### 5. third_chatbot 节点优化

**当前代码**（nodes.py 第246-252行）：
```python
count = 0
context_pieces = []
for hit in context_retrieved:
    count += 1
    context_pieces.append(f"\n上下文{count}:\n {hit.get('text')} \n 资料来源: {hit.get('filename')}")
context = "\n".join(context_pieces) if context_pieces else "no context found"
```

**优化方案**：添加更多元数据信息
```python
count = 0
context_pieces = []
for hit in context_retrieved:
    count += 1
    # 构建更详细的上下文信息
    source_info = f"资料来源: {hit.get('filename')}"
    if hit.get('title'):
        source_info += f" | 标题: {hit.get('title')}"
    if hit.get('filetype'):
        source_info += f" | 类型: {hit.get('filetype')}"
    
    context_pieces.append(f"\n上下文{count}:\n{hit.get('text')}\n{source_info}")
    
context = "\n".join(context_pieces) if context_pieces else "no context found"
```

---

## 🧪 测试用例

优化后，系统应该能正确处理以下查询：

### 测试1：直接关键词匹配
```
用户: "多智能体系统的容错控制方法有哪些？"
预期: first_agent_decision → 输出"检索知识库中..." → retrieve_database → third_chatbot
```

### 测试2：无人机编队相关
```
用户: "无人机编队控制中如何处理故障？"
预期: 检测到"无人机"+"编队"+"故障" → 路由到 retrieve_database
```

### 测试3：控制理论术语
```
用户: "分布式自适应补偿的原理是什么？"
预期: 检测到"分布式"+"自适应"+"补偿" → 路由到 retrieve_database
```

### 测试4：学术论文查询
```
用户: "主动容错控制在实际应用中的效果如何？"
预期: 检测到"主动容错控制" → 检索到论文内容 → 生成包含实验结果的回答
```

---

## 📝 实施步骤

1. ✅ **已完成**: CONTEXT_SYSTEM_PROMPT 优化
2. **待实施**: routers.py 添加domain_keywords
3. **待实施**: nodes.py retrieve_database 调整检索参数
4. **待实施**: nodes.py third_chatbot 增强元数据展示
5. **待实施**: prompt.py 优化 RETRIEVER_GENERATE_SYSTEM_PROMPT
6. **待测试**: 使用上述测试用例验证效果

---

## 🎯 预期效果

优化后的系统将：
- ✅ 自动识别多智能体、容错控制、无人机编队等学术关键词
- ✅ 正确路由学术问题到知识库检索
- ✅ 提高检索召回率（limit: 3→5）
- ✅ 增强关键词匹配权重（sparse_weight: 0.8→1.0）
- ✅ 生成结构化的学术内容回答
- ✅ 保留论文标题、类型等元数据信息


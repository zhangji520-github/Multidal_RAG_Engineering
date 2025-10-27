# 工作流程详细说明

## 完整流程图

```
START
  ↓
process_input (处理输入)
  ↓
summarize_if_needed (智能摘要)
  ↓
first_agent_decision (路由决策)
  ├─→ search_context (检索历史对话)
  │     ├─→ second_agent_generate (基于历史生成答案)
  │     └─→ retrieve_database (未找到历史，检索知识库)
  ├─→ retrieve_database (检索知识库)
  ├─→ web_search_node (网络搜索)
  └─→ END (简单问题直接结束)
  
retrieve_database → third_chatbot (基于知识库生成答案)

所有生成答案的路径 → evaluate_node (评估答案质量)
  ├─→ 分数 >= 0.75 → END ✅
  └─→ 分数 < 0.75 → human_approval_node (人工审核)

human_approval_node
  ├─→ approved (批准) → END ✅
  └─→ rejected (拒绝)
        ├─→ 首次拒绝 → fourth_chatbot (网络搜索备选)
        │     ↓
        │   web_search_node (执行搜索)
        │     ↓
        │   fourth_chatbot (生成答案)
        │     ↓
        │   evaluate_node (再次评估) ⚠️ 新增
        │     ↓
        │   human_approval_node (二次审核) ⚠️ 新增
        │     ├─→ approved → END ✅
        │     └─→ rejected → END ⚠️ (避免无限循环)
        │
        └─→ 二次拒绝 → END ⚠️ (已使用网络搜索，不再重试)
```

## 关键修改点

### 1. ⚠️ 网络搜索结果也需要评估和审核

**修改前：**
```
fourth_chatbot → web_search_node → fourth_chatbot → END ❌
```
直接结束，没有质量检查。

**修改后：**
```
fourth_chatbot → web_search_node → fourth_chatbot → evaluate_node → human_approval_node
```
网络搜索的答案也要经过评估和人工审核。

### 2. ⚠️ 防止无限循环

**修改前：**
```
human_approval_node (rejected) → fourth_chatbot → ... → human_approval_node (rejected) → fourth_chatbot → ∞
```
可能无限循环。

**修改后：**
```python
def route_after_human_approval(state):
    if approved:
        return END
    else:
        # 检查是否已使用过网络搜索
        has_web_search = any(msg.name == 'web_search' for msg in messages)
        
        if has_web_search:
            return END  # 已用过网络搜索，不再重试
        else:
            return "fourth_chatbot"  # 首次拒绝，尝试网络搜索
```

最多只会尝试一次网络搜索。

## 审核流程详解

### 场景1: 知识库答案被拒绝

```
1. retrieve_database → third_chatbot (生成答案A)
2. evaluate_node → 分数 0.3 < 0.75
3. human_approval_node → 用户输入: reject
4. route_after_human_approval → 检查：无网络搜索记录 → fourth_chatbot
5. fourth_chatbot → 调用 web_search 工具
6. web_search_node → 返回搜索结果
7. fourth_chatbot → 生成答案B（基于搜索结果）
8. evaluate_node → 分数 0.8 >= 0.75 → END ✅
```

### 场景2: 网络搜索答案也被拒绝

```
1. retrieve_database → third_chatbot (生成答案A)
2. evaluate_node → 分数 0.3 < 0.75
3. human_approval_node → 用户输入: reject
4. fourth_chatbot → web_search_node → fourth_chatbot (生成答案B)
5. evaluate_node → 分数 0.4 < 0.75
6. human_approval_node → 用户输入: reject
7. route_after_human_approval → 检查：已有网络搜索记录 → END ⚠️
   （系统提示：已尝试网络搜索，无法提供更好答案）
```

### 场景3: 网络搜索答案质量高

```
1. retrieve_database → third_chatbot (生成答案A)
2. evaluate_node → 分数 0.3 < 0.75
3. human_approval_node → 用户输入: reject
4. fourth_chatbot → web_search_node → fourth_chatbot (生成答案B)
5. evaluate_node → 分数 0.9 >= 0.75 → END ✅
   （无需人工审核，直接通过）
```

## 代码修改总结

### 文件1: `src/final_rag/workflow.py`

```python
# 修改前
builder.add_conditional_edges(
    "fourth_chatbot", 
    tools_condition,
    {
        "tools": "web_search_node",
        '__end__': END  # ❌ 直接结束
    }
)

# 修改后
builder.add_conditional_edges(
    "fourth_chatbot", 
    tools_condition,
    {
        "tools": "web_search_node",
        '__end__': "evaluate_node"  # ✅ 进入评估
    }
)
```

### 文件2: `src/final_rag/utils/routers.py`

```python
# 修改前
def route_after_human_approval(state):
    if human_answer == "approved":
        return END
    else:
        return "fourth_chatbot"  # ❌ 可能无限循环

# 修改后
def route_after_human_approval(state):
    if human_answer == "approved":
        return END
    else:
        messages = state.get("messages", [])
        has_web_search = any(
            hasattr(msg, 'name') and msg.name == 'web_search' 
            for msg in messages
        )
        
        if has_web_search:
            return END  # ✅ 已用过网络搜索，停止重试
        else:
            return "fourth_chatbot"  # ✅ 首次拒绝，尝试网络搜索
```

## 优点

1. **质量保证**：所有答案（包括网络搜索结果）都经过评估和审核
2. **防止循环**：最多只尝试一次网络搜索，避免无限循环
3. **用户体验**：如果两次答案都不满意，系统知道何时停止
4. **一致性**：所有答案生成路径都遵循相同的质量控制流程

## 测试建议

### 测试用例1: 首次拒绝，网络搜索成功
```
用户: "检索上下文，上海今天天气"
→ 知识库无相关内容 → 评分低 → 拒绝
→ 网络搜索 → 返回实时天气 → 评分高 → 自动通过 ✅
```

### 测试用例2: 两次拒绝
```
用户: "检索上下文，上海今天天气"
→ 知识库无相关内容 → 评分低 → 拒绝
→ 网络搜索 → 返回天气信息 → 评分低 → 拒绝
→ 系统停止，显示最终答案 ⚠️
```

### 测试用例3: 网络搜索直接通过
```
用户: "检索上下文，最新AI新闻"
→ 知识库内容过时 → 评分低 → 拒绝
→ 网络搜索 → 返回最新新闻 → 评分 0.85 → 直接通过 ✅
```


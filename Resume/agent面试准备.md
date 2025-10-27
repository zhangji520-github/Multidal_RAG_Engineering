# Agent 面试题准备

## 1. 多智能体交互方式有几种？

### 面试回答（口语化表述）

"多智能体交互方式主要有**两种核心模式**：**Tool Calling模式**和**Handoffs模式**。

**第一种是Tool Calling模式，也叫Supervisor模式**。这种模式下有一个中央协调者，我们叫它Supervisor Agent，它负责调度多个专业的Worker Agent。这些Worker Agent就像工具一样被调用，它们不直接跟用户交互，而是执行完任务后把结果返回给Supervisor。这种模式的控制流是**中心化**的，所有的路由决策都通过Supervisor来做。

这种模式特别适合**任务编排和结构化工作流**的场景。比如说，我在我的多模态RAG项目里就用了类似的思路。我的项目里有多个Agent：第一个Agent负责决策是否需要检索历史上下文，第二个Agent基于检索到的上下文生成回答，第三个Agent基于知识库检索生成答案，第四个Agent在人工拒绝后启用网络搜索。这些Agent按照预定的工作流协作，每个Agent都有明确的职责边界。

**第二种是Handoffs模式，也叫交接模式**。在这种模式下，Agent之间可以直接传递控制权。当前活跃的Agent觉得需要另一个专家Agent来帮忙时，它会把控制权和状态交给下一个Agent。新的Agent可以直接跟用户交互，直到它决定再次交接或结束对话。这种模式的控制流是**去中心化**的，Agent可以动态改变谁是活跃的。

Handoffs模式更适合**多领域对话和专家接管**的场景。比如一个客服系统，用户最开始跟通用客服Agent交流，但当涉及技术问题时，可以交接给技术支持Agent，这个技术Agent会继续跟用户对话。

**另外还有一些衍生的交互方式**，比如**Agent-to-Agent通信协议**（A2A协议），这是一种更底层的通信机制，Agent可以通过JSON-RPC消息进行点对点通信，这在分布式多Agent系统中比较有用。

在我的项目中，虽然主要采用的是类似Supervisor的模式，但我还融入了**Human-in-the-Loop机制**。当系统评估分数低于阈值时，会触发人工审核节点，这其实也是一种特殊的交互模式，让人类审核者作为特殊的'Agent'参与到工作流中。

**总结一下**，两种核心模式的选择标准是：如果有多个独立领域、每个领域逻辑复杂、需要集中控制，就用Tool Calling/Supervisor模式；如果Agent需要跟用户对话、需要点对点协作，就用Handoffs模式。"

---

### 知识点总结

#### 两种核心交互模式对比

| 交互模式 | 控制流 | 工作方式 | 适用场景 | 示例 |
|---------|-------|---------|---------|------|
| **Tool Calling (Supervisor)** | 中心化 | Supervisor调度Worker Agent，Worker不直接与用户交互 | 任务编排、结构化工作流、多领域工具管理 | 个人助理系统（日历+邮件Agent） |
| **Handoffs** | 去中心化 | Agent直接传递控制权，新Agent可与用户直接交互 | 多领域对话、专家接管、上下文切换 | 客服系统（通用→技术支持） |

#### 项目实践关联

**我的多模态RAG项目架构**（类似Supervisor模式）：

```
用户输入
   ↓
process_input（输入处理）
   ↓
first_agent_decision（决策Agent）
   ├→ search_context → second_agent_generate（基于历史上下文）
   ├→ retrieve_database → third_chatbot（基于知识库）
   └→ web_search_node → fourth_chatbot（基于网络搜索）
```

**核心设计特点**：

1. **多Agent协作**：4个专业Agent各司其职
   - Agent 1：智能路由决策
   - Agent 2：历史上下文生成
   - Agent 3：知识库问答
   - Agent 4：网络搜索补充

2. **条件路由机制**：
   - `route_after_first_agent`：决策后的智能分支
   - `route_llm_or_retrieve_database`：历史检索失败的降级
   - `route_after_human_approval`：人工审核后的分支

3. **Human-in-the-Loop**：
   - 当评估分数 < 0.75 时触发人工审核
   - 使用 LangGraph 的 `interrupt()` 机制暂停工作流
   - 支持批准/拒绝决策

#### 参考资料

- [LangChain Multi-agent 文档](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Build a supervisor agent 教程](https://docs.langchain.com/oss/python/langchain/supervisor)
- [Agent-to-agent communication](https://docs.langchain.com/langsmith/server-a2a)

---



from typing import Dict, Any, List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, \
    convert_to_messages




# ----------------- 辅助函数 -----------------
def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        # print(update_label)
        # print("\n")

        if not node_update or not hasattr(node_update, '__iter__'):
            continue
        if 'messages' not in node_update:
            if isinstance(node_update, Sequence) and isinstance(node_update[-1], BaseMessage):
                pretty_print_message(node_update[-1])
            else:
                pass
                # print(node_update)
            # print("--------------\n")
            continue
        
        # 处理 messages：如果已经是 BaseMessage 对象，直接使用；否则尝试转换
        raw_messages = node_update["messages"]
        if isinstance(raw_messages, BaseMessage):
            # 单个消息对象
            messages = [raw_messages]
        elif isinstance(raw_messages, list):
            # 检查是否所有元素都已经是 BaseMessage
            if all(isinstance(m, BaseMessage) for m in raw_messages):
                messages = raw_messages
            else:
                # 需要转换（可能包含字典格式的消息）
                try:
                    messages = convert_to_messages(raw_messages)
                except Exception:
                    # 转换失败，尝试手动过滤
                    messages = [m for m in raw_messages if isinstance(m, BaseMessage)]
                    if not messages:
                        # 没有有效消息，跳过此节点
                        continue
        else:
            # 尝试转换
            try:
                messages = convert_to_messages(raw_messages)
            except Exception:
                # 转换失败，跳过此节点
                continue
        
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def pretty_print_message(message, indent=False):
    # 尝试使用 pretty_repr，如果失败或返回空字符串，则手动格式化
    try:
        pretty_message = message.pretty_repr(html=True)
        
        # 如果 pretty_repr 返回空字符串或只有空白字符（多模态消息可能出现这种情况）
        if not pretty_message or not pretty_message.strip():
            # 手动格式化消息
            msg_type = message.__class__.__name__
            content = getattr(message, 'content', '')
            
            # 如果 content 是列表（多模态消息）
            if isinstance(content, list):
                content_str = f"[多模态内容: {len(content)} 个元素]"
                # 尝试提取文本部分
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                if text_parts:
                    content_str += f"\n文本: {text_parts[0][:100]}..."
            else:
                content_str = str(content)[:200] if content else "(空内容)"
            
            pretty_message = f"{'='*30} {msg_type} {'='*30}\n{content_str}\n"
        
        # 截断过长的 base64 数据（只保留前10个字符 + ...）
        if 'data:image' in pretty_message and 'base64,' in pretty_message:
            import re
            # 匹配 base64 数据并截断
            def truncate_base64(match):
                prefix = match.group(1)  # data:image/...;base64,
                base64_data = match.group(2)
                if len(base64_data) > 10:
                    return f"{prefix}{base64_data[:10]}...[已截断]"
                return match.group(0)
            
            pretty_message = re.sub(
                r'(data:image/[^;]+;base64,)([A-Za-z0-9+/=]+)',
                truncate_base64,
                pretty_message
            )
        
        if not indent:
            print(pretty_message)
        else:
            indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
            print(indented)
    except Exception as e:
        # 如果所有方法都失败，至少打印消息类型
        print(f"[ERROR] 无法打印消息: {message.__class__.__name__} (错误: {str(e)[:50]})")
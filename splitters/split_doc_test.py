import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llm_utils import qwen_embeddings
from langchain_experimental.text_splitter import SemanticChunker
# 第一层：语义分块 (Semantic Chunking) MarkdownHeaderTextSplitter 的作用是根据文档的逻辑结构（即标题）进行初步分割。它确保每个分块都围绕一个特定的主题或子主题（例如，一个章节或一个小节）。
headers_to_split_on = [
    ('#', "Header 1"),
    ('##', "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_document = """# 人工智能发展历程
## 早期发展
人工智能的概念最早可以追溯到20世纪40年代。当时科学家们开始思考机器是否能够模拟人类的思维过程。图灵测试的提出标志着人工智能研究的正式开始。早期的研究主要集中在符号推理和专家系统上。

机器学习作为人工智能的一个重要分支，在这个时期也开始萌芽。感知机的发明为后来的神经网络发展奠定了基础。然而，由于计算能力的限制，早期的人工智能发展相对缓慢。

## 现代突破
21世纪以来，随着计算能力的显著提升和大数据的兴起，人工智能迎来了新的发展高潮。深度学习技术的突破使得计算机在图像识别、语音识别和自然语言处理等领域取得了惊人的进展。

卷积神经网络在计算机视觉领域的应用革命性地改变了图像处理的方式。同时，循环神经网络和注意力机制的发展为自然语言处理带来了新的可能性。

# 未来展望
人工智能的未来发展将更加注重通用人工智能的研究。强化学习、迁移学习等技术将成为重要的研究方向。同时，人工智能的伦理问题也日益受到关注，如何确保AI系统的安全性和可解释性将是未来需要重点解决的问题。

# 英雄联盟语录
## 亚索
死亡如风，常伴吾身。
"""

# split_text 方法会返回一个 Document 对象的列表。每个 Document 对象包含两部分：
# page_content: 该块的实际文本内容。
# metadata: 一个字典，包含了分割时所依据的标题信息。键名就是你在 headers_to_split_on 中定义的（如 "Header 1", "Header 2"），值是对应的标题文本。
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits)
# 第二层：技术分块 (Technical Chunking) 目的：RecursiveCharacterTextSplitter 的作用是在第一层语义分块的基础上，对过大的分块进行进一步切割，使其符合预设的 chunk_size 和 chunk_overlap。
# 创建一个文本分割器，设定 chunk_size 和 chunk_overlap
print('--------------------------------')
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3, # 例如，每个分块最多500个字符
    chunk_overlap=0, # 重叠50个字符
    # 注意：不需要再指定 separators，它会自动处理
)

# 对第一步产生的所有 Document 进行再分割
# final_splits = recursive_text_splitter.split_documents(md_header_splits)
# print(final_splits)

# 语义切割 SemanticChunker 
print("=== 对已分割的文档进行语义切割 ===")
text_splitter = SemanticChunker(
    qwen_embeddings, breakpoint_threshold_type="percentile"
)

semantic_splits = text_splitter.split_documents(md_header_splits)
print(f"语义切割结果 ({len(semantic_splits)} 个分块):")
for i, doc in enumerate(semantic_splits):
    print(f"分块 {i+1}: {doc.page_content[:50]}...")
    print(f"元数据: {doc.metadata}")
    print("---")

print("\n=== 对整个文档直接进行语义切割 ===")
# 直接对整个原始文档进行语义切割，看看效果
# semantic_splitter_direct = SemanticChunker(
#     qwen_embeddings,
#     breakpoint_threshold_type="standard_deviation",  # 尝试不同的阈值类型
#     breakpoint_threshold_amount=0.5  # 调整阈值，让它更容易切割
# )
#
# # 提取纯文本内容
# raw_text = markdown_document.replace('#', '').replace('##', '')  # 移除markdown标记
# direct_semantic_splits = semantic_splitter_direct.create_documents([raw_text])
# print(f"直接语义切割结果 ({len(direct_semantic_splits)} 个分块):")
# for i, doc in enumerate(direct_semantic_splits):
#     print(f"分块 {i+1}: {doc.page_content[:100]}...")
#     print("---")
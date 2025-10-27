from llm_utils import qwen3_max

def web_search(query: str) -> str:
    """
    Search the internet for real-time information. Use when you need current events or up-to-date facts.

    Args:
        query: The search query

    Returns:
        str: Search results from the internet
    """
    response = qwen3_max.invoke(f"Search web and answer: {query}")
    print(response.content)

if __name__ == "__main__":
    web_search("上海今天的天气怎么样?")
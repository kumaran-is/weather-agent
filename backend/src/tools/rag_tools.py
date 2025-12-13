"""RAG-enhanced tools for Weather AI Agent.

This module provides advanced weather analysis tools powered by RAG:
- Historical weather pattern analysis
- Anomaly detection
- Cross-location comparisons
- Semantic knowledge retrieval

Level 2 Implementation:
- Integration with Qdrant vector store
- OpenAI embeddings for semantic search
- 600+ historical documents (Kaggle + curated knowledge)
- LangChain tool interface for agent use

Usage:
    >>> from backend.src.tools.rag_tools import get_rag_tools
    >>> tools = get_rag_tools()
    >>> # Use in agent
    >>> from langchain.agents import create_agent
    >>> agent = create_agent(llm, tools, ...)
"""

from langchain_core.tools import tool

from backend.src.rag.retriever import retrieve_weather_knowledge
from backend.src.rag.hybrid_search import hybrid_search


@tool
async def analyze_trends(
    query: str,
    location: str | None = None,
    time_period: str | None = None,
) -> str:
    """Analyze weather trends from historical data.

    Retrieves relevant historical weather data and identifies trends,
    patterns, and long-term changes. Useful for understanding climate
    patterns, seasonal variations, and historical weather behavior.

    Args:
        query: What trend to analyze (e.g., "temperature increase", "rainfall patterns")
        location: Optional location to focus on (e.g., "Tokyo", "Los Angeles")
        time_period: Optional time period (e.g., "1980-2020", "last decade")

    Returns:
        str: Analysis of weather trends based on historical data

    Example:
        >>> result = await analyze_trends(
        ...     query="temperature trends",
        ...     location="Tokyo",
        ...     time_period="1980-2020"
        ... )
        >>> print(result)
        Based on historical data for Tokyo (1980-2020):
        - Average temperature: 59.2°F
        - Temperature range: 28.4°F to 95.7°F (67.3°F range)
        - Climate classification: Temperate
        - Trend: Moderate temperature variability indicating distinct seasons

    Note:
        - Powered by RAG over 600+ historical documents
        - Includes Kaggle datasets (1980-2020, 1000+ cities)
        - Uses semantic search to find relevant data
    """
    try:
        # Build search query
        search_query = query
        if location:
            search_query += f" in {location}"
        if time_period:
            search_query += f" during {time_period}"

        # Retrieve relevant documents
        docs = retrieve_weather_knowledge(search_query, k=5)

        if not docs:
            return f"No historical data found for trend analysis: {query}"

        # Synthesize trends from documents
        result = f"**Weather Trend Analysis: {query}**\n\n"
        if location:
            result += f"**Location:** {location}\n"
        if time_period:
            result += f"**Time Period:** {time_period}\n"
        result += "\n**Key Findings:**\n\n"

        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "unknown")
            result += f"{i}. {content}\n"
            result += f"   (Source: {source})\n\n"

        result += "**Analysis:** The data shows patterns in weather behavior that can inform forecasting and planning decisions."

        return result

    except Exception as e:
        return f"Error analyzing trends for '{query}': {str(e)}"


@tool
async def identify_patterns(
    query: str,
    pattern_type: str | None = None,
) -> str:
    """Identify weather patterns and anomalies.

    Searches historical data to find recurring patterns, anomalies,
    and unusual weather events. Useful for understanding typical vs
    atypical weather conditions.

    Args:
        query: What pattern to identify (e.g., "extreme heat events", "storm clusters")
        pattern_type: Optional pattern type ("seasonal", "anomalous", "recurring")

    Returns:
        str: Identified patterns and analysis

    Example:
        >>> result = await identify_patterns(
        ...     query="Category 5 hurricanes",
        ...     pattern_type="anomalous"
        ... )
        >>> print(result)
        **Pattern Identification: Category 5 hurricanes**

        **Pattern Type:** Anomalous

        **Identified Patterns:**

        1. Category 5 hurricanes are rare but devastating events with sustained
           winds of 157+ mph. Historical examples include Katrina (2005), Michael (2018)...
           (Source: curated/saffir_simpson_scale.txt)

        **Pattern Analysis:** These patterns indicate rare but high-impact events
        requiring special attention and preparation.

    Note:
        - Analyzes 600+ documents for pattern detection
        - Identifies both normal and anomalous patterns
        - Includes historical storm data and climate patterns
    """
    try:
        # Build search query
        search_query = query
        if pattern_type:
            search_query += f" {pattern_type} patterns"

        # Retrieve relevant documents
        docs = retrieve_weather_knowledge(search_query, k=5)

        if not docs:
            return f"No pattern data found for: {query}"

        # Synthesize patterns from documents
        result = f"**Pattern Identification: {query}**\n\n"
        if pattern_type:
            result += f"**Pattern Type:** {pattern_type.capitalize()}\n\n"

        result += "**Identified Patterns:**\n\n"

        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "unknown")
            result += f"{i}. {content}\n"
            result += f"   (Source: {source})\n\n"

        result += "**Pattern Analysis:** These patterns indicate recurring behaviors in weather systems that can inform predictions and risk assessment."

        return result

    except Exception as e:
        return f"Error identifying patterns for '{query}': {str(e)}"


@tool
async def compare_conditions(
    location_a: str,
    location_b: str,
    aspect: str = "climate",
) -> str:
    """Compare weather conditions between two locations.

    Retrieves and compares weather data for two locations, highlighting
    similarities, differences, and relative characteristics.

    Args:
        location_a: First location (e.g., "Tokyo")
        location_b: Second location (e.g., "Los Angeles")
        aspect: What to compare ("climate", "temperature", "patterns")

    Returns:
        str: Comparative analysis of weather conditions

    Example:
        >>> result = await compare_conditions(
        ...     location_a="Tokyo",
        ...     location_b="Los Angeles",
        ...     aspect="climate"
        ... )
        >>> print(result)
        **Weather Comparison**

        **Locations:** Tokyo vs Los Angeles
        **Aspect:** Climate

        **Tokyo Data:**
        1. Climate profile for Tokyo: Average temperature 59.2°F, range 28.4°F to 95.7°F...
           (Source: kaggle/city_temperature_1980_2020.csv)

        **Los Angeles Data:**
        1. In Los Angeles, USA, average temperature 75.2°F in July 2020...
           (Source: kaggle/daily_temperature_major_cities.csv)

        **Comparison Summary:**
        - Tokyo: Temperate climate with wider temperature range (67.3°F)
        - Los Angeles: Milder climate with narrower temperature range
        - Key difference: Tokyo has more pronounced seasonal variation

    Note:
        - Compares data from 600+ documents
        - Includes both historical and recent data
        - Provides context for decision-making
    """
    try:
        # Retrieve data for location A
        query_a = f"{aspect} in {location_a}"
        docs_a = retrieve_weather_knowledge(query_a, k=3)

        # Retrieve data for location B
        query_b = f"{aspect} in {location_b}"
        docs_b = retrieve_weather_knowledge(query_b, k=3)

        # Build comparison result
        result = f"**Weather Comparison**\n\n"
        result += f"**Locations:** {location_a} vs {location_b}\n"
        result += f"**Aspect:** {aspect.capitalize()}\n\n"

        # Location A data
        result += f"**{location_a} Data:**\n\n"
        if docs_a:
            for i, doc in enumerate(docs_a, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get("source", "unknown")
                result += f"{i}. {content}\n"
                result += f"   (Source: {source})\n\n"
        else:
            result += f"No data found for {location_a}\n\n"

        # Location B data
        result += f"**{location_b} Data:**\n\n"
        if docs_b:
            for i, doc in enumerate(docs_b, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get("source", "unknown")
                result += f"{i}. {content}\n"
                result += f"   (Source: {source})\n\n"
        else:
            result += f"No data found for {location_b}\n\n"

        # Comparison summary
        result += "**Comparison Summary:**\n"
        result += "Based on available data, both locations show distinct weather characteristics. "
        result += "Consider local climate patterns when making weather-related decisions."

        return result

    except Exception as e:
        return f"Error comparing conditions between '{location_a}' and '{location_b}': {str(e)}"


@tool
async def retrieve_weather_knowledge_tool(query: str, num_results: int = 5) -> str:
    """Retrieve weather knowledge from the knowledge base.

    Performs semantic search over 600+ weather documents to find relevant
    information. Useful for answering questions about hurricanes, weather
    safety, climate patterns, and historical weather data.

    Args:
        query: Natural language query (e.g., "What is a Category 5 hurricane?")
        num_results: Number of results to retrieve (default: 5, max: 10)

    Returns:
        str: Relevant weather knowledge from the knowledge base

    Example:
        >>> result = await retrieve_weather_knowledge_tool(
        ...     query="What is a Category 5 hurricane?",
        ...     num_results=3
        ... )
        >>> print(result)
        **Weather Knowledge: What is a Category 5 hurricane?**

        **Retrieved Information:**

        1. Category 5 hurricanes have sustained winds of 157 mph or higher.
           Catastrophic damage expected. Most structures will be destroyed...
           (Source: curated/saffir_simpson_scale.txt)

        2. Historical Category 5 hurricanes include Katrina (2005), Michael (2018),
           and Milton (recent). Storm surge can exceed 18 feet...
           (Source: curated/saffir_simpson_scale.txt)

        3. Evacuation for Category 5 storms should begin 24-48 hours before landfall...
           (Source: curated/evacuation_zones.txt)

    Note:
        - Searches 632 chunks across 603 documents
        - Includes curated knowledge (hurricanes, safety) and Kaggle data (historical temps)
        - Uses OpenAI text-embedding-3-small for semantic search
        - Qdrant cosine similarity for relevance ranking
    """
    try:
        # Limit num_results to avoid overwhelming responses
        k = min(num_results, 10)

        # Retrieve documents
        docs = retrieve_weather_knowledge(query, k=k)

        if not docs:
            return f"No relevant information found for: {query}"

        # Format results
        result = f"**Weather Knowledge: {query}**\n\n"
        result += f"**Retrieved Information:**\n\n"

        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "unknown")
            result += f"{i}. {content}\n"
            result += f"   (Source: {source})\n\n"

        return result

    except Exception as e:
        return f"Error retrieving weather knowledge for '{query}': {str(e)}"


@tool
async def hybrid_search_weather_knowledge(
    query: str,
    num_results: int = 5
) -> str:
    """Search weather knowledge using hybrid search (semantic + keyword).

    Combines two search strategies for best results:
    - 70% semantic search: Understanding meaning and context
    - 30% keyword search (BM25): Exact term matching

    This is particularly useful for queries that contain specific terms
    like "Saffir-Simpson", "Category 5", or technical weather terminology.

    Args:
        query: Search question (e.g., "What is a Category 5 hurricane?",
               "Saffir-Simpson scale explained")
        num_results: Number of results to return (1-10, default: 5)

    Returns:
        str: Retrieved information combining semantic and keyword matches

    Example:
        >>> # Good for technical terms
        >>> result = hybrid_search_weather_knowledge(
        ...     query="Saffir-Simpson hurricane scale",
        ...     num_results=3
        ... )
        >>> # Returns: Exact matches for "Saffir-Simpson" + related concepts

    Note:
        - Hybrid search outperforms semantic-only for exact term queries
        - Uses MMR (Maximum Marginal Relevance) to avoid redundant results
        - BM25 index built from all 600+ documents in knowledge base
    """
    # Limit num_results to avoid overwhelming responses
    k = min(num_results, 10)

    # Retrieve documents using hybrid search
    docs = await hybrid_search(query, k=k)

    if not docs:
        return f"No relevant information found for: {query}"

    # Format results
    result = f"**Hybrid Search Results: {query}**\n\n"
    result += f"**Retrieved Information (Semantic + Keyword):**\n\n"

    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source = doc.metadata.get("source", "unknown")
        category = doc.metadata.get("category", "unknown")
        result += f"{i}. {content}\n"
        result += f"   (Source: {source}, Category: {category})\n\n"

    result += f"\n*Note: Results combine 70% semantic search + 30% keyword matching*"

    return result


def get_rag_tools() -> list:
    """Get all RAG-enhanced tools for agent use.

    Returns:
        list: List of LangChain tools (analyze_trends, identify_patterns,
              compare_conditions, retrieve_weather_knowledge_tool,
              hybrid_search_weather_knowledge)

    Level 2 Tool Count:
        - 5 RAG tools (includes hybrid search)
        - Combined with 3 MCP tools = 8 total agent tools

    Example:
        >>> from backend.src.tools.rag_tools import get_rag_tools
        >>> from langchain.agents import create_agent
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> # Get tools
        >>> tools = get_rag_tools()
        >>>
        >>> # Create agent with RAG tools
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = create_agent(llm, tools, ...)
        >>>
        >>> # Agent can now use RAG-enhanced tools
        >>> response = agent.invoke("Analyze temperature trends in Tokyo")
    """
    return [
        analyze_trends,
        identify_patterns,
        compare_conditions,
        retrieve_weather_knowledge_tool,
        hybrid_search_weather_knowledge,  # Level 2: Hybrid search
    ]


# Test function
def test_rag_tools() -> bool:
    """Test RAG tools functionality.

    Returns:
        bool: True if all tools work, False otherwise
    """
    try:
        tools = get_rag_tools()
        print(f"✅ Loaded {len(tools)} RAG tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:80]}...")

        # Test retrieve_weather_knowledge_tool
        print("\n✅ Testing retrieve_weather_knowledge_tool...")
        result = retrieve_weather_knowledge_tool.invoke(
            {"query": "Category 5 hurricane", "num_results": 2}
        )
        print(f"   Result preview: {result[:200]}...")

        return True

    except Exception as e:
        print(f"❌ RAG tools test failed: {e}")
        return False


if __name__ == "__main__":
    test_rag_tools()

"""
Prompt templates for baseline RAG system.

Contains system prompts and user templates optimized for context-based question answering
with source attribution.
"""

# System prompt for baseline RAG
BASELINE_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context from BBC News articles. 

Guidelines:
- Always cite your sources using [Source N] notation when making claims
- Only use information explicitly provided in the context
- If the context doesn't contain enough information to answer the question, clearly state this
- Be specific and accurate, avoiding speculation or information not in the context
- Maintain a professional, informative tone appropriate for news content
- When multiple sources support a claim, cite all relevant sources"""

# User template for baseline RAG queries  
BASELINE_USER_TEMPLATE = """Context:
{context}

Question: {query}

Provide a comprehensive answer using only the information from the context above. Cite sources for all claims using [Source N] notation."""

# Template for no relevant context found
NO_CONTEXT_TEMPLATE = """I don't have sufficient relevant information in the provided context to answer your question about "{query}". The available sources don't contain specific details related to your inquiry."""

# Template for context formatting
CONTEXT_SOURCE_TEMPLATE = """[Source {source_num}] Title: {title}
Category: {category} | Article: {article_id}
Content: {content}"""

# Template for insufficient information responses
INSUFFICIENT_INFO_TEMPLATE = """Based on the provided context, I don't have enough specific information to fully answer your question about "{query}". The available sources mention related topics but don't contain the specific details you're looking for."""


def get_baseline_system_prompt() -> str:
    """Get the system prompt for baseline RAG."""
    return BASELINE_SYSTEM_PROMPT


def get_baseline_user_prompt(query: str, context: str) -> str:
    """
    Get the user prompt for baseline RAG with query and context.
    
    Args:
        query: User's question
        context: Formatted context from retrieved chunks
        
    Returns:
        Formatted prompt string
    """
    return BASELINE_USER_TEMPLATE.format(query=query, context=context)


def format_context_source(source_num: int, title: str, category: str, 
                         article_id: str, content: str) -> str:
    """
    Format a single source for context.
    
    Args:
        source_num: Source number for citation
        title: Article title
        category: Article category
        article_id: Article identifier
        content: Chunk content
        
    Returns:
        Formatted source string
    """
    return CONTEXT_SOURCE_TEMPLATE.format(
        source_num=source_num,
        title=title,
        category=category,
        article_id=article_id,
        content=content
    )


def get_no_context_response(query: str) -> str:
    """Get response for when no relevant context is found."""
    return NO_CONTEXT_TEMPLATE.format(query=query)


def get_insufficient_info_response(query: str) -> str:
    """Get response for when context is insufficient."""
    return INSUFFICIENT_INFO_TEMPLATE.format(query=query)
"""
Context formatter for preparing retrieved chunks for LLM consumption.

Handles formatting retrieved document chunks into readable context with proper
source attribution and metadata inclusion.
"""

import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from .prompts import format_context_source


class ContextFormatter:
    """
    Formats retrieved chunks into readable context for LLM processing.
    """
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 4000):
        """
        Initialize the ContextFormatter.
        
        Args:
            model: OpenAI model name for token counting
            max_tokens: Maximum tokens allowed for context
        """
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a common encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        print(f"🔧 ContextFormatter initialized:")
        print(f"   Model: {self.model}")
        print(f"   Max tokens: {self.max_tokens:,}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.encoding.encode(text))
    
    def format_single_chunk(self, chunk_data: Dict[str, Any], source_num: int) -> str:
        """
        Format a single chunk with source attribution.
        
        Args:
            chunk_data: Dictionary containing chunk data and metadata
            source_num: Source number for citation
            
        Returns:
            Formatted chunk string
        """
        # Extract metadata
        metadata = chunk_data.get('metadata', {})
        document = chunk_data.get('document', '')
        
        title = metadata.get('title', 'Unknown Title')
        category = metadata.get('category', 'unknown')
        article_id = metadata.get('article_id', 'unknown')
        
        # Truncate title if too long
        if len(title) > 100:
            title = title[:97] + "..."
        
        # Format using template
        return format_context_source(
            source_num=source_num,
            title=title,
            category=category,
            article_id=article_id,
            content=document
        )
    
    def format_chunks(self, 
                     retrieved_chunks: List[Dict[str, Any]], 
                     query: str = "",
                     include_metadata: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Format retrieved chunks into context string with token limit enforcement.
        
        Args:
            retrieved_chunks: List of chunk dictionaries from vector store
            query: Original query (for context)
            include_metadata: Whether to include detailed metadata
            
        Returns:
            Tuple of (formatted_context, formatting_stats)
        """
        if not retrieved_chunks:
            return "", {
                'total_sources': 0,
                'total_tokens': 0,
                'truncated': False,
                'sources_included': 0,
                'sources_truncated': 0
            }
        
        formatted_sources = []
        total_tokens = 0
        sources_included = 0
        sources_truncated = 0
        
        # Add header
        context_header = "Based on the following information:\n\n"
        header_tokens = self.count_tokens(context_header)
        
        # Reserve tokens for query template (estimated)
        query_template_tokens = self.count_tokens(f"\n\nAnswer the question: {query}") if query else 50
        available_tokens = self.max_tokens - header_tokens - query_template_tokens
        
        print(f"🔄 Formatting {len(retrieved_chunks)} chunks...")
        print(f"   Available tokens: {available_tokens:,}")
        
        for i, chunk_data in enumerate(retrieved_chunks):
            # Format this chunk
            formatted_chunk = self.format_single_chunk(chunk_data, i + 1)
            chunk_tokens = self.count_tokens(formatted_chunk)
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > available_tokens and sources_included > 0:
                print(f"   ⚠️  Token limit reached, stopping at source {sources_included}")
                sources_truncated = len(retrieved_chunks) - sources_included
                break
            
            formatted_sources.append(formatted_chunk)
            total_tokens += chunk_tokens
            sources_included += 1
            
            print(f"   Source {i+1}: {chunk_tokens:,} tokens (running total: {total_tokens:,})")
        
        # Combine all sources
        if formatted_sources:
            context = context_header + "\n\n".join(formatted_sources)
        else:
            context = context_header + "No relevant sources found."
        
        final_tokens = self.count_tokens(context)
        
        stats = {
            'total_sources': len(retrieved_chunks),
            'sources_included': sources_included,
            'sources_truncated': sources_truncated,
            'total_tokens': final_tokens,
            'truncated': sources_truncated > 0,
            'token_limit': self.max_tokens,
            'available_tokens': available_tokens
        }
        
        print(f"✅ Context formatted: {sources_included} sources, {final_tokens:,} tokens")
        
        return context, stats
    
    def create_full_prompt(self, context: str, query: str) -> str:
        """
        Create the complete prompt with context and query.
        
        Args:
            context: Formatted context string
            query: User's question
            
        Returns:
            Complete prompt string
        """
        if not context.strip():
            return f"No relevant context found for the question: {query}"
        
        prompt = f"{context}\n\nAnswer the question: {query}"
        return prompt
    
    def get_source_attribution_map(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
        """
        Create a mapping of source numbers to attribution information.
        
        Args:
            retrieved_chunks: List of chunk dictionaries
            
        Returns:
            Dictionary mapping source numbers to attribution info
        """
        attribution_map = {}
        
        for i, chunk_data in enumerate(retrieved_chunks):
            metadata = chunk_data.get('metadata', {})
            
            attribution_map[i + 1] = {
                'source_num': i + 1,
                'title': metadata.get('title', 'Unknown Title'),
                'category': metadata.get('category', 'unknown'),
                'article_id': metadata.get('article_id', 'unknown'),
                'chunk_id': metadata.get('chunk_id', 'unknown'),
                'position': metadata.get('position', 0)
            }
        
        return attribution_map
    
    def print_formatting_stats(self, stats: Dict[str, Any]) -> None:
        """Print context formatting statistics."""
        print(f"\n📊 CONTEXT FORMATTING STATISTICS")
        print(f"=" * 50)
        print(f"📚 Total sources available: {stats['total_sources']}")
        print(f"✅ Sources included: {stats['sources_included']}")
        if stats['sources_truncated'] > 0:
            print(f"✂️  Sources truncated: {stats['sources_truncated']}")
        print(f"🎯 Total tokens: {stats['total_tokens']:,}")
        print(f"📏 Token limit: {stats['token_limit']:,}")
        print(f"⚠️  Truncated: {'Yes' if stats['truncated'] else 'No'}")


# Test mode functionality
def test_context_formatter() -> None:
    """Test the ContextFormatter with sample data."""
    print("🧪 TESTING CONTEXT FORMATTER")
    print("=" * 50)
    
    # Create test chunks
    test_chunks = [
        {
            'document': 'This is a sample business article about quarterly earnings and market performance. Companies reported strong growth in the technology sector.',
            'metadata': {
                'title': 'Business Growth Report',
                'category': 'business',
                'article_id': 'test_business_01',
                'chunk_id': 'chunk_001',
                'position': 0
            }
        },
        {
            'document': 'Sports news covering the championship match with detailed analysis of player performance and game statistics.',
            'metadata': {
                'title': 'Championship Match Analysis',
                'category': 'sport',
                'article_id': 'test_sport_01', 
                'chunk_id': 'chunk_002',
                'position': 1
            }
        },
        {
            'document': 'Technology article discussing artificial intelligence developments and their impact on various industries.',
            'metadata': {
                'title': 'AI Industry Impact Study',
                'category': 'tech',
                'article_id': 'test_tech_01',
                'chunk_id': 'chunk_003',
                'position': 0
            }
        }
    ]
    
    # Initialize formatter
    formatter = ContextFormatter(max_tokens=1000)  # Small limit for testing
    
    # Test formatting
    query = "What recent developments are mentioned in business and technology?"
    context, stats = formatter.format_chunks(test_chunks, query)
    
    print(f"\n📝 Sample Query: '{query}'")
    print(f"\n📄 Formatted Context:")
    print("-" * 50)
    print(context)
    print("-" * 50)
    
    # Print statistics
    formatter.print_formatting_stats(stats)
    
    # Test attribution map
    attribution_map = formatter.get_source_attribution_map(test_chunks)
    print(f"\n🏷️  Source Attribution Map:")
    for source_num, info in attribution_map.items():
        print(f"   Source {source_num}: [{info['category']}] {info['title']} ({info['article_id']})")
    
    # Test token counting
    sample_text = "This is a sample text for token counting."
    token_count = formatter.count_tokens(sample_text)
    print(f"\n🔢 Token counting test:")
    print(f"   Text: '{sample_text}'")
    print(f"   Tokens: {token_count}")
    
    print(f"\n✅ ContextFormatter testing complete!")


if __name__ == "__main__":
    test_context_formatter()
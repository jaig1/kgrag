"""
Document Chunking Module

Implements sentence-aware chunking with overlapping windows for optimal
context preservation in vector embeddings.
"""

import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from ..config import config


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    article_id: str
    position: int
    total_chunks: int
    category: str
    title: str
    is_first_chunk: bool
    is_last_chunk: bool
    char_count: int
    word_count: int
    sentence_count: int
    overlap_with_previous: int
    overlap_with_next: int


class DocumentChunker:
    """
    Handles sentence-aware chunking of documents with overlapping windows.
    
    Creates chunks that preserve sentence boundaries and maintain context
    through overlapping content between adjacent chunks.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Ensure overlap is less than chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")
        
        # Pattern for sentence boundaries (more sophisticated than simple period)
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n\s*(?=[A-Z])|(?<=[.!?])\s*$'
        )
        
        # Pattern for additional sentence markers
        self.sentence_markers = re.compile(r'[.!?]+')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving sentence boundaries.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
        
        # Split by sentence pattern
        sentences = self.sentence_pattern.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                cleaned_sentences.append(sentence)
        
        # If no sentences found, split by periods as fallback
        if not cleaned_sentences:
            parts = text.split('.')
            cleaned_sentences = [part.strip() + '.' for part in parts if part.strip()]
        
        return cleaned_sentences
    
    def create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """
        Create chunks from sentences with overlapping windows.
        
        Args:
            sentences: List of sentences to chunk
            
        Returns:
            List of text chunks
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        sentence_index = 0
        
        while sentence_index < len(sentences):
            # Add sentences to current chunk until size limit is reached
            while (sentence_index < len(sentences) and 
                   len(current_chunk + " " + sentences[sentence_index]) <= self.chunk_size):
                
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentences[sentence_index]
                sentence_index += 1
            
            # If we have content, save the chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Prepare for next chunk with overlap
                if sentence_index < len(sentences):
                    # Find overlap content from the end of current chunk
                    overlap_content = self.extract_overlap_content(current_chunk)
                    current_chunk = overlap_content
                else:
                    current_chunk = ""
            else:
                # Handle case where single sentence is longer than chunk_size
                if sentence_index < len(sentences):
                    long_sentence = sentences[sentence_index]
                    # Split long sentence into smaller parts
                    sentence_chunks = self.split_long_sentence(long_sentence)
                    chunks.extend(sentence_chunks)
                    sentence_index += 1
                    current_chunk = ""
        
        return chunks
    
    def extract_overlap_content(self, chunk: str) -> str:
        """
        Extract content from end of chunk for overlap with next chunk.
        
        Args:
            chunk: Current chunk text
            
        Returns:
            Overlap content for next chunk
        """
        if len(chunk) <= self.chunk_overlap:
            return chunk
        
        # Try to find sentence boundaries for clean overlap
        overlap_start = len(chunk) - self.chunk_overlap
        
        # Look for sentence boundary within overlap region
        overlap_region = chunk[overlap_start:]
        sentence_boundaries = list(self.sentence_markers.finditer(overlap_region))
        
        if sentence_boundaries:
            # Start from the first complete sentence in overlap region
            first_boundary = sentence_boundaries[0]
            overlap_start_adjusted = overlap_start + first_boundary.end()
            return chunk[overlap_start_adjusted:].strip()
        
        # Fallback: use word boundaries
        words = chunk.split()
        if len(words) > 0:
            # Calculate approximate word count for overlap
            avg_word_length = len(chunk) / len(words)
            overlap_words = max(1, int(self.chunk_overlap / avg_word_length))
            return " ".join(words[-overlap_words:])
        
        return chunk[-self.chunk_overlap:]
    
    def split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence that's longer than chunk_size into smaller parts.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List of sentence parts
        """
        if len(sentence) <= self.chunk_size:
            return [sentence]
        
        parts = []
        words = sentence.split()
        current_part = ""
        
        for word in words:
            if len(current_part + " " + word) <= self.chunk_size:
                if current_part:
                    current_part += " "
                current_part += word
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def chunk_document(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document with metadata.
        
        Args:
            article: Article dictionary with text and metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if 'text' not in article:
            raise ValueError("Article must contain 'text' field")
        
        text = article['text']
        article_id = article.get('article_id', 'unknown')
        category = article.get('category', 'unknown')
        title = article.get('generated_title', article.get('title', 'Untitled'))
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Create chunks from sentences
        chunk_texts = self.create_chunks_from_sentences(sentences)
        
        # Create chunk objects with metadata
        chunks = []
        total_chunks = len(chunk_texts)
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{article_id}_chunk_{i:03d}"
            
            # Calculate overlap information
            overlap_with_previous = 0
            overlap_with_next = 0
            
            if i > 0:
                # Calculate overlap with previous chunk
                prev_chunk = chunk_texts[i-1]
                overlap_with_previous = self.calculate_overlap(prev_chunk, chunk_text)
            
            if i < len(chunk_texts) - 1:
                # Calculate overlap with next chunk
                next_chunk = chunk_texts[i+1]
                overlap_with_next = self.calculate_overlap(chunk_text, next_chunk)
            
            # Count sentences in chunk
            chunk_sentences = self.split_into_sentences(chunk_text)
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                article_id=article_id,
                position=i,
                total_chunks=total_chunks,
                category=category,
                title=title,
                is_first_chunk=(i == 0),
                is_last_chunk=(i == total_chunks - 1),
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                sentence_count=len(chunk_sentences),
                overlap_with_previous=overlap_with_previous,
                overlap_with_next=overlap_with_next
            )
            
            chunk_dict = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'metadata': metadata.__dict__,
                # Preserve original article metadata
                'source_article': {
                    'article_id': article_id,
                    'category': category,
                    'title': title,
                    'original_index': article.get('original_index'),
                    'label': article.get('label'),
                    'label_text': article.get('label_text')
                }
            }
            
            chunks.append(chunk_dict)
        
        return chunks
    
    def calculate_overlap(self, text1: str, text2: str) -> int:
        """
        Calculate character overlap between two text chunks.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Number of overlapping characters
        """
        if not text1 or not text2:
            return 0
        
        # Find the longest common suffix of text1 and prefix of text2
        max_overlap = min(len(text1), len(text2), self.chunk_overlap)
        
        for i in range(max_overlap, 0, -1):
            if text1[-i:].strip() == text2[:i].strip():
                return i
        
        return 0
    
    def chunk_documents(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Chunk multiple documents and calculate statistics.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (all_chunks, chunking_stats)
        """
        print(f"✂️ Chunking {len(articles)} documents...")
        print(f"   Chunk size: {self.chunk_size} characters")
        print(f"   Chunk overlap: {self.chunk_overlap} characters")
        
        all_chunks = []
        chunk_counts = []
        chunk_sizes = []
        total_overlap = 0
        
        for i, article in enumerate(articles):
            try:
                chunks = self.chunk_document(article)
                all_chunks.extend(chunks)
                
                chunk_counts.append(len(chunks))
                for chunk in chunks:
                    chunk_sizes.append(chunk['metadata']['char_count'])
                    total_overlap += chunk['metadata']['overlap_with_previous']
                
                if (i + 1) % 10 == 0 or (i + 1) == len(articles):
                    print(f"   Chunked {i + 1}/{len(articles)} articles... ({len(all_chunks)} chunks so far)")
                    
            except Exception as e:
                print(f"❌ Error chunking article {i}: {e}")
                continue
        
        # Calculate statistics
        avg_chunks_per_article = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        stats = {
            'total_articles': len(articles),
            'total_chunks': len(all_chunks),
            'avg_chunks_per_article': avg_chunks_per_article,
            'min_chunks_per_article': min(chunk_counts) if chunk_counts else 0,
            'max_chunks_per_article': max(chunk_counts) if chunk_counts else 0,
            'avg_chunk_size': avg_chunk_size,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'total_overlap_chars': total_overlap,
            'avg_overlap_per_chunk': total_overlap / len(all_chunks) if all_chunks else 0,
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap
        }
        
        print(f"✅ Chunking complete!")
        print(f"   Total chunks created: {len(all_chunks)}")
        print(f"   Average chunks per article: {avg_chunks_per_article:.1f}")
        print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        
        return all_chunks, stats


def main():
    """Demo function showing document chunking capabilities."""
    print("✂️ Document Chunker Demo")
    print("=" * 40)
    
    # Sample article for testing
    sample_article = {
        'article_id': 'demo_001',
        'category': 'demo',
        'text': """This is the first sentence of our demo article. It contains multiple sentences that will be used for chunking demonstration. The chunker should preserve sentence boundaries while creating overlapping chunks for better context preservation.

        Here is another paragraph with more content. This paragraph contains additional sentences that will help demonstrate the chunking algorithm. The system should handle paragraph breaks and maintain readability.
        
        Finally, we have a third paragraph to ensure we have enough content for multiple chunks. This will show how the overlapping mechanism works between chunks and how metadata is preserved throughout the process.""",
        'generated_title': 'Demo Article for Chunking'
    }
    
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    
    print(f"Original article length: {len(sample_article['text'])} characters")
    print(f"Chunk size: {chunker.chunk_size}, Overlap: {chunker.chunk_overlap}")
    
    chunks = chunker.chunk_document(sample_article)
    
    print(f"\nGenerated {len(chunks)} chunks:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Length: {chunk['metadata']['char_count']} chars")
        print(f"Overlap with previous: {chunk['metadata']['overlap_with_previous']} chars")
        print(f"Text: {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
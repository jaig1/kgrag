"""
Text Preprocessing Module

Handles cleaning and normalization of raw text data for the KG-Enhanced RAG system.
Prepares text for optimal chunking and embedding generation.
"""

import re
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class PreprocessingStats:
    """Statistics from text preprocessing."""
    original_length: int
    cleaned_length: int
    chars_removed: int
    whitespace_normalized: bool
    special_chars_processed: bool
    title_generated: bool


class TextPreprocessor:
    """
    Handles text cleaning and normalization for document processing.
    
    Removes extra whitespace, normalizes special characters, and generates
    titles from content when needed.
    """
    
    def __init__(self):
        """Initialize the text preprocessor with cleaning patterns."""
        # Pattern for multiple whitespace (spaces, tabs, newlines)
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Pattern for special characters (keep essential punctuation)
        self.special_char_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\'\"\$\%\&]')
        
        # Pattern for multiple consecutive periods
        self.multiple_periods_pattern = re.compile(r'\.{3,}')
        
        # Pattern for sentence boundaries (for title extraction)
        self.sentence_boundary_pattern = re.compile(r'[.!?]+\s+')
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and normalize to single spaces.
        
        Args:
            text: Input text with potential extra whitespace
            
        Returns:
            Text with normalized whitespace
        """
        # Replace all whitespace sequences with single space
        normalized = self.whitespace_pattern.sub(' ', text)
        return normalized.strip()
    
    def normalize_special_characters(self, text: str) -> str:
        """
        Remove or normalize special characters while keeping essential punctuation.
        
        Args:
            text: Input text with potential special characters
            
        Returns:
            Text with normalized special characters
        """
        # Replace multiple periods with ellipsis
        text = self.multiple_periods_pattern.sub('...', text)
        
        # Remove unwanted special characters
        text = self.special_char_pattern.sub(' ', text)
        
        # Clean up any extra spaces introduced
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def generate_title_from_content(self, text: str, max_words: int = 10) -> str:
        """
        Generate a title from the first sentence or first N words of content.
        
        Args:
            text: Input text content
            max_words: Maximum number of words for the title
            
        Returns:
            Generated title string
        """
        if not text.strip():
            return "Untitled Article"
        
        # Try to get first sentence
        sentences = self.sentence_boundary_pattern.split(text)
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            
            # If first sentence is reasonable length, use it
            words = first_sentence.split()
            if len(words) <= max_words:
                return first_sentence
            else:
                # Use first N words if sentence is too long
                return ' '.join(words[:max_words]) + '...'
        
        # Fallback: use first N words
        words = text.split()
        if len(words) <= max_words:
            return ' '.join(words)
        else:
            return ' '.join(words[:max_words]) + '...'
    
    def preprocess_text(self, text: str, generate_title: bool = False) -> Tuple[str, str, PreprocessingStats]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw input text
            generate_title: Whether to generate a title from content
            
        Returns:
            Tuple of (cleaned_text, generated_title, preprocessing_stats)
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        
        original_length = len(text)
        
        # Step 1: Remove extra whitespace
        cleaned_text = self.remove_extra_whitespace(text)
        
        # Step 2: Normalize special characters
        cleaned_text = self.normalize_special_characters(cleaned_text)
        
        # Step 3: Final cleanup
        cleaned_text = cleaned_text.strip()
        
        # Step 4: Generate title if requested
        title = ""
        if generate_title:
            title = self.generate_title_from_content(cleaned_text)
        
        # Calculate statistics
        cleaned_length = len(cleaned_text)
        stats = PreprocessingStats(
            original_length=original_length,
            cleaned_length=cleaned_length,
            chars_removed=original_length - cleaned_length,
            whitespace_normalized=True,
            special_chars_processed=True,
            title_generated=generate_title
        )
        
        return cleaned_text, title, stats
    
    def preprocess_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a complete article with metadata.
        
        Args:
            article: Article dictionary with text and metadata
            
        Returns:
            Article dictionary with preprocessed text and additional metadata
        """
        if 'text' not in article:
            raise ValueError("Article must contain 'text' field")
        
        # Preprocess the text
        cleaned_text, generated_title, stats = self.preprocess_text(
            article['text'], 
            generate_title=True
        )
        
        # Create enhanced article with preprocessing metadata
        enhanced_article = article.copy()
        enhanced_article.update({
            'original_text': article['text'],
            'text': cleaned_text,
            'generated_title': generated_title,
            'preprocessing_stats': {
                'original_length': stats.original_length,
                'cleaned_length': stats.cleaned_length,
                'chars_removed': stats.chars_removed,
                'compression_ratio': stats.cleaned_length / stats.original_length if stats.original_length > 0 else 0
            }
        })
        
        return enhanced_article
    
    def preprocess_articles(self, articles: list) -> Tuple[list, Dict[str, Any]]:
        """
        Preprocess a list of articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (preprocessed_articles, summary_stats)
        """
        preprocessed_articles = []
        total_original_length = 0
        total_cleaned_length = 0
        total_chars_removed = 0
        
        print(f"🔄 Preprocessing {len(articles)} articles...")
        
        for i, article in enumerate(articles):
            try:
                enhanced_article = self.preprocess_article(article)
                preprocessed_articles.append(enhanced_article)
                
                stats = enhanced_article['preprocessing_stats']
                total_original_length += stats['original_length']
                total_cleaned_length += stats['cleaned_length']
                total_chars_removed += stats['chars_removed']
                
                if (i + 1) % 10 == 0 or (i + 1) == len(articles):
                    print(f"   Processed {i + 1}/{len(articles)} articles...")
                    
            except Exception as e:
                print(f"❌ Error preprocessing article {i}: {e}")
                continue
        
        # Calculate summary statistics
        avg_compression = total_cleaned_length / total_original_length if total_original_length > 0 else 0
        
        summary_stats = {
            'total_articles': len(preprocessed_articles),
            'total_original_length': total_original_length,
            'total_cleaned_length': total_cleaned_length,
            'total_chars_removed': total_chars_removed,
            'average_compression_ratio': avg_compression,
            'average_original_length': total_original_length / len(preprocessed_articles) if preprocessed_articles else 0,
            'average_cleaned_length': total_cleaned_length / len(preprocessed_articles) if preprocessed_articles else 0
        }
        
        print(f"✅ Preprocessing complete!")
        print(f"   Articles processed: {len(preprocessed_articles)}")
        print(f"   Average compression ratio: {avg_compression:.3f}")
        print(f"   Total characters removed: {total_chars_removed:,}")
        
        return preprocessed_articles, summary_stats


def main():
    """Demo function showing text preprocessing capabilities."""
    print("🧹 Text Preprocessor Demo")
    print("=" * 40)
    
    # Example text with various issues
    sample_text = """
    This    is   a sample    text with   multiple    spaces.
    
    It has    tabs	and	multiple	newlines...
    
    
    And some @@special@@ characters!!! That need... normalization???
    """
    
    preprocessor = TextPreprocessor()
    
    print("Original text:")
    print(repr(sample_text))
    print(f"Length: {len(sample_text)} characters")
    
    cleaned_text, title, stats = preprocessor.preprocess_text(sample_text, generate_title=True)
    
    print(f"\nCleaned text:")
    print(repr(cleaned_text))
    print(f"Length: {len(cleaned_text)} characters")
    print(f"Generated title: {title}")
    
    print(f"\nPreprocessing stats:")
    print(f"  Original length: {stats.original_length}")
    print(f"  Cleaned length: {stats.cleaned_length}")
    print(f"  Characters removed: {stats.chars_removed}")
    print(f"  Compression ratio: {stats.cleaned_length / stats.original_length:.3f}")


if __name__ == "__main__":
    main()
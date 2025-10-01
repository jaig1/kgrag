"""
Document Processing Pipeline

Orchestrates the complete preprocessing and chunking workflow for the 
KG-Enhanced RAG system. Transforms raw articles into processed chunks
ready for embedding and vector storage.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .preprocessor import TextPreprocessor
from .chunker import DocumentChunker
from ..config import config


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline.
    
    Handles loading raw articles, preprocessing text, chunking documents,
    and saving processed results with comprehensive statistics.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.config = config
        
        # Processing statistics
        self.processing_stats = {}
    
    def load_raw_articles(self, file_path: Path = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load raw articles from JSON file.
        
        Args:
            file_path: Optional path to articles file
            
        Returns:
            Tuple of (articles_list, dataset_metadata)
        """
        if file_path is None:
            file_path = self.config.get_bbc_subset_path()
        
        print(f"📂 Loading raw articles from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            articles = data.get('articles', [])
            metadata = data.get('dataset_info', {})
            
            print(f"✅ Loaded {len(articles)} articles successfully")
            return articles, metadata
            
        except Exception as e:
            print(f"❌ Error loading articles: {e}")
            raise
    
    def process_articles(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process articles through the complete pipeline.
        
        Args:
            articles: List of raw article dictionaries
            
        Returns:
            Tuple of (processed_articles, all_chunks, processing_statistics)
        """
        print("🔄 Starting document processing pipeline...")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Preprocess articles
        print("📋 Step 1: Text Preprocessing")
        processed_articles, preprocessing_stats = self.preprocessor.preprocess_articles(articles)
        print()
        
        # Step 2: Chunk documents
        print("✂️ Step 2: Document Chunking")
        all_chunks, chunking_stats = self.chunker.chunk_documents(processed_articles)
        print()
        
        # Step 3: Calculate comprehensive statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        comprehensive_stats = self.calculate_comprehensive_stats(
            articles, processed_articles, all_chunks, 
            preprocessing_stats, chunking_stats, processing_time
        )
        
        return processed_articles, all_chunks, comprehensive_stats
    
    def calculate_comprehensive_stats(self, 
                                    raw_articles: List[Dict[str, Any]], 
                                    processed_articles: List[Dict[str, Any]], 
                                    all_chunks: List[Dict[str, Any]],
                                    preprocessing_stats: Dict[str, Any],
                                    chunking_stats: Dict[str, Any],
                                    processing_time: float) -> Dict[str, Any]:
        """
        Calculate comprehensive processing statistics.
        
        Args:
            raw_articles: Original raw articles
            processed_articles: Preprocessed articles
            all_chunks: All generated chunks
            preprocessing_stats: Preprocessing statistics
            chunking_stats: Chunking statistics
            processing_time: Total processing time in seconds
            
        Returns:
            Comprehensive statistics dictionary
        """
        # Category distribution in chunks
        category_chunks = {}
        for chunk in all_chunks:
            category = chunk['source_article']['category']
            category_chunks[category] = category_chunks.get(category, 0) + 1
        
        # Chunk size distribution
        chunk_sizes = [chunk['metadata']['char_count'] for chunk in all_chunks]
        chunk_word_counts = [chunk['metadata']['word_count'] for chunk in all_chunks]
        
        # Overlap statistics
        overlaps = [chunk['metadata']['overlap_with_previous'] for chunk in all_chunks 
                   if not chunk['metadata']['is_first_chunk']]
        
        return {
            'processing_metadata': {
                'processing_date': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'chunk_size_config': self.chunker.chunk_size,
                'chunk_overlap_config': self.chunker.chunk_overlap,
                'total_raw_articles': len(raw_articles),
                'total_processed_articles': len(processed_articles),
                'total_chunks_created': len(all_chunks)
            },
            'preprocessing_stats': preprocessing_stats,
            'chunking_stats': chunking_stats,
            'chunk_distribution': {
                'by_category': category_chunks,
                'chunk_sizes': {
                    'mean': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    'min': min(chunk_sizes) if chunk_sizes else 0,
                    'max': max(chunk_sizes) if chunk_sizes else 0,
                    'std': self.calculate_std(chunk_sizes) if chunk_sizes else 0
                },
                'word_counts': {
                    'mean': sum(chunk_word_counts) / len(chunk_word_counts) if chunk_word_counts else 0,
                    'min': min(chunk_word_counts) if chunk_word_counts else 0,
                    'max': max(chunk_word_counts) if chunk_word_counts else 0,
                    'std': self.calculate_std(chunk_word_counts) if chunk_word_counts else 0
                },
                'overlaps': {
                    'mean': sum(overlaps) / len(overlaps) if overlaps else 0,
                    'min': min(overlaps) if overlaps else 0,
                    'max': max(overlaps) if overlaps else 0,
                    'total_overlap_chars': sum(overlaps) if overlaps else 0
                }
            },
            'quality_metrics': {
                'chunks_per_article_ratio': len(all_chunks) / len(processed_articles) if processed_articles else 0,
                'average_chunk_utilization': (sum(chunk_sizes) / len(chunk_sizes)) / self.chunker.chunk_size if chunk_sizes else 0,
                'overlap_efficiency': (sum(overlaps) / len(overlaps)) / self.chunker.chunk_overlap if overlaps else 0
            }
        }
    
    def calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_processed_data(self, 
                          processed_articles: List[Dict[str, Any]], 
                          all_chunks: List[Dict[str, Any]], 
                          stats: Dict[str, Any],
                          output_path: Path = None) -> Path:
        """
        Save processed data to JSON file.
        
        Args:
            processed_articles: Preprocessed articles
            all_chunks: All generated chunks
            stats: Processing statistics
            output_path: Optional output file path
            
        Returns:
            Path where data was saved
        """
        if output_path is None:
            output_path = self.config.PROCESSED_DATA_PATH / "processed_articles.json"
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving processed data to: {output_path}")
        
        # Create output structure
        output_data = {
            'processing_info': {
                'created_date': datetime.now().isoformat(),
                'source_config': {
                    'chunk_size': self.chunker.chunk_size,
                    'chunk_overlap': self.chunker.chunk_overlap,
                    'project_root': str(self.config.PROJECT_ROOT)
                }
            },
            'statistics': stats,
            'articles': processed_articles,
            'all_chunks': all_chunks
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            file_size = output_path.stat().st_size
            print(f"✅ Processed data saved successfully")
            print(f"   File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            raise
    
    def display_processing_summary(self, stats: Dict[str, Any]):
        """
        Display a comprehensive processing summary.
        
        Args:
            stats: Processing statistics
        """
        print("\n📊 Processing Summary")
        print("=" * 50)
        
        # Processing metadata
        meta = stats['processing_metadata']
        print(f"📅 Processing completed: {meta['processing_date'][:19]}")
        print(f"⏱️  Processing time: {meta['processing_time_seconds']:.2f} seconds")
        print(f"📄 Articles processed: {meta['total_processed_articles']}")
        print(f"📝 Chunks created: {meta['total_chunks_created']}")
        
        # Chunking statistics
        chunking = stats['chunking_stats']
        print(f"\n✂️ Chunking Statistics:")
        print(f"   Average chunks per article: {chunking['avg_chunks_per_article']:.1f}")
        print(f"   Chunks per article range: {chunking['min_chunks_per_article']}-{chunking['max_chunks_per_article']}")
        print(f"   Average chunk size: {chunking['avg_chunk_size']:.0f} chars")
        print(f"   Chunk size range: {chunking['min_chunk_size']}-{chunking['max_chunk_size']} chars")
        
        # Category distribution
        print(f"\n🏷️ Category Distribution:")
        category_dist = stats['chunk_distribution']['by_category']
        for category, count in sorted(category_dist.items()):
            print(f"   {category.capitalize()}: {count} chunks")
        
        # Quality metrics
        quality = stats['quality_metrics']
        print(f"\n✅ Quality Metrics:")
        print(f"   Chunk utilization: {quality['average_chunk_utilization']:.1%}")
        print(f"   Overlap efficiency: {quality['overlap_efficiency']:.1%}")
        
        # Preprocessing impact
        preprocessing = stats['preprocessing_stats']
        print(f"\n🧹 Preprocessing Impact:")
        print(f"   Compression ratio: {preprocessing['average_compression_ratio']:.3f}")
        print(f"   Characters removed: {preprocessing['total_chars_removed']:,}")
    
    def display_sample_chunks(self, chunks: List[Dict[str, Any]], num_samples: int = 3):
        """
        Display sample chunks for manual verification.
        
        Args:
            chunks: List of all chunks
            num_samples: Number of sample chunks to display
        """
        print(f"\n📄 Sample Chunks (showing {num_samples} of {len(chunks)})")
        print("=" * 70)
        
        # Show chunks from different articles
        shown_articles = set()
        samples_shown = 0
        
        for chunk in chunks:
            article_id = chunk['source_article']['article_id']
            
            # Skip if we've already shown a chunk from this article
            if article_id in shown_articles and samples_shown > 0:
                continue
            
            if samples_shown >= num_samples:
                break
            
            print(f"\n🔹 Chunk {samples_shown + 1}:")
            print(f"   ID: {chunk['chunk_id']}")
            print(f"   Article: {article_id} ({chunk['source_article']['category']})")
            print(f"   Position: {chunk['metadata']['position'] + 1}/{chunk['metadata']['total_chunks']}")
            print(f"   Length: {chunk['metadata']['char_count']} chars, {chunk['metadata']['word_count']} words")
            print(f"   Overlap: {chunk['metadata']['overlap_with_previous']} chars with previous")
            
            # Show first 200 characters of text
            text_preview = chunk['text'][:200]
            if len(chunk['text']) > 200:
                text_preview += "..."
            print(f"   Text: {text_preview}")
            
            shown_articles.add(article_id)
            samples_shown += 1
    
    def verify_chunk_reconstruction(self, 
                                  articles: List[Dict[str, Any]], 
                                  chunks: List[Dict[str, Any]], 
                                  num_verify: int = 3) -> Dict[str, bool]:
        """
        Verify that chunks can reconstruct original articles.
        
        Args:
            articles: Original processed articles
            chunks: Generated chunks
            num_verify: Number of articles to verify
            
        Returns:
            Dictionary mapping article_id to reconstruction success
        """
        print(f"\n🔍 Verifying chunk reconstruction for {num_verify} articles...")
        
        # Group chunks by article
        chunks_by_article = {}
        for chunk in chunks:
            article_id = chunk['source_article']['article_id']
            if article_id not in chunks_by_article:
                chunks_by_article[article_id] = []
            chunks_by_article[article_id].append(chunk)
        
        # Sort chunks by position
        for article_id in chunks_by_article:
            chunks_by_article[article_id].sort(key=lambda x: x['metadata']['position'])
        
        verification_results = {}
        articles_tested = 0
        
        for article in articles[:num_verify]:
            if articles_tested >= num_verify:
                break
                
            article_id = article['article_id']
            original_text = article['text']
            
            if article_id not in chunks_by_article:
                print(f"❌ No chunks found for article {article_id}")
                verification_results[article_id] = False
                continue
            
            # Reconstruct text from chunks (removing overlaps)
            article_chunks = chunks_by_article[article_id]
            reconstructed_parts = []
            
            for i, chunk in enumerate(article_chunks):
                chunk_text = chunk['text']
                
                # For chunks after the first, remove overlap with previous
                if i > 0:
                    overlap = chunk['metadata']['overlap_with_previous']
                    if overlap > 0:
                        chunk_text = chunk_text[overlap:]
                
                reconstructed_parts.append(chunk_text)
            
            reconstructed_text = " ".join(reconstructed_parts).strip()
            
            # Clean both texts for comparison (normalize whitespace)
            original_clean = " ".join(original_text.split())
            reconstructed_clean = " ".join(reconstructed_text.split())
            
            # Calculate similarity
            similarity = self.calculate_text_similarity(original_clean, reconstructed_clean)
            success = similarity > 0.95  # 95% similarity threshold
            
            print(f"   {article_id}: {'✅' if success else '❌'} Similarity: {similarity:.1%}")
            verification_results[article_id] = success
            articles_tested += 1
        
        success_rate = sum(verification_results.values()) / len(verification_results) if verification_results else 0
        print(f"\n📊 Reconstruction success rate: {success_rate:.1%} ({sum(verification_results.values())}/{len(verification_results)})")
        
        return verification_results
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (simple character-based).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        max_len = max(len(text1), len(text2))
        min_len = min(len(text1), len(text2))
        
        # Count matching characters in the common length
        matches = sum(1 for i in range(min_len) if text1[i] == text2[i])
        
        # Add penalty for length difference
        length_similarity = min_len / max_len
        character_similarity = matches / min_len if min_len > 0 else 0
        
        return (character_similarity * 0.7) + (length_similarity * 0.3)


def main():
    """
    Main function for command-line execution of the processing pipeline.
    """
    print("🔄 Document Processing Pipeline")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Load raw articles
        articles, dataset_metadata = processor.load_raw_articles()
        
        if not articles:
            print("❌ No articles found to process")
            return False
        
        print(f"📊 Dataset info: {len(articles)} articles from {dataset_metadata.get('source', 'unknown')}")
        
        # Process articles
        processed_articles, all_chunks, stats = processor.process_articles(articles)
        
        # Save processed data
        output_path = processor.save_processed_data(processed_articles, all_chunks, stats)
        
        # Display results
        processor.display_processing_summary(stats)
        processor.display_sample_chunks(all_chunks, num_samples=3)
        
        # Verify reconstruction
        processor.verify_chunk_reconstruction(processed_articles, all_chunks, num_verify=3)
        
        print(f"\n🎉 Processing pipeline completed successfully!")
        print(f"📁 Processed data saved to: {output_path}")
        print(f"📊 Generated {len(all_chunks)} chunks from {len(processed_articles)} articles")
        
        # Validate expected ranges
        avg_chunks = len(all_chunks) / len(processed_articles)
        if 3 <= avg_chunks <= 5:
            print(f"✅ Chunk count per article ({avg_chunks:.1f}) is within expected range (3-5)")
        else:
            print(f"⚠️ Chunk count per article ({avg_chunks:.1f}) is outside expected range (3-5)")
        
        if 150 <= len(all_chunks) <= 250:
            print(f"✅ Total chunk count ({len(all_chunks)}) is within expected range (150-250)")
        else:
            print(f"⚠️ Total chunk count ({len(all_chunks)}) is outside expected range (150-250)")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
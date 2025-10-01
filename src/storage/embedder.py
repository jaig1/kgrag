"""
Embedder class for generating and caching OpenAI embeddings.

Handles rate limiting, caching, and batch processing for efficient embedding generation.
"""

import os
import pickle
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


class Embedder:
    """
    Handles embedding generation using OpenAI's API with caching and rate limiting.
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", cache_dir: Path = None):
        """
        Initialize the Embedder.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            cache_dir: Directory to store embedding cache
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir or Path("data/processed")
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        # Statistics tracking
        self.stats = {
            'total_embedded': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }
        
        # Load existing cache
        self.cache = self.load_cache()
        
        print(f"🔧 Embedder initialized:")
        print(f"   Model: {self.model}")
        print(f"   Cache file: {self.cache_file}")
        print(f"   Cached embeddings: {len(self.cache):,}")
    
    def load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"📂 Loaded {len(cache):,} cached embeddings")
                return cache
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
                return {}
        return {}
    
    def save_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"💾 Saved {len(self.cache):,} embeddings to cache")
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5)
    )
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API with retry logic."""
        start_time = time.time()
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Update statistics
            self.stats['api_calls'] += 1
            self.stats['total_time'] += time.time() - start_time
            
            return embeddings
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Validate input text
        if not text or not text.strip():
            print("⚠️  Empty text provided to embedder, returning zero vector")
            # Return a zero vector of the expected dimension (1536 for text-embedding-3-small)
            return [0.0] * 1536
        
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # Generate new embedding
        embeddings = self._call_embedding_api([text])
        embedding = embeddings[0]
        
        # Cache the result
        self.cache[cache_key] = embedding
        self.stats['total_embedded'] += 1
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 100, progress_interval: int = 20) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            progress_interval: Print progress every N chunks
            
        Returns:
            List of embedding vectors
        """
        print(f"🔄 Embedding {len(texts):,} texts in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Check cache for each text in batch
            uncached_indices = []
            uncached_texts = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    batch_embeddings.append(self.cache[cache_key])
                    self.stats['cache_hits'] += 1
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_indices.append(j)
                    uncached_texts.append(text)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self._call_embedding_api(uncached_texts)
                
                # Cache and insert new embeddings
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    cache_key = self._get_cache_key(batch_texts[idx])
                    self.cache[cache_key] = embedding
                    batch_embeddings[idx] = embedding
                    self.stats['total_embedded'] += 1
            
            all_embeddings.extend(batch_embeddings)
            
            # Progress reporting
            if (i // batch_size + 1) % progress_interval == 0 or i + batch_size >= len(texts):
                processed = min(i + batch_size, len(texts))
                cache_rate = (self.stats['cache_hits'] / processed * 100) if processed > 0 else 0
                print(f"   📊 Progress: {processed:,}/{len(texts):,} ({processed/len(texts)*100:.1f}%) - Cache hit rate: {cache_rate:.1f}%")
        
        return all_embeddings
    
    def get_stats(self) -> Dict:
        """Get embedding generation statistics."""
        total_processed = self.stats['total_embedded'] + self.stats['cache_hits']
        cache_hit_rate = (self.stats['cache_hits'] / total_processed * 100) if total_processed > 0 else 0
        avg_time_per_call = (self.stats['total_time'] / self.stats['api_calls']) if self.stats['api_calls'] > 0 else 0
        
        return {
            'total_texts_processed': total_processed,
            'new_embeddings_generated': self.stats['total_embedded'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate_percent': cache_hit_rate,
            'api_calls_made': self.stats['api_calls'],
            'total_api_time_seconds': self.stats['total_time'],
            'average_time_per_api_call': avg_time_per_call,
            'cache_size': len(self.cache),
            'model_used': self.model
        }
    
    def print_stats(self) -> None:
        """Print embedding generation statistics."""
        stats = self.get_stats()
        
        print(f"\n📊 EMBEDDING GENERATION STATISTICS")
        print(f"=" * 50)
        print(f"📝 Total texts processed: {stats['total_texts_processed']:,}")
        print(f"🆕 New embeddings generated: {stats['new_embeddings_generated']:,}")
        print(f"💾 Cache hits: {stats['cache_hits']:,}")
        print(f"🎯 Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")
        print(f"🔌 API calls made: {stats['api_calls_made']:,}")
        print(f"⏱️  Total API time: {stats['total_api_time_seconds']:.2f}s")
        if stats['api_calls_made'] > 0:
            print(f"⚡ Avg time per API call: {stats['average_time_per_api_call']:.2f}s")
        print(f"🗂️  Cache size: {stats['cache_size']:,} embeddings")
        print(f"🤖 Model: {stats['model_used']}")


# Test mode functionality
def test_embedder(api_key: str, test_mode: bool = True) -> None:
    """
    Test the Embedder class with a small subset of data.
    
    Args:
        api_key: OpenAI API key
        test_mode: If True, use only a few sample texts
    """
    print("🧪 TESTING EMBEDDER CLASS")
    print("=" * 50)
    
    # Create test data
    if test_mode:
        test_texts = [
            "This is a test document about artificial intelligence.",
            "The weather today is sunny and warm.",
            "Technology companies are investing in machine learning.",
            "Sports news: The championship game was exciting.",
            "This is a test document about artificial intelligence."  # Duplicate to test caching
        ]
        print(f"🧪 Test mode: Using {len(test_texts)} sample texts")
    else:
        # This would load real data in production
        test_texts = ["Sample text for production mode"]
        print("🚀 Production mode: Would use real data")
    
    # Initialize embedder
    embedder = Embedder(api_key)
    
    # Test single embedding
    print(f"\n📝 Testing single text embedding...")
    embedding = embedder.embed_text(test_texts[0])
    print(f"   ✅ Generated embedding with dimension: {len(embedding)}")
    
    # Test batch embedding
    print(f"\n📦 Testing batch embedding...")
    batch_embeddings = embedder.embed_batch(test_texts, batch_size=2, progress_interval=1)
    print(f"   ✅ Generated {len(batch_embeddings)} embeddings")
    
    # Test caching (run same text again)
    print(f"\n💾 Testing caching...")
    cached_embedding = embedder.embed_text(test_texts[0])
    print(f"   ✅ Cache working: {len(cached_embedding)} dimensions")
    
    # Save cache
    embedder.save_cache()
    
    # Print statistics
    embedder.print_stats()
    
    print(f"\n✅ Embedder testing complete!")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.config import Config
    
    # Check for test mode
    test_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        test_mode = False
    
    try:
        config = Config()
        test_embedder(config.OPENAI_API_KEY, test_mode=test_mode)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
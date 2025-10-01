"""
VectorStore class for ChromaDB integration with OpenAI embeddings.

Provides functionality for storing, searching, and managing document embeddings
with metadata filtering and similarity search capabilities.
"""

import os
import sys
import json
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Handle relative imports
if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.storage.embedder import Embedder
else:
    from .embedder import Embedder


class VectorStore:
    """
    Vector database wrapper for ChromaDB with embedding and search functionality.
    """
    
    def __init__(self, 
                 db_path: Path, 
                 collection_name: str = "bbc_news_chunks",
                 embedding_model: str = "text-embedding-3-small",
                 api_key: str = None):
        """
        Initialize VectorStore with ChromaDB.
        
        Args:
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the collection to create/use
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key for embedding generation
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedder if API key provided
        self.embedder = None
        if api_key:
            self.embedder = Embedder(api_key, model=embedding_model)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        print(f"🗄️  VectorStore initialized:")
        print(f"   Database path: {self.db_path}")
        print(f"   Collection: {self.collection_name}")
        print(f"   Documents in collection: {self.collection.count():,}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            print(f"📂 Retrieved existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "BBC News article chunks with embeddings"}
            )
            print(f"🆕 Created new collection: {self.collection_name}")
            return collection
    
    def add_documents(self, 
                     documents: List[str],
                     metadatas: List[Dict[str, Any]],
                     ids: List[str],
                     embeddings: Optional[List[List[float]]] = None,
                     batch_size: int = 100,
                     progress_interval: int = 20) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            embeddings: Pre-computed embeddings (optional)
            batch_size: Number of documents to process per batch
            progress_interval: Print progress every N batches
        """
        if len(documents) != len(metadatas) != len(ids):
            raise ValueError("Documents, metadatas, and ids must have the same length")
        
        print(f"📚 Adding {len(documents):,} documents to collection...")
        
        # Generate embeddings if not provided
        if embeddings is None and self.embedder is not None:
            print(f"🔄 Generating embeddings for {len(documents):,} documents...")
            embeddings = self.embedder.embed_batch(
                documents, 
                batch_size=batch_size, 
                progress_interval=progress_interval
            )
        elif embeddings is None:
            raise ValueError("No embeddings provided and no embedder configured")
        
        # Add documents in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            batch_documents = documents[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end] if embeddings else None
            
            try:
                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
                
                # Progress reporting
                if (i // batch_size + 1) % progress_interval == 0 or batch_end >= len(documents):
                    print(f"   📊 Progress: {batch_end:,}/{len(documents):,} documents added ({batch_end/len(documents)*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Failed to add batch {i}-{batch_end}: {e}")
                raise
        
        # Verify insertion
        final_count = self.collection.count()
        print(f"✅ Successfully added documents. Collection now has {final_count:,} documents")
    
    def similarity_search(self, 
                         query: str, 
                         top_k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None,
                         include_distances: bool = True) -> Dict[str, Any]:
        """
        Perform similarity search on the collection.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Metadata filters to apply
            include_distances: Whether to include similarity scores
            
        Returns:
            Dictionary with search results
        """
        if self.embedder is None:
            raise ValueError("No embedder configured for search")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
        )
        
        # Debug: Check what ChromaDB is returning
        distances = results['distances'][0] if include_distances and results.get('distances') else []
        print(f"   🔍 VectorStore Debug: Found {len(distances)} distances: {distances[:3] if distances else 'None'}...")
        
        return {
            'query': query,
            'top_k': top_k,
            'filter_metadata': filter_metadata,
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'ids': results['ids'][0] if results['ids'] else [],
            'distances': distances
        }
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(ids=[chunk_id], include=['documents', 'metadatas'])
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving document {chunk_id}: {e}")
            return None
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search documents by metadata only (no similarity).
        
        Args:
            metadata_filter: Metadata filters to apply
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            documents = []
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return documents
        except Exception as e:
            print(f"❌ Error searching by metadata: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Sample some documents to analyze metadata
        sample_results = self.collection.peek(limit=min(10, count))
        
        # Analyze metadata fields
        metadata_fields = set()
        categories = set()
        
        if sample_results['metadatas']:
            for metadata in sample_results['metadatas']:
                metadata_fields.update(metadata.keys())
                if 'category' in metadata:
                    categories.add(metadata['category'])
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'metadata_fields': list(metadata_fields),
            'categories_found': list(categories),
            'sample_ids': sample_results['ids'][:5] if sample_results['ids'] else []
        }
    
    def print_stats(self) -> None:
        """Print collection statistics."""
        stats = self.get_collection_stats()
        
        print(f"\n📊 VECTOR STORE STATISTICS")
        print(f"=" * 50)
        print(f"📚 Collection: {stats['collection_name']}")
        print(f"📄 Total documents: {stats['total_documents']:,}")
        print(f"🏷️  Metadata fields: {', '.join(stats['metadata_fields'])}")
        print(f"📂 Categories found: {', '.join(stats['categories_found'])}")
        print(f"🆔 Sample IDs: {', '.join(stats['sample_ids'])}")
    
    def reset_collection(self) -> None:
        """Delete all documents in the collection (for testing)."""
        print(f"⚠️  Resetting collection: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            print(f"✅ Collection reset successfully")
        except Exception as e:
            print(f"❌ Failed to reset collection: {e}")


# Test mode functionality
def test_vector_store(api_key: str, test_mode: bool = True) -> None:
    """
    Test the VectorStore class with sample data.
    
    Args:
        api_key: OpenAI API key
        test_mode: If True, use small sample dataset
    """
    print("🧪 TESTING VECTOR STORE CLASS")
    print("=" * 50)
    
    # Initialize vector store
    db_path = Path("data/test_chroma_db" if test_mode else "data/chroma_db")
    vector_store = VectorStore(
        db_path=db_path,
        collection_name="test_bbc_chunks" if test_mode else "bbc_news_chunks",
        api_key=api_key
    )
    
    if test_mode:
        # Reset collection for clean test
        vector_store.reset_collection()
        
        # Create test documents
        test_documents = [
            "This is a business article about company earnings and market trends.",
            "Sports news covering the latest football match results and player transfers.",
            "Technology report discussing artificial intelligence and machine learning advances.",
            "Entertainment news about celebrity events and movie premieres.",
            "Political analysis of recent election results and policy changes."
        ]
        
        test_metadatas = [
            {"article_id": "test_business_01", "chunk_id": "chunk_1", "category": "business", "position": 1, "title": "Market Report", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "test_sport_01", "chunk_id": "chunk_2", "category": "sport", "position": 1, "title": "Football Update", "is_first_chunk": True, "is_last_chunk": True},
            {"article_id": "test_tech_01", "chunk_id": "chunk_3", "category": "tech", "position": 1, "title": "AI Advances", "is_first_chunk": True, "is_last_chunk": False},
            {"article_id": "test_entertainment_01", "chunk_id": "chunk_4", "category": "entertainment", "position": 1, "title": "Celebrity News", "is_first_chunk": True, "is_last_chunk": True},
            {"article_id": "test_politics_01", "chunk_id": "chunk_5", "category": "politics", "position": 1, "title": "Election Analysis", "is_first_chunk": True, "is_last_chunk": False}
        ]
        
        test_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        
        print(f"🧪 Test mode: Using {len(test_documents)} sample documents")
        
        # Add documents
        print(f"\n📚 Testing document addition...")
        vector_store.add_documents(
            documents=test_documents,
            metadatas=test_metadatas,
            ids=test_ids,
            batch_size=2,
            progress_interval=1
        )
        
        # Test similarity search
        print(f"\n🔍 Testing similarity search...")
        search_results = vector_store.similarity_search(
            query="latest business news",
            top_k=3
        )
        print(f"   Query: '{search_results['query']}'")
        print(f"   Found {len(search_results['documents'])} results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results['documents'], 
            search_results['metadatas'], 
            search_results['distances']
        )):
            print(f"      {i+1}. [{metadata['category']}] {doc[:60]}... (distance: {distance:.3f})")
        
        # Test metadata filtering
        print(f"\n🏷️  Testing metadata filtering...")
        filtered_results = vector_store.similarity_search(
            query="news update",
            top_k=5,
            filter_metadata={"category": "sport"}
        )
        print(f"   Filter: category='sport'")
        print(f"   Found {len(filtered_results['documents'])} results:")
        for i, (doc, metadata) in enumerate(zip(filtered_results['documents'], filtered_results['metadatas'])):
            print(f"      {i+1}. [{metadata['category']}] {doc[:60]}...")
        
        # Test get by ID
        print(f"\n🆔 Testing get by ID...")
        doc_result = vector_store.get_by_id("chunk_3")
        if doc_result:
            print(f"   Found document: [{doc_result['metadata']['category']}] {doc_result['document'][:60]}...")
        else:
            print(f"   Document not found")
        
        # Test metadata-only search
        print(f"\n📋 Testing metadata-only search...")
        metadata_results = vector_store.search_by_metadata(
            metadata_filter={"category": "tech"},
            limit=10
        )
        print(f"   Found {len(metadata_results)} tech documents")
        
        # Print statistics
        vector_store.print_stats()
        
        # Print embedder stats
        if vector_store.embedder:
            vector_store.embedder.print_stats()
        
        print(f"\n✅ VectorStore testing complete!")
        
    else:
        print("🚀 Production mode: Would load and process real data")
        vector_store.print_stats()


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
        test_vector_store(config.OPENAI_API_KEY, test_mode=test_mode)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
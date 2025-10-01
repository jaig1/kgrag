"""
Graph Construction Pipeline Script

Orchestrates the complete entity extraction, resolution, and knowledge graph construction process.
Integrates EntityExtractor, EntityResolver, and KnowledgeGraphBuilder modules.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.entity_resolver import EntityResolver
from src.ingestion.knowledge_graph_builder import KnowledgeGraphBuilder


def load_processed_articles(config: Config) -> list:
    """Load processed articles from Phase 2."""
    processed_file = config.PROCESSED_DATA_PATH / 'processed_articles.json'
    
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed articles file not found: {processed_file}")
    
    print(f"📂 Loading processed articles from {processed_file}")
    
    with open(processed_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    print(f"   Loaded {len(articles)} processed articles")
    
    return articles


def run_entity_extraction(config: Config, articles: list, test_mode: bool = False) -> dict:
    """Run entity extraction on processed articles."""
    print("\n" + "="*50)
    print("PHASE 3A: ENTITY EXTRACTION")
    if test_mode:
        print("🧪 TEST MODE: Processing only 1 article")
    print("="*50)
    
    extraction_file = config.PROCESSED_DATA_PATH / 'extracted_entities.json'
    
    # Check if extraction results already exist
    if extraction_file.exists():
        print(f"📂 Loading existing extraction results from {extraction_file}")
        with open(extraction_file, 'r') as f:
            extraction_data = json.load(f)
        print(f"   ✅ Loaded existing results: {extraction_data['extraction_metadata']['successful_extractions']} successful extractions")
        return extraction_data
    
    # Limit articles for testing if in test mode
    if test_mode:
        articles = articles[:1]
        print(f"   🧪 Testing with {len(articles)} article(s)")
    
    # Initialize extractor
    extractor = EntityExtractor(config.OPENAI_API_KEY)
    
    # Run extraction
    extraction_results = extractor.extract_entities_from_articles(articles)
    
    # Save extraction results
    extraction_file = config.PROCESSED_DATA_PATH / 'extracted_entities.json'
    print(f"\n💾 Saving extraction results to {extraction_file}")
    
    # Convert EntityExtractionResult objects to dictionaries for JSON serialization
    results_dict = []
    successful_count = 0
    failed_count = 0
    
    for r in extraction_results:
        if r.extraction_successful:
            successful_count += 1
        else:
            failed_count += 1
            
        result_dict = {
            'article_id': r.article_id,
            'extraction_successful': r.extraction_successful,
            'entities': {
                'persons': r.persons if r.persons else [],
                'organizations': r.organizations if r.organizations else [],
                'locations': r.locations if r.locations else [],
                'events': r.events if r.events else [],
                'topics': r.topics if r.topics else []
            }
        }
        
        if r.error_message:
            result_dict['error_message'] = r.error_message
        
        results_dict.append(result_dict)
    
    extraction_data = {
        'extraction_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(articles),
            'successful_extractions': successful_count,
            'failed_extractions': failed_count,
            'test_mode': test_mode
        },
        'extraction_results': results_dict
    }
    
    with open(extraction_file, 'w') as f:
        json.dump(extraction_data, f, indent=2)
    
    print(f"✅ Extraction results saved ({extraction_file.stat().st_size} bytes)")
    
    return extraction_data


def run_entity_resolution(extraction_data: dict) -> dict:
    """Run entity resolution on extraction results."""
    print("\n" + "="*50)
    print("PHASE 3B: ENTITY RESOLUTION")
    print("="*50)
    
    # Initialize resolver
    resolver = EntityResolver()
    
    # Run resolution
    extraction_results = extraction_data.get('extraction_results', [])
    resolution_results = resolver.resolve_entities(extraction_results)
    
    # Save resolution results
    config = Config()
    resolution_file = config.PROCESSED_DATA_PATH / 'resolved_entities.json'
    print(f"\n💾 Saving resolution results to {resolution_file}")
    
    resolution_data = {
        'resolution_metadata': {
            'timestamp': datetime.now().isoformat(),
            **resolution_results.get('resolution_metadata', {})
        },
        'resolved_entities': resolution_results.get('resolved_entities', {}),
        'entity_articles_mapping': resolution_results.get('entity_articles_mapping', {})
    }
    
    with open(resolution_file, 'w') as f:
        json.dump(resolution_data, f, indent=2)
    
    print(f"✅ Resolution results saved ({resolution_file.stat().st_size} bytes)")
    
    return resolution_data


def run_graph_construction(config: Config, resolution_data: dict, articles: list) -> dict:
    """Run knowledge graph construction."""
    print("\n" + "="*50)
    print("PHASE 3C: KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*50)
    
    # Initialize builder
    builder = KnowledgeGraphBuilder()
    
    # Create graph
    graph = builder.create_graph(resolution_data, articles)
    
    # Calculate statistics
    stats = builder.get_graph_statistics(graph)
    
    # Save graph
    graph_file = config.PROCESSED_DATA_PATH / 'knowledge_graph.graphml'
    builder.save_graph(graph, str(graph_file))
    
    # Save statistics
    stats_file = config.PROCESSED_DATA_PATH / 'graph_statistics.json'
    print(f"\n💾 Saving graph statistics to {stats_file}")
    
    graph_data = {
        'graph_metadata': {
            'timestamp': datetime.now().isoformat(),
            'graph_file': str(graph_file),
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        },
        'statistics': stats
    }
    
    with open(stats_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"✅ Graph statistics saved ({stats_file.stat().st_size} bytes)")
    
    return graph_data, graph


def print_phase_summary(extraction_data: dict, resolution_data: dict, graph_data: dict):
    """Print comprehensive phase summary."""
    print("\n" + "="*60)
    print("PHASE 3 COMPLETION SUMMARY")
    print("="*60)
    
    # Extraction summary
    ext_meta = extraction_data.get('extraction_metadata', {})
    print(f"\n📊 Entity Extraction Results:")
    print(f"   Total articles processed: {ext_meta.get('total_articles', 0)}")
    print(f"   Successful extractions: {ext_meta.get('successful_extractions', 0)}")
    print(f"   Failed extractions: {ext_meta.get('failed_extractions', 0)}")
    
    # Resolution summary
    res_meta = resolution_data.get('resolution_metadata', {})
    entity_counts = res_meta.get('unique_entity_counts', {})
    print(f"\n🔗 Entity Resolution Results:")
    print(f"   Unique persons: {entity_counts.get('persons', 0)}")
    print(f"   Unique organizations: {entity_counts.get('organizations', 0)}")
    print(f"   Unique locations: {entity_counts.get('locations', 0)}")
    print(f"   Unique events: {entity_counts.get('events', 0)}")
    print(f"   Unique topics: {entity_counts.get('topics', 0)}")
    
    # Graph summary
    graph_meta = graph_data.get('graph_metadata', {})
    basic_metrics = graph_data.get('statistics', {}).get('basic_metrics', {})
    print(f"\n📈 Knowledge Graph Results:")
    print(f"   Total nodes: {graph_meta.get('num_nodes', 0)}")
    print(f"   Total edges: {graph_meta.get('num_edges', 0)}")
    print(f"   Connected components: {basic_metrics.get('num_connected_components', 0)}")
    print(f"   Graph density: {basic_metrics.get('density', 0.0)}")
    print(f"   Average degree: {basic_metrics.get('average_degree', 0.0)}")
    
    # Top entities by centrality
    centrality = graph_data.get('statistics', {}).get('centrality_analysis', {})
    top_degree = centrality.get('top_degree_centrality', [])
    
    if top_degree:
        print(f"\n⭐ Most Central Entities (by degree):")
        for i, (entity, score) in enumerate(top_degree[:3]):
            entity_type = entity.split(':')[0] if ':' in entity else 'unknown'
            entity_name = entity.split(':', 1)[1] if ':' in entity else entity
            print(f"   {i+1}. {entity_name} ({entity_type}) - {score}")
    
    # Files created
    config = Config()
    print(f"\n📁 Files Created:")
    print(f"   Entity extractions: {config.PROCESSED_DATA_PATH / 'extracted_entities.json'}")
    print(f"   Resolved entities: {config.PROCESSED_DATA_PATH / 'resolved_entities.json'}")
    print(f"   Knowledge graph: {config.PROCESSED_DATA_PATH / 'knowledge_graph.graphml'}")
    print(f"   Graph statistics: {config.PROCESSED_DATA_PATH / 'graph_statistics.json'}")
    
    print(f"\n✅ Phase 3 Complete! Knowledge graph ready for exploration and analysis.")
    print(f"💡 Next: Run the graph exploration notebook to visualize and analyze the knowledge graph.")


def main(test_mode: bool = True):
    """Main pipeline execution function."""
    print("🚀 Knowledge Graph Construction Pipeline")
    print("=" * 50)
    if test_mode:
        print("🧪 TEST MODE: Processing 1 article to validate pipeline")
    else:
        print("Executing Phase 3: Entity Extraction & Knowledge Graph Construction")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        config.validate_config()
        
        # Load processed articles from Phase 2
        articles = load_processed_articles(config)
        
        # Step 1: Entity Extraction
        extraction_data = run_entity_extraction(config, articles, test_mode)
        
        # Step 2: Entity Resolution
        resolution_data = run_entity_resolution(extraction_data)
        
        # Step 3: Knowledge Graph Construction
        graph_data, graph = run_graph_construction(config, resolution_data, articles)
        
        # Print comprehensive summary
        print_phase_summary(extraction_data, resolution_data, graph_data)
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Print some debugging info
        import traceback
        print(f"\n🔍 Full error traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    import sys
    # Check if we should run in full mode
    test_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        test_mode = False
        print("⚠️  Full mode requested - will process all 50 articles")
    
    success = main(test_mode)
    sys.exit(0 if success else 1)
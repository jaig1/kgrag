"""
Graph Traversal Module for KG-Enhanced RAG System.

This module provides functionality to navigate the knowledge graph for multi-hop reasoning,
entity discovery, and relationship exploration.
"""

import time
import pickle
from typing import Dict, List, Any, Tuple, Set, Optional
from pathlib import Path
import sys
from collections import defaultdict, deque

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# NetworkX for graph operations
import networkx as nx


class GraphTraversal:
    """
    Provides graph traversal functionality for the knowledge graph.
    
    This class enables multi-hop exploration of entity relationships,
    finding related entities and articles, and extracting context information.
    """
    
    def __init__(self, graph_path: Path = None):
        """
        Initialize the GraphTraversal with the knowledge graph.
        
        Args:
            graph_path: Path to the pickled NetworkX graph file
        """
        self.graph_path = graph_path or Path("data/processed/knowledge_graph.pkl")
        self.graph = None
        
        # Statistics tracking
        self.stats = {
            'graph_loaded_time': 0.0,
            'total_traversals': 0,
            'total_hops_explored': 0,
            'entities_found': 0,
            'articles_found': 0,
            'average_traversal_time': 0.0
        }
        
        # Load the graph
        self._load_graph()
        
        if self.graph is not None:
            print(f"🕸️  GraphTraversal initialized:")
            print(f"   Graph file: {self.graph_path}")
            print(f"   Nodes: {self.graph.number_of_nodes():,}")
            print(f"   Edges: {self.graph.number_of_edges():,}")
        else:
            print(f"❌ GraphTraversal initialization failed - no graph loaded")
    
    def _load_graph(self) -> None:
        """Load the knowledge graph from the pickle file."""
        start_time = time.time()
        
        try:
            if not self.graph_path.exists():
                print(f"❌ Graph file not found: {self.graph_path}")
                return
            
            print(f"📂 Loading knowledge graph from {self.graph_path}...")
            
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            load_time = time.time() - start_time
            self.stats['graph_loaded_time'] = load_time
            
            print(f"✅ Knowledge graph loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            self.graph = None
    
    def find_entity_in_graph(self, entity_name: str, entity_type: str = None) -> List[str]:
        """
        Find entity nodes in the graph that match the given name and type.
        
        Args:
            entity_name: Name of the entity to find
            entity_type: Optional entity type filter (person, organization, etc.)
            
        Returns:
            List of node IDs that match the criteria
        """
        if self.graph is None:
            print("❌ No graph loaded")
            return []
        
        matching_nodes = []
        entity_name_lower = entity_name.lower()
        
        print(f"🔍 Searching for entity: '{entity_name}' (type: {entity_type or 'any'})")
        
        # Search through all nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', '')
            node_name = node_data.get('name', '')
            
            # Skip article nodes - we only want entity nodes
            if node_type == 'article':
                continue
            
            # Type filter if specified
            if entity_type and node_type != entity_type:
                continue
            
            # Name matching (flexible)
            if (entity_name_lower in node_name.lower() or 
                node_name.lower() in entity_name_lower or
                entity_name_lower == node_name.lower()):
                matching_nodes.append(node_id)
        
        print(f"   Found {len(matching_nodes)} matching nodes")
        return matching_nodes
    
    def multi_hop_traversal(self, start_entity: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Perform multi-hop traversal starting from an entity node.
        
        Args:
            start_entity: Starting entity node ID or name
            max_hops: Maximum number of hops to traverse
            
        Returns:
            Dictionary containing related entities, articles, and traversal paths
        """
        if self.graph is None:
            return self._empty_traversal_result()
        
        start_time = time.time()
        
        # Find starting nodes if entity name provided instead of node ID
        start_nodes = []
        if start_entity in self.graph:
            start_nodes = [start_entity]
        else:
            start_nodes = self.find_entity_in_graph(start_entity)
        
        if not start_nodes:
            print(f"❌ Starting entity '{start_entity}' not found in graph")
            return self._empty_traversal_result()
        
        print(f"🕸️  Multi-hop traversal from '{start_entity}' (max hops: {max_hops})")
        
        # Perform traversal from each starting node
        all_related_entities = set()
        all_related_articles = set()
        all_paths = []
        
        for start_node in start_nodes:
            result = self._traverse_from_node(start_node, max_hops)
            all_related_entities.update(result['entities'])
            all_related_articles.update(result['articles'])
            all_paths.extend(result['paths'])
        
        # Update statistics
        traversal_time = time.time() - start_time
        self._update_traversal_stats(traversal_time, max_hops, len(all_related_entities), len(all_related_articles))
        
        result = {
            'start_entity': start_entity,
            'start_nodes': start_nodes,
            'max_hops': max_hops,
            'related_entities': list(all_related_entities),
            'related_articles': list(all_related_articles),
            'traversal_paths': all_paths,
            'traversal_time': traversal_time,
            'entities_count': len(all_related_entities),
            'articles_count': len(all_related_articles),
            'paths_count': len(all_paths)
        }
        
        print(f"✅ Traversal complete: {len(all_related_entities)} entities, {len(all_related_articles)} articles in {traversal_time:.2f}s")
        
        return result
    
    def _traverse_from_node(self, start_node: str, max_hops: int) -> Dict[str, Any]:
        """Traverse from a single node using BFS."""
        visited = set()
        related_entities = set()
        related_articles = set()
        paths = []
        
        # BFS queue: (current_node, path, hop_count)
        queue = deque([(start_node, [start_node], 0)])
        visited.add(start_node)
        
        while queue:
            current_node, path, hop_count = queue.popleft()
            
            # Get current node data
            node_data = self.graph.nodes[current_node]
            node_type = node_data.get('type', '')
            
            # Collect entities and articles
            if node_type == 'article':
                related_articles.add(current_node)
            elif node_type in ['person', 'organization', 'location', 'event', 'topic']:
                if current_node != start_node:  # Don't include the starting entity
                    related_entities.add(current_node)
            
            # Continue traversal if within hop limit
            if hop_count < max_hops:
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path, hop_count + 1))
                        
                        # Store interesting paths (those that reach entities or articles)
                        neighbor_data = self.graph.nodes[neighbor]
                        neighbor_type = neighbor_data.get('type', '')
                        if (neighbor_type == 'article' or 
                            neighbor_type in ['person', 'organization', 'location', 'event', 'topic']):
                            paths.append({
                                'path': new_path,
                                'hop_count': hop_count + 1,
                                'end_type': neighbor_type,
                                'end_name': neighbor_data.get('name', neighbor)
                            })
        
        return {
            'entities': related_entities,
            'articles': related_articles,
            'paths': paths
        }
    
    def find_related_entities(self, query_entities: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """
        Find all entities related to a list of query entities within hop limit.
        
        Args:
            query_entities: List of entity names or IDs
            max_hops: Maximum hops to traverse from each entity
            
        Returns:
            Dictionary with related entities and their relationships
        """
        if self.graph is None:
            return {'related_entities': [], 'entity_relationships': {}}
        
        print(f"🔍 Finding entities related to: {query_entities}")
        
        all_related_entities = set()
        entity_relationships = defaultdict(list)
        traversal_results = {}
        
        for entity in query_entities:
            print(f"\n🕸️  Traversing from: {entity}")
            result = self.multi_hop_traversal(entity, max_hops)
            
            # Collect related entities
            all_related_entities.update(result['related_entities'])
            
            # Store relationships
            for related_entity in result['related_entities']:
                entity_relationships[entity].append(related_entity)
            
            traversal_results[entity] = result
        
        # Remove duplicates and original query entities
        final_related_entities = list(all_related_entities - set(query_entities))
        
        result = {
            'query_entities': query_entities,
            'related_entities': final_related_entities,
            'entity_relationships': dict(entity_relationships),
            'traversal_results': traversal_results,
            'total_related_count': len(final_related_entities)
        }
        
        print(f"✅ Found {len(final_related_entities)} unique related entities")
        
        return result
    
    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Get comprehensive context information for an entity.
        
        Args:
            entity_id: Entity node ID or name
            
        Returns:
            Dictionary with entity attributes, connected articles, and relationships
        """
        if self.graph is None:
            return {'error': 'No graph loaded'}
        
        # Find the entity node
        entity_nodes = []
        if entity_id in self.graph:
            entity_nodes = [entity_id]
        else:
            entity_nodes = self.find_entity_in_graph(entity_id)
        
        if not entity_nodes:
            return {'error': f'Entity "{entity_id}" not found in graph'}
        
        print(f"📖 Getting context for entity: {entity_id}")
        
        # Use the first matching entity node
        entity_node = entity_nodes[0]
        entity_data = self.graph.nodes[entity_node]
        
        # Get direct connections
        connected_articles = []
        connected_entities = []
        relationship_types = set()
        
        for neighbor in self.graph.neighbors(entity_node):
            neighbor_data = self.graph.nodes[neighbor]
            neighbor_type = neighbor_data.get('type', '')
            
            # Get edge data for relationship type
            edge_data = self.graph.get_edge_data(entity_node, neighbor)
            if edge_data:
                relationship_types.add(edge_data.get('relationship', 'connected'))
            
            if neighbor_type == 'article':
                connected_articles.append({
                    'article_id': neighbor,
                    'title': neighbor_data.get('title', ''),
                    'category': neighbor_data.get('category', ''),
                    'relationship': edge_data.get('relationship', 'mentioned_in') if edge_data else 'mentioned_in'
                })
            elif neighbor_type in ['person', 'organization', 'location', 'event', 'topic']:
                connected_entities.append({
                    'entity_id': neighbor,
                    'name': neighbor_data.get('name', ''),
                    'type': neighbor_type,
                    'relationship': edge_data.get('relationship', 'related_to') if edge_data else 'related_to'
                })
        
        context = {
            'entity_id': entity_node,
            'entity_attributes': entity_data,
            'connected_articles': connected_articles,
            'connected_entities': connected_entities,
            'relationship_types': list(relationship_types),
            'total_connections': len(connected_articles) + len(connected_entities),
            'articles_count': len(connected_articles),
            'entities_count': len(connected_entities)
        }
        
        print(f"✅ Context retrieved: {len(connected_articles)} articles, {len(connected_entities)} entities")
        
        return context
    
    def _empty_traversal_result(self) -> Dict[str, Any]:
        """Return empty traversal result structure."""
        return {
            'start_entity': '',
            'start_nodes': [],
            'max_hops': 0,
            'related_entities': [],
            'related_articles': [],
            'traversal_paths': [],
            'traversal_time': 0.0,
            'entities_count': 0,
            'articles_count': 0,
            'paths_count': 0
        }
    
    def _update_traversal_stats(self, traversal_time: float, hops: int, entities_found: int, articles_found: int) -> None:
        """Update traversal statistics."""
        self.stats['total_traversals'] += 1
        self.stats['total_hops_explored'] += hops
        self.stats['entities_found'] += entities_found
        self.stats['articles_found'] += articles_found
        
        # Update average traversal time
        total_time = self.stats['average_traversal_time'] * (self.stats['total_traversals'] - 1) + traversal_time
        self.stats['average_traversal_time'] = total_time / self.stats['total_traversals']
    
    def get_traversal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive traversal statistics."""
        return {
            'graph_info': {
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0,
                'load_time': self.stats['graph_loaded_time']
            },
            'traversal_stats': {
                'total_traversals': self.stats['total_traversals'],
                'total_hops_explored': self.stats['total_hops_explored'],
                'total_entities_found': self.stats['entities_found'],
                'total_articles_found': self.stats['articles_found'],
                'average_traversal_time': self.stats['average_traversal_time']
            }
        }
    
    def print_statistics(self) -> None:
        """Print comprehensive traversal statistics."""
        stats = self.get_traversal_statistics()
        
        print(f"\n📊 GRAPH TRAVERSAL STATISTICS")
        print(f"=" * 50)
        
        graph_info = stats['graph_info']
        print(f"🕸️  Graph Information:")
        print(f"   Nodes: {graph_info['nodes']:,}")
        print(f"   Edges: {graph_info['edges']:,}")
        print(f"   Load time: {graph_info['load_time']:.2f}s")
        
        traversal_stats = stats['traversal_stats']
        if traversal_stats['total_traversals'] > 0:
            print(f"\n🔍 Traversal Statistics:")
            print(f"   Total traversals: {traversal_stats['total_traversals']}")
            print(f"   Total hops explored: {traversal_stats['total_hops_explored']}")
            print(f"   Entities found: {traversal_stats['total_entities_found']:,}")
            print(f"   Articles found: {traversal_stats['total_articles_found']:,}")
            print(f"   Average traversal time: {traversal_stats['average_traversal_time']:.2f}s")
        else:
            print(f"\n🔍 No traversals performed yet")


# Simple test function for single entity
def test_single_entity() -> None:
    """Test the GraphTraversal with a single entity to validate functionality."""
    print("🧪 TESTING SINGLE ENTITY TRAVERSAL")
    print("=" * 50)
    
    # Initialize graph traversal
    traversal = GraphTraversal()
    
    if traversal.graph is None:
        print("❌ Cannot test - no graph loaded")
        return
    
    # Test with one known entity
    test_entity = "Microsoft"
    
    print(f"📋 Testing with entity: '{test_entity}'\n")
    
    # Test 1: Find entity in graph
    print(f"🔍 Finding '{test_entity}' in graph...")
    nodes = traversal.find_entity_in_graph(test_entity)
    print(f"   Found {len(nodes)} matching nodes")
    if nodes:
        print(f"   Node IDs: {nodes}")
    
    # Test 2: Single entity traversal
    print(f"\n🕸️  Performing multi-hop traversal...")
    result = traversal.multi_hop_traversal(test_entity, max_hops=2)
    
    print(f"\n📋 TRAVERSAL RESULTS:")
    print(f"   Start entity: {result['start_entity']}")
    print(f"   Start nodes: {result['start_nodes']}")
    print(f"   Related entities: {result['entities_count']}")
    print(f"   Related articles: {result['articles_count']}")
    print(f"   Traversal paths: {result['paths_count']}")
    print(f"   Time: {result['traversal_time']:.2f}s")
    
    # Show sample results
    if result['related_entities']:
        print(f"\n🏷️  Sample related entities ({min(5, len(result['related_entities']))}):")
        for i, entity in enumerate(result['related_entities'][:5]):
            print(f"   {i+1}. {entity}")
    
    if result['related_articles']:
        print(f"\n📄 Sample related articles ({min(3, len(result['related_articles']))}):")
        for i, article in enumerate(result['related_articles'][:3]):
            print(f"   {i+1}. {article}")
    
    # Test 3: Entity context if we found results
    if result['related_entities']:
        print(f"\n📖 Getting context for first related entity...")
        sample_entity = result['related_entities'][0]
        context = traversal.get_entity_context(sample_entity)
        
        if 'error' not in context:
            print(f"   Entity: {sample_entity}")
            print(f"   Type: {context['entity_attributes'].get('type', 'unknown')}")
            print(f"   Name: {context['entity_attributes'].get('name', 'unknown')}")
            print(f"   Connected articles: {context['articles_count']}")
            print(f"   Connected entities: {context['entities_count']}")
        else:
            print(f"   ❌ Error: {context['error']}")
    
    # Print final statistics
    traversal.print_statistics()
    
    print(f"\n✅ Single entity testing complete!")
    
    return result


# Full test function for the GraphTraversal
def test_graph_traversal() -> None:
    """Test the GraphTraversal with known entities from the knowledge graph."""
    print("🧪 TESTING GRAPH TRAVERSAL")
    print("=" * 60)
    
    # Initialize graph traversal
    traversal = GraphTraversal()
    
    if traversal.graph is None:
        print("❌ Cannot test - no graph loaded")
        return
    
    # Test with known entities from the BBC news dataset
    test_entities = [
        "Microsoft",       # Technology company
        "Tim Henman",      # Tennis player
        "Google",          # Another tech company
        "tennis"           # Topic/sport
    ]
    
    print(f"📋 Testing with entities: {test_entities}\n")
    
    # Test 1: Find entities in graph
    print(f"{'='*15} TEST 1: Entity Finding {'='*15}")
    for entity in test_entities:
        nodes = traversal.find_entity_in_graph(entity)
        print(f"   '{entity}' → {len(nodes)} nodes found: {nodes[:3]}{'...' if len(nodes) > 3 else ''}")
    
    # Test 2: Single entity traversal
    print(f"\n{'='*15} TEST 2: Single Traversal {'='*15}")
    test_entity = test_entities[0]  # Use first entity
    
    result = traversal.multi_hop_traversal(test_entity, max_hops=2)
    
    print(f"Traversal from '{test_entity}':")
    print(f"   Related entities: {result['entities_count']}")
    print(f"   Related articles: {result['articles_count']}")
    print(f"   Traversal paths: {result['paths_count']}")
    print(f"   Time: {result['traversal_time']:.2f}s")
    
    if result['related_entities']:
        print(f"   Sample entities: {result['related_entities'][:5]}")
    
    # Test 3: Multi-entity relationship finding
    print(f"\n{'='*15} TEST 3: Multi-Entity Relations {'='*15}")
    
    multi_result = traversal.find_related_entities(test_entities[:2], max_hops=2)
    
    print(f"Related entities for {test_entities[:2]}:")
    print(f"   Total unique related: {multi_result['total_related_count']}")
    
    for query_entity, related in multi_result['entity_relationships'].items():
        print(f"   '{query_entity}' → {len(related)} related entities")
    
    # Test 4: Entity context
    print(f"\n{'='*15} TEST 4: Entity Context {'='*15}")
    
    if result['related_entities']:
        sample_entity = result['related_entities'][0]
        context = traversal.get_entity_context(sample_entity)
        
        if 'error' not in context:
            print(f"Context for '{sample_entity}':")
            print(f"   Type: {context['entity_attributes'].get('type', 'unknown')}")
            print(f"   Connected articles: {context['articles_count']}")
            print(f"   Connected entities: {context['entities_count']}")
            print(f"   Relationship types: {context['relationship_types']}")
        else:
            print(f"   Error getting context: {context['error']}")
    
    # Print final statistics
    traversal.print_statistics()
    
    print(f"\n✅ GraphTraversal testing complete!")


if __name__ == "__main__":
    # Run single entity test first
    result = test_single_entity()
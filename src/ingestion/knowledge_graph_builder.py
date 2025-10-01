"""
Knowledge Graph Builder Module

Constructs NetworkX knowledge graph from resolved entities.
Creates typed nodes and relationships for comprehensive analysis.
"""

import networkx as nx
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import json


class KnowledgeGraphBuilder:
    """
    Builds NetworkX knowledge graph from resolved entities and articles.
    
    Creates a comprehensive graph with typed nodes and multiple relationship types
    for entity analysis and navigation.
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        
        # Relationship type definitions
        self.relation_types = {
            'mentions': 'Article mentions entity',
            'co_occurrence': 'Entities mentioned in same article',
            'same_category': 'Articles in same BBC category',
            'person_organization': 'Person associated with organization',
            'person_location': 'Person associated with location',
            'organization_location': 'Organization located in location',
            'event_location': 'Event occurred at location',
            'event_person': 'Person involved in event',
            'event_organization': 'Organization involved in event'
        }
        
        # Node type definitions
        self.node_types = {
            'article': 'BBC News Article',
            'person': 'Named Person',
            'organization': 'Organization/Company',
            'location': 'Geographic Location',
            'event': 'Named Event',
            'topic': 'Topic/Subject',
            'category': 'BBC News Category'
        }
    
    def create_graph(self, resolved_entities: Dict[str, Any], processed_articles: List[Dict[str, Any]]) -> nx.Graph:
        """
        Create NetworkX graph from resolved entities and processed articles.
        
        Args:
            resolved_entities: Output from EntityResolver.resolve_entities()
            processed_articles: List of processed article data
            
        Returns:
            NetworkX graph with typed nodes and relationships
        """
        print("🔄 Building knowledge graph...")
        
        # Create new graph
        G = nx.Graph()
        
        # Add metadata to graph
        G.graph['created_by'] = 'KnowledgeGraphBuilder'
        G.graph['description'] = 'BBC News Knowledge Graph'
        G.graph['node_types'] = self.node_types
        G.graph['relation_types'] = self.relation_types
        
        # Get entity mappings
        entity_articles = resolved_entities.get('entity_articles_mapping', {})
        entities = resolved_entities.get('resolved_entities', {})
        
        # Create article lookup
        articles_dict = {article['article_id']: article for article in processed_articles}
        
        # Step 1: Add article nodes
        article_nodes_added = 0
        categories = set()
        
        for article in processed_articles:
            article_id = article['article_id']
            
            # Add article node
            G.add_node(
                f"article:{article_id}",
                type='article',
                title=article.get('generated_title', 'Unknown Title'),
                category=article.get('category', 'unknown'),
                text_length=len(article.get('text', '')),
                chunk_count=len(article.get('chunks', [])),
                original_text=article.get('text', ''),
                summary=article.get('text', '')[:200] + '...' if len(article.get('text', '')) > 200 else article.get('text', '')
            )
            article_nodes_added += 1
            
            # Track categories
            category = article.get('category', 'unknown')
            categories.add(category)
        
        # Step 2: Add category nodes
        category_nodes_added = 0
        for category in categories:
            G.add_node(
                f"category:{category}",
                type='category',
                name=category.replace('_', ' ').title(),
                category_id=category
            )
            category_nodes_added += 1
        
        # Step 3: Add entity nodes
        person_nodes_added = 0
        for person in entities.get('persons', []):
            node_id = f"person:{person['name']}"
            G.add_node(
                node_id,
                type='person',
                name=person['name'],
                roles=person.get('roles', []),
                article_count=person.get('article_count', 0)
            )
            person_nodes_added += 1
        
        org_nodes_added = 0
        for org in entities.get('organizations', []):
            node_id = f"organization:{org['name']}"
            G.add_node(
                node_id,
                type='organization',
                name=org['name'],
                types=org.get('types', []),
                article_count=org.get('article_count', 0)
            )
            org_nodes_added += 1
        
        location_nodes_added = 0
        for location in entities.get('locations', []):
            node_id = f"location:{location['name']}"
            G.add_node(
                node_id,
                type='location',
                name=location['name'],
                types=location.get('types', []),
                article_count=location.get('article_count', 0)
            )
            location_nodes_added += 1
        
        event_nodes_added = 0
        for event in entities.get('events', []):
            node_id = f"event:{event['name']}"
            G.add_node(
                node_id,
                type='event',
                name=event['name'],
                dates=event.get('dates', []),
                article_count=event.get('article_count', 0)
            )
            event_nodes_added += 1
        
        topic_nodes_added = 0
        for topic in entities.get('topics', []):
            node_id = f"topic:{topic}"
            # Count articles for this topic
            topic_article_count = len(entity_articles.get(node_id, []))
            G.add_node(
                node_id,
                type='topic',
                name=topic,
                article_count=topic_article_count
            )
            topic_nodes_added += 1
        
        # Step 4: Add relationships
        mention_edges_added = 0
        category_edges_added = 0
        
        # Add article-category relationships
        for article in processed_articles:
            article_id = article['article_id']
            category = article.get('category', 'unknown')
            
            G.add_edge(
                f"article:{article_id}",
                f"category:{category}",
                type='same_category',
                weight=1.0
            )
            category_edges_added += 1
        
        # Add entity-article mention relationships
        for entity_id, article_list in entity_articles.items():
            for article_id in article_list:
                if f"article:{article_id}" in G and entity_id in G:
                    G.add_edge(
                        entity_id,
                        f"article:{article_id}",
                        type='mentions',
                        weight=1.0
                    )
                    mention_edges_added += 1
        
        # Step 5: Add co-occurrence relationships
        cooccurrence_edges_added = 0
        
        # Group entities by articles they appear in
        article_entities = defaultdict(list)
        for entity_id, article_list in entity_articles.items():
            for article_id in article_list:
                article_entities[article_id].append(entity_id)
        
        # Add co-occurrence edges between entities in same articles
        for article_id, entity_list in article_entities.items():
            # Add edges between all pairs of entities in this article
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    entity1 = entity_list[i]
                    entity2 = entity_list[j]
                    
                    if entity1 in G and entity2 in G:
                        # Check if edge already exists, if so increase weight
                        if G.has_edge(entity1, entity2):
                            G[entity1][entity2]['weight'] += 1.0
                        else:
                            G.add_edge(
                                entity1,
                                entity2,
                                type='co_occurrence',
                                weight=1.0
                            )
                            cooccurrence_edges_added += 1
        
        # Step 6: Add semantic relationships (person-org, org-location, etc.)
        semantic_edges_added = 0
        
        # This would require more sophisticated entity relationship detection
        # For now, we'll add basic inferred relationships based on co-occurrence patterns
        
        # Find highly connected person-organization pairs
        for person_node in [n for n, d in G.nodes(data=True) if d['type'] == 'person']:
            person_articles = set(entity_articles.get(person_node, []))
            
            for org_node in [n for n, d in G.nodes(data=True) if d['type'] == 'organization']:
                org_articles = set(entity_articles.get(org_node, []))
                
                # If person and org appear together in multiple articles, add stronger relationship
                common_articles = person_articles.intersection(org_articles)
                if len(common_articles) >= 2:  # Threshold for semantic relationship
                    if not G.has_edge(person_node, org_node):
                        G.add_edge(
                            person_node,
                            org_node,
                            type='person_organization',
                            weight=float(len(common_articles))
                        )
                        semantic_edges_added += 1
        
        # Print statistics
        print(f"✅ Knowledge graph construction complete!")
        print(f"📊 Graph Statistics:")
        print(f"   Total nodes: {G.number_of_nodes()}")
        print(f"   Total edges: {G.number_of_edges()}")
        print(f"")
        print(f"   Nodes by type:")
        print(f"     Articles: {article_nodes_added}")
        print(f"     Categories: {category_nodes_added}")
        print(f"     Persons: {person_nodes_added}")
        print(f"     Organizations: {org_nodes_added}")
        print(f"     Locations: {location_nodes_added}")
        print(f"     Events: {event_nodes_added}")
        print(f"     Topics: {topic_nodes_added}")
        print(f"")
        print(f"   Edges by type:")
        print(f"     Mentions: {mention_edges_added}")
        print(f"     Category: {category_edges_added}")
        print(f"     Co-occurrence: {cooccurrence_edges_added}")
        print(f"     Semantic: {semantic_edges_added}")
        
        return G
    
    def get_graph_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Calculate comprehensive graph statistics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary with graph statistics and metrics
        """
        print("📊 Calculating graph statistics...")
        
        # Basic metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Node type distribution
        node_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Edge type distribution
        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Centrality measures (for connected components)
        degree_centrality = {}
        betweenness_centrality = {}
        closeness_centrality = {}
        
        if num_nodes > 0 and num_edges > 0:
            # Calculate for largest connected component if graph is not fully connected
            if nx.is_connected(G):
                largest_cc = G
            else:
                largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            
            if largest_cc.number_of_nodes() > 1:
                degree_centrality = nx.degree_centrality(largest_cc)
                if largest_cc.number_of_nodes() > 2:
                    betweenness_centrality = nx.betweenness_centrality(largest_cc)
                    closeness_centrality = nx.closeness_centrality(largest_cc)
        
        # Find most central nodes
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Connected components
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        largest_component_size = max(len(cc) for cc in connected_components) if connected_components else 0
        
        # Average degree
        if num_nodes > 0:
            avg_degree = sum(dict(G.degree()).values()) / num_nodes
        else:
            avg_degree = 0
        
        # Density
        density = nx.density(G) if num_nodes > 1 else 0
        
        statistics = {
            'basic_metrics': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_connected_components': num_components,
                'largest_component_size': largest_component_size,
                'average_degree': round(avg_degree, 2),
                'density': round(density, 4)
            },
            'node_type_distribution': node_types,
            'edge_type_distribution': edge_types,
            'centrality_analysis': {
                'top_degree_centrality': [(node, round(score, 4)) for node, score in top_degree],
                'top_betweenness_centrality': [(node, round(score, 4)) for node, score in top_betweenness]
            }
        }
        
        return statistics
    
    def save_graph(self, G: nx.Graph, filepath: str) -> None:
        """
        Save NetworkX graph to file.
        
        Args:
            G: NetworkX graph to save
            filepath: Path to save graph file
        """
        print(f"💾 Saving graph to {filepath}...")
        
        # Create a copy of the graph with serializable attributes
        G_copy = G.copy()
        
        # Convert list/dict attributes to strings for GraphML compatibility
        for node, attrs in G_copy.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = str(value)
        
        for u, v, attrs in G_copy.edges(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = str(value)
        
        # Try GraphML first, fallback to pickle if that fails
        try:
            nx.write_graphml(G_copy, filepath)
        except Exception as e:
            print(f"   ⚠️ GraphML save failed: {e}")
            # Save as pickle instead
            pickle_path = filepath.replace('.graphml', '.pkl')
            print(f"   💾 Saving as pickle: {pickle_path}")
            import pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(G, f)
            print(f"   ✅ Saved as pickle format")
        
        print(f"✅ Graph saved successfully!")
    
    def load_graph(self, filepath: str) -> nx.Graph:
        """
        Load NetworkX graph from file.
        
        Args:
            filepath: Path to graph file
            
        Returns:
            Loaded NetworkX graph
        """
        print(f"📂 Loading graph from {filepath}...")
        
        # Load GraphML
        G = nx.read_graphml(filepath)
        
        print(f"✅ Graph loaded successfully!")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        
        return G
    
    def find_entity_neighborhoods(self, G: nx.Graph, entity_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Find neighborhood of an entity in the graph.
        
        Args:
            G: NetworkX graph
            entity_id: ID of entity to analyze
            max_distance: Maximum distance for neighborhood
            
        Returns:
            Dictionary with neighborhood information
        """
        if entity_id not in G:
            return {'error': f'Entity {entity_id} not found in graph'}
        
        # Get ego graph (subgraph of neighbors)
        ego_graph = nx.ego_graph(G, entity_id, radius=max_distance)
        
        # Analyze neighborhood
        neighbors_by_distance = {}
        for distance in range(1, max_distance + 1):
            neighbors_at_distance = set()
            for node in ego_graph.nodes():
                if node != entity_id:
                    try:
                        shortest_path_length = nx.shortest_path_length(G, entity_id, node)
                        if shortest_path_length == distance:
                            neighbors_at_distance.add(node)
                    except nx.NetworkXNoPath:
                        continue
            neighbors_by_distance[distance] = list(neighbors_at_distance)
        
        # Get neighbor types
        neighbor_types = {}
        for node in ego_graph.nodes():
            if node != entity_id:
                node_type = G.nodes[node].get('type', 'unknown')
                if node_type not in neighbor_types:
                    neighbor_types[node_type] = []
                neighbor_types[node_type].append(node)
        
        return {
            'entity_id': entity_id,
            'neighborhood_size': ego_graph.number_of_nodes() - 1,  # Exclude the entity itself
            'neighborhood_edges': ego_graph.number_of_edges(),
            'neighbors_by_distance': neighbors_by_distance,
            'neighbors_by_type': neighbor_types,
            'ego_graph': ego_graph
        }


def main():
    """Demo function for knowledge graph builder."""
    print("🔄 Knowledge Graph Builder Demo")
    print("=" * 40)
    
    # Sample resolved entities
    sample_resolved = {
        'resolved_entities': {
            'persons': [
                {'name': 'Tim Cook', 'roles': ['CEO'], 'article_count': 2}
            ],
            'organizations': [
                {'name': 'Apple', 'types': ['Technology Company'], 'article_count': 2}
            ],
            'locations': [
                {'name': 'United States', 'types': ['Country'], 'article_count': 2}
            ],
            'events': [
                {'name': 'Product Launch', 'dates': ['2024-01-15'], 'article_count': 1}
            ],
            'topics': ['Technology', 'Business']
        },
        'entity_articles_mapping': {
            'person:Tim Cook': ['article_001', 'article_002'],
            'organization:Apple': ['article_001', 'article_002'],
            'location:United States': ['article_001', 'article_002'],
            'event:Product Launch': ['article_001'],
            'topic:Technology': ['article_001', 'article_002'],
            'topic:Business': ['article_002']
        }
    }
    
    # Sample processed articles
    sample_articles = [
        {
            'id': 'article_001',
            'title': 'Apple Announces New Product',
            'category': 'tech',
            'text': 'Apple CEO Tim Cook announced a new product launch...',
            'chunks': [{'text': 'chunk1'}, {'text': 'chunk2'}]
        },
        {
            'id': 'article_002',
            'title': 'Tech Industry Update',
            'category': 'business',
            'text': 'The technology industry continues to evolve...',
            'chunks': [{'text': 'chunk1'}]
        }
    ]
    
    builder = KnowledgeGraphBuilder()
    
    print("Building graph from sample data...")
    graph = builder.create_graph(sample_resolved, sample_articles)
    
    print("\nCalculating statistics...")
    stats = builder.get_graph_statistics(graph)
    
    print(f"\nGraph Analysis:")
    print(f"Basic metrics: {stats['basic_metrics']}")
    print(f"Node types: {stats['node_type_distribution']}")
    print(f"Edge types: {stats['edge_type_distribution']}")
    
    # Analyze entity neighborhood
    print(f"\nAnalyzing 'person:Tim Cook' neighborhood...")
    neighborhood = builder.find_entity_neighborhoods(graph, 'person:Tim Cook')
    print(f"Neighborhood size: {neighborhood['neighborhood_size']}")
    print(f"Neighbors by type: {neighborhood['neighbors_by_type']}")


if __name__ == "__main__":
    main()
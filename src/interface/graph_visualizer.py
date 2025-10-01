"""
Graph Visualization Component for KG-Enhanced RAG

Interactive knowledge graph visualization using Plotly for Streamlit integration.
Optimized for fast rendering with simplified responsive design.
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


class GraphVisualizer:
    """Interactive graph visualization component for KG-Enhanced RAG results."""
    
    def __init__(self, knowledge_graph_path: str = None, max_nodes: int = 50, max_hops: int = 2):
        """
        Initialize the graph visualizer.
        
        Args:
            knowledge_graph_path: Path to the knowledge graph pickle file
            max_nodes: Maximum number of nodes to display for performance
            max_hops: Maximum traversal depth from query entities
        """
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.knowledge_graph = None
        self.node_colors = {
            'article': '#1f77b4',      # Blue
            'person': '#2ca02c',       # Green
            'organization': '#ff7f0e', # Orange
            'location': '#d62728',     # Red
            'topic': '#9467bd',        # Purple
            'event': '#8c564b',        # Brown
            'query_entity': '#ffd700'  # Gold
        }
        self.node_shapes = {
            'article': 'circle',
            'person': 'circle',
            'organization': 'square',
            'location': 'triangle-up',
            'topic': 'diamond',
            'event': 'hexagon',
            'query_entity': 'star'
        }
        
        # Load knowledge graph if path provided
        if knowledge_graph_path:
            self.load_knowledge_graph(knowledge_graph_path)
    
    def load_knowledge_graph(self, graph_path: str) -> bool:
        """Load the knowledge graph from pickle file."""
        try:
            graph_file = Path(graph_path)
            if graph_file.exists():
                with open(graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                return True
            else:
                st.warning(f"Knowledge graph file not found: {graph_path}")
                return False
        except Exception as e:
            st.error(f"Error loading knowledge graph: {str(e)}")
            return False
    
    def extract_subgraph(self, query_entities: List[str], related_entities: List[Dict], 
                        articles: List[str]) -> Tuple[nx.Graph, Dict]:
        """
        Extract relevant subgraph for visualization.
        
        Args:
            query_entities: Entities from the original query
            related_entities: Entities discovered during KG traversal
            articles: Article IDs that were retrieved
            
        Returns:
            Tuple of (NetworkX graph, metadata dict)
        """
        if not self.knowledge_graph:
            return nx.Graph(), {}
        
        # Create subgraph
        subgraph = nx.Graph()
        metadata = {'query_entities': query_entities, 'node_types': {}}
        
        # Add query entities as priority nodes
        priority_nodes = set()
        for entity in query_entities:
            if entity in self.knowledge_graph.nodes():
                priority_nodes.add(entity)
                node_data = self.knowledge_graph.nodes[entity]
                entity_type = node_data.get('type', 'topic')
                subgraph.add_node(entity, **node_data, is_query_entity=True)
                metadata['node_types'][entity] = entity_type
            else:
                # Add entity even if not in main graph (create synthetic node)
                priority_nodes.add(entity)
                subgraph.add_node(entity, name=entity, type='query_entity', is_query_entity=True)
                metadata['node_types'][entity] = 'query_entity'
        
        # Add related entities (limited by max_nodes for performance)
        added_nodes = len(priority_nodes)
        for entity_info in related_entities:
            if added_nodes >= self.max_nodes:
                break
            
            # Handle both string and dictionary formats
            if isinstance(entity_info, dict):
                entity_name = entity_info.get('name', str(entity_info))
            else:
                entity_name = str(entity_info)
                
            if entity_name in self.knowledge_graph.nodes() and entity_name not in subgraph.nodes():
                node_data = self.knowledge_graph.nodes[entity_name]
                entity_type = node_data.get('type', 'topic')
                subgraph.add_node(entity_name, **node_data, is_query_entity=False)
                metadata['node_types'][entity_name] = entity_type
                added_nodes += 1
        
        # Add article nodes
        for article_id in articles[:10]:  # Limit to top 10 articles for clarity
            if added_nodes >= self.max_nodes:
                break
                
            if article_id in self.knowledge_graph.nodes():
                node_data = self.knowledge_graph.nodes[article_id]
                subgraph.add_node(article_id, **node_data, is_query_entity=False)
                metadata['node_types'][article_id] = 'article'
                added_nodes += 1
        
        # Add edges between nodes in subgraph
        for node1 in subgraph.nodes():
            for node2 in subgraph.nodes():
                if node1 != node2 and self.knowledge_graph.has_edge(node1, node2):
                    edge_data = self.knowledge_graph.edges[node1, node2]
                    subgraph.add_edge(node1, node2, **edge_data)
        
        return subgraph, metadata
    
    def create_visualization(self, subgraph: nx.Graph, metadata: Dict) -> go.Figure:
        """
        Create interactive Plotly visualization.
        
        Args:
            subgraph: NetworkX graph to visualize
            metadata: Additional metadata about the graph
            
        Returns:
            Plotly Figure object
        """
        if len(subgraph.nodes()) == 0:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No graph data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Knowledge Graph Visualization",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            return fig
        
        # Use spring layout for node positioning
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        except:
            # Fallback to random layout if spring layout fails
            pos = {node: (np.random.random(), np.random.random()) 
                   for node in subgraph.nodes()}
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_hover = []
        node_symbols = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            node_data = subgraph.nodes[node]
            node_type = metadata['node_types'].get(node, 'topic')
            is_query_entity = node_data.get('is_query_entity', False)
            
            # Color coding
            if is_query_entity:
                node_colors.append(self.node_colors['query_entity'])
                node_sizes.append(20)  # Larger for query entities
            else:
                node_colors.append(self.node_colors.get(node_type, self.node_colors['topic']))
                node_sizes.append(15)
            
            # Node symbols (simplified for performance)
            node_symbols.append('circle')  # Use circles for all for simplicity
            
            # Node labels and hover info
            display_name = node_data.get('name', node)[:30]  # Truncate long names
            node_text.append(display_name)
            
            # Hover information
            hover_info = f"<b>{display_name}</b><br>"
            hover_info += f"Type: {node_type.title()}<br>"
            hover_info += f"Connections: {subgraph.degree(node)}<br>"
            if is_query_entity:
                hover_info += "⭐ Query Entity<br>"
            
            # Add additional metadata if available
            if 'category' in node_data:
                hover_info += f"Category: {node_data['category']}<br>"
            
            node_hover.append(hover_info)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge information
            edge_data = subgraph.edges[edge[0], edge[1]]
            edge_type = edge_data.get('relationship', 'connected')
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Type: {edge_type}")
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes with click event support
        # Create individual customdata for each node (not a list of all nodes)
        node_customdata = [[str(node)] for node in subgraph.nodes()]
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=node_hover,
            showlegend=False,
            customdata=node_customdata,  # Store individual node IDs for click events
            name="nodes"
        ))
        
        # Layout configuration
        fig.update_layout(
            title={
                'text': "🕸️ Knowledge Graph Visualization",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ 
                dict(
                    text="Hover over nodes for details • Gold nodes are query entities",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='white'
        )
        
        return fig
    
    def render_in_streamlit(self, kg_enhanced_result: Dict, container=None) -> bool:
        """
        Render the graph visualization in Streamlit.
        
        Args:
            kg_enhanced_result: KG-Enhanced RAG result containing graph data
            container: Streamlit container to render in (optional)
            
        Returns:
            True if successfully rendered, False otherwise
        """
        if container is None:
            container = st
        
        # Check if we have the necessary data
        if not kg_enhanced_result:
            container.info("Graph visualization requires KG-Enhanced RAG results")
            return False
        
        # Try to extract graph data from different possible locations
        raw_result = None
        if 'kg_enhanced_raw' in kg_enhanced_result:
            raw_result = kg_enhanced_result['kg_enhanced_raw']
        elif 'kg_enhanced' in kg_enhanced_result:
            raw_result = kg_enhanced_result['kg_enhanced']
        else:
            # Fallback: use the result directly if it contains the expected fields
            raw_result = kg_enhanced_result
        
        if not raw_result:
            container.info("No KG-Enhanced RAG data available for visualization")
            return False
        
        # Get entities and related data from KG-Enhanced RAG structure
        query_entities = []
        related_entities = []
        articles = []
        
        # Extract from KG-Enhanced RAG explanation structure
        if 'explanation' in raw_result and 'steps' in raw_result['explanation']:
            for step in raw_result['explanation']['steps']:
                if 'entities_found' in step.get('details', {}):
                    query_entities = step['details']['entities_found']
                elif 'related_entities' in step.get('details', {}):
                    related_entities = step['details']['related_entities']
        
        # Extract from graph_enhancement metadata
        if 'graph_enhancement' in raw_result:
            graph_enh = raw_result['graph_enhancement']
            # We know entities exist if counts are > 0
            
        # Extract from sources
        if 'sources' in raw_result:
            articles = [source.get('article_id', source.get('id', f"article_{i}")) 
                       for i, source in enumerate(raw_result['sources'])]
        
        # If still no entities but we have articles, extract from knowledge graph
        if articles and not query_entities and not related_entities and self.knowledge_graph:
            # Get entities connected to the articles from the knowledge graph
            article_entities = []
            entity_names = set()
            
            for article in articles[:10]:  # Limit to first 10 articles
                if article in self.knowledge_graph.nodes():
                    # Get neighbors (connected entities) of this article
                    neighbors = list(self.knowledge_graph.neighbors(article))
                    for neighbor in neighbors[:5]:  # Get up to 5 entities per article
                        if neighbor not in entity_names:  # Avoid duplicates
                            article_entities.append(neighbor)
                            entity_names.add(neighbor)
            
            # Use discovered entities
            if article_entities:
                query_entities = article_entities[:4]  # First 4 as query entities
                related_entities = [{'name': entity} for entity in article_entities[4:20]]  # Rest as related
                container.info(f"Extracted {len(query_entities)} query entities and {len(related_entities)} related entities from article connections.")
        
        # Create expandable graph section
        with container.expander("🕸️ **Knowledge Graph Visualization**", expanded=False):
            if not self.knowledge_graph:
                # Try to load knowledge graph
                graph_path = "data/processed/knowledge_graph.pkl"
                if not self.load_knowledge_graph(graph_path):
                    st.error("Knowledge graph not available for visualization")
                    return False
            
            # Show loading message
            with st.spinner("Generating knowledge graph visualization..."):
                try:
                    # If still no entities after extraction, create a meaningful demo graph
                    if not query_entities and not related_entities:
                        if self.knowledge_graph and len(self.knowledge_graph.nodes()) > 0:
                            # Get a diverse sample of nodes by type
                            all_nodes = list(self.knowledge_graph.nodes(data=True))
                            
                            # Separate nodes by type
                            articles_nodes = [n[0] for n in all_nodes if n[1].get('type') == 'article'][:5]
                            person_nodes = [n[0] for n in all_nodes if n[1].get('type') == 'person'][:3]
                            org_nodes = [n[0] for n in all_nodes if n[1].get('type') == 'organization'][:3]
                            topic_nodes = [n[0] for n in all_nodes if n[1].get('type') == 'topic'][:4]
                            
                            # Create a mixed demo graph
                            query_entities = person_nodes[:2] + org_nodes[:1] if person_nodes or org_nodes else list(self.knowledge_graph.nodes())[:3]
                            related_entities = [{'name': node} for node in (topic_nodes + org_nodes[1:] + person_nodes[2:])]
                            articles = articles_nodes if articles_nodes else articles
                            
                            container.info("Created demonstration graph with sample entities and relationships.")
                    
                    # Extract subgraph
                    subgraph, metadata = self.extract_subgraph(
                        query_entities, related_entities, articles
                    )
                    
                    # Create visualization
                    fig = self.create_visualization(subgraph, metadata)
                    
                    # Display the graph
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show graph statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes", len(subgraph.nodes()))
                    with col2:
                        st.metric("Connections", len(subgraph.edges()))
                    with col3:
                        st.metric("Query Entities", len(query_entities))
                    
                    # Legend
                    st.markdown("""
                    **Legend:** 
                    🟡 Query Entities • 🔵 Articles • 🟢 People • 🟠 Organizations • 🔴 Locations • 🟣 Topics
                    """)
                    
                    return True
                    
                except Exception as e:
                    st.error(f"Error generating graph visualization: {str(e)}")
                    return False


def create_graph_visualizer(graph_path: str = "data/processed/knowledge_graph.pkl") -> GraphVisualizer:
    """
    Factory function to create a GraphVisualizer instance.
    
    Args:
        graph_path: Path to the knowledge graph pickle file
        
    Returns:
        GraphVisualizer instance
    """
    return GraphVisualizer(knowledge_graph_path=graph_path)
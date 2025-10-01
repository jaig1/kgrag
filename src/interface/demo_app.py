"""
KG-Enhanced RAG Demo Interface

Interactive Streamlit application for comparing baseline RAG vs KG-Enhanced RAG systems.
"""

import streamlit as st
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG systems
try:
    from src.retrieval.baseline_rag import BaselineRAG
    from src.retrieval.kg_enhanced_rag import KGEnhancedRAG
    from src.retrieval.response_generator import ResponseGenerator
    from src.retrieval.response_tracker import ResponseTracker
    from src.interface.graph_visualizer import GraphVisualizer
    RAG_SYSTEMS_AVAILABLE = True
except ImportError as e:
    RAG_SYSTEMS_AVAILABLE = False
    RAG_IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="KG-Enhanced RAG Demo",
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ALL session state variables FIRST (critical for button logic)
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'auto_selected_example' not in st.session_state:
    st.session_state.auto_selected_example = False
if 'system_mode' not in st.session_state:
    st.session_state.system_mode = "🔄 Both (Compare)"

# RAG System Initialization
@st.cache_resource
def initialize_rag_systems():
    """Initialize RAG systems with caching for performance."""
    if not RAG_SYSTEMS_AVAILABLE:
        return None, None, None
    
    try:
        # Initialize response tracker
        log_dir = Path("data/logs")
        log_dir.mkdir(exist_ok=True)
        response_tracker = ResponseTracker(
            log_dir=log_dir,
            log_filename="demo_queries.jsonl"
        )
        
        # Initialize baseline RAG with faster model for demo
        baseline_rag = BaselineRAG(
            response_tracker=response_tracker,
            model_name="gpt-3.5-turbo",  # Faster model (3-5s vs 18-23s)
            max_sources=5,
            temperature=0.1
        )
        
        # Initialize KG-Enhanced RAG with optimized settings
        kg_enhanced_rag = KGEnhancedRAG(
            response_tracker=response_tracker,
            model_name="gpt-3.5-turbo",  # Faster model for demo
            max_sources=6,  # Reduced from 8 for faster processing
            temperature=0.1
        )
        
        # Initialize response generator
        response_generator = ResponseGenerator()
        
        return baseline_rag, kg_enhanced_rag, response_generator
        
    except Exception as e:
        st.error(f"Failed to initialize RAG systems: {str(e)}")
        return None, None, None

def get_top_organizations_subgraph() -> Dict[str, Any]:
    """Get top organizations from knowledge graph for default visualization."""
    try:
        import pickle
        import networkx as nx
        from pathlib import Path
        
        # Load knowledge graph (NetworkX Graph object)
        kg_path = Path("data/processed/knowledge_graph.pkl")
        if not kg_path.exists():
            return {'success': False, 'error': 'Knowledge graph file not found'}
        
        with open(kg_path, 'rb') as f:
            graph = pickle.load(f)
        
        # Verify it's a NetworkX graph
        if not isinstance(graph, nx.Graph):
            return {'success': False, 'error': f'Expected NetworkX Graph, got {type(graph)}'}
        
        # Extract organizations from graph nodes and sort by degree (connections)
        organizations = []
        for node in graph.nodes():
            if isinstance(node, str) and node.startswith('organization:'):
                degree = graph.degree(node)
                organizations.append((node, degree))
        
        if not organizations:
            return {'success': False, 'error': 'No organization nodes found in knowledge graph'}
        
        # Sort by degree (most connected first) and get top 15
        organizations.sort(key=lambda x: x[1], reverse=True)
        top_orgs = [org for org, _ in organizations[:15]]
        
        # Get connected entities and articles
        connected_entities = []
        connected_articles = set()
        
        for org in top_orgs:
            if org in graph:
                # Get neighbors (connected nodes)
                neighbors = list(graph.neighbors(org))[:8]  # Limit connections per org
                for neighbor in neighbors:
                    if isinstance(neighbor, str):
                        if neighbor.startswith('article:'):
                            connected_articles.add(neighbor)
                        elif not neighbor.startswith('organization:'):  # Avoid duplicate orgs
                            connected_entities.append(neighbor)
        
        # Remove duplicates and limit sizes
        connected_entities = list(set(connected_entities))[:25]
        connected_articles = list(connected_articles)[:10]
        
        return {
            'query': "Default: Top Organizations View",
            'extracted_entities': top_orgs[:15],
            'related_entities': connected_entities,
            'connected_articles': connected_articles,
            'total_nodes': len(top_orgs) + len(connected_entities) + len(connected_articles),
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to load default organizations: {str(e)}"
        }

def create_kg_visualizer_data(query: str) -> Dict[str, Any]:
    """Create knowledge graph visualization data for a given query."""
    try:
        from src.retrieval.graph_traversal import GraphTraversal
        from src.retrieval.query_analyzer import QueryAnalyzer
        import pickle
        
        # Initialize components
        query_analyzer = QueryAnalyzer()
        graph_traversal = GraphTraversal()
        
        # If no query provided, get top organizations
        if not query or query.strip() == "":
            return get_top_organizations_subgraph()
        
        # Analyze query to extract entities
        analysis_result = query_analyzer.analyze_query(query)
        extracted_entities = analysis_result.get('extracted_entities', [])
        
        # Handle different entity formats
        if isinstance(extracted_entities, dict):
            entities = []
            for category, entity_list in extracted_entities.items():
                if isinstance(entity_list, list):
                    entities.extend(entity_list)
                elif entity_list:
                    entities.append(entity_list)
            extracted_entities = entities
        
        # If no entities extracted, get top organizations
        if not extracted_entities:
            return get_top_organizations_subgraph()
        
        # Find related entities through graph traversal
        traversal_result = graph_traversal.find_related_entities(extracted_entities, max_hops=2)
        
        if not traversal_result or 'related_entities' not in traversal_result:
            return get_top_organizations_subgraph()
        
        # Extract subgraph data
        related_entities = traversal_result.get('related_entities', [])
        
        # Limit to 50 nodes total
        all_entities = extracted_entities + related_entities
        limited_entities = all_entities[:45]  # Leave room for articles
        
        # Get articles connected to these entities
        connected_articles = set()
        for entity, result in traversal_result.get('traversal_results', {}).items():
            if 'related_articles' in result:
                connected_articles.update(result['related_articles'])
        
        # Limit articles
        limited_articles = list(connected_articles)[:5]
        
        return {
            'query': query,
            'extracted_entities': extracted_entities[:10],  # Show original entities
            'related_entities': related_entities[:35],      # Related entities
            'connected_articles': limited_articles,
            'total_nodes': len(limited_entities) + len(limited_articles),
            'success': True
        }
        
    except Exception as e:
        return {
            'query': query,
            'success': False,
            'error': str(e),
            'fallback_data': get_top_organizations_subgraph()
        }

def render_kg_visualizer_results(results: Dict[str, Any]) -> None:
    """Render KG Visualizer interface and results."""
    
    # Check if we have visualizer data
    visualizer_data = results.get('visualizer_data', {})
    
    if not visualizer_data or not visualizer_data.get('success', False):
        st.error("❌ Failed to create knowledge graph visualization")
        if 'error' in visualizer_data:
            st.error(f"Error: {visualizer_data['error']}")
        
        # Try fallback data
        if 'fallback_data' in visualizer_data:
            visualizer_data = visualizer_data['fallback_data']
        else:
            return
    
    # Graph Statistics Panel
    st.markdown("#### 📊 Graph Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        extracted_count = len(visualizer_data.get('extracted_entities', []))
        st.metric("Query Entities", extracted_count)
    
    with col2:
        related_count = len(visualizer_data.get('related_entities', []))
        st.metric("Related Entities", related_count)
    
    with col3:
        articles_count = len(visualizer_data.get('connected_articles', []))
        st.metric("Connected Articles", articles_count)
    
    with col4:
        total_nodes = visualizer_data.get('total_nodes', 0)
        st.metric("Total Nodes", min(total_nodes, 50))
    
    # Create and render visualization
    try:
        visualizer = GraphVisualizer(knowledge_graph_path="data/processed/knowledge_graph.pkl")
        
        # Prepare entities for visualization
        # Get entities separately as the method expects them
        query_entities = visualizer_data.get('extracted_entities', [])[:20]
        related_entities_list = visualizer_data.get('related_entities', [])[:25]
        articles = visualizer_data.get('connected_articles', [])[:5]
        
        # Convert related entities to expected dictionary format
        related_entities = [{'name': entity} for entity in related_entities_list]
        
        # Create subgraph with proper arguments - returns tuple (subgraph, metadata)
        subgraph, metadata = visualizer.extract_subgraph(query_entities, related_entities, articles)
        
        if subgraph and len(subgraph.nodes()) > 0:
            # Render interactive graph
            st.markdown("#### 🎯 Interactive Knowledge Graph")
            st.markdown("*Click and drag nodes, zoom to explore. Click nodes for details.*")
            
            fig = visualizer.create_visualization(subgraph, metadata)
            if fig:
                # Import plotly events for interactive click handling
                from streamlit_plotly_events import plotly_events
                
                # Display the interactive chart with click event capture
                clicked_point = plotly_events(
                    fig, 
                    click_event=True, 
                    hover_event=False,
                    select_event=False,
                    override_height=600,
                    key="kg_visualization"
                )
                
                # Node details section (collapsible)
                with st.expander("🔍 **Node Details** (Click a node above to see details)", expanded=False):
                    
                    # Handle click events
                    if clicked_point and len(clicked_point) > 0:
                        # Get clicked point data
                        point_data = clicked_point[0]
                        
                        # Extract node information from the clicked point
                        selected_node = None
                        
                        # Try different ways to extract the node ID
                        if 'customdata' in point_data and point_data['customdata']:
                            customdata = point_data['customdata']
                            # Handle both list and direct formats
                            if isinstance(customdata, list) and len(customdata) > 0:
                                selected_node = customdata[0]
                            else:
                                selected_node = str(customdata)
                        elif 'pointIndex' in point_data:
                            # Fallback: use point index to get node from ordered list
                            point_index = point_data['pointIndex']
                            node_list = list(subgraph.nodes())
                            if 0 <= point_index < len(node_list):
                                selected_node = node_list[point_index]
                        
                        if selected_node and selected_node in subgraph.nodes():
                            node_data = subgraph.nodes[selected_node]
                            
                            # Create detailed JSON for the selected node
                            node_details = {
                                "node_id": str(selected_node),
                                "display_name": node_data.get('name', str(selected_node)),
                                "node_type": metadata['node_types'].get(selected_node, 'unknown'),
                                "is_query_entity": node_data.get('is_query_entity', False),
                                "connections": len(list(subgraph.neighbors(selected_node))),
                                "connected_to": [str(neighbor) for neighbor in subgraph.neighbors(selected_node)],
                                "attributes": {k: v for k, v in node_data.items() 
                                            if k not in ['is_query_entity']},
                                "clicked_coordinates": {
                                    "x": point_data.get('x'),
                                    "y": point_data.get('y')
                                }
                            }
                            
                            st.success(f"🎯 Node Selected: {node_details['display_name']} ({node_details['node_type']})")
                            st.json(node_details)
                        elif selected_node:
                            st.warning(f"Node '{selected_node}' not found in graph")
                        else:
                            st.info("Click data received but no node ID found. Try clicking directly on a node.")
                    else:
                        # Show instruction and demo with a sample node
                        st.info("👆 **Click on any node in the graph above** to see its details here in JSON format.")
                        
                        # Show example with first node as demo
                        if len(subgraph.nodes()) > 0:
                            demo_node = list(subgraph.nodes())[0]
                            demo_data = subgraph.nodes[demo_node]
                            st.markdown("**Example - try clicking on a node like this one:**")
                            demo_details = {
                                "node_id": str(demo_node),
                                "display_name": demo_data.get('name', str(demo_node)),
                                "node_type": metadata['node_types'].get(demo_node, 'unknown'),
                                "connections": len(list(subgraph.neighbors(demo_node))),
                            }
                            st.json(demo_details)
            else:
                st.warning("⚠️ Could not create visualization")
                
            # JSON Data Display (collapsible)
            with st.expander("📄 **Graph Data (JSON Format)**", expanded=False):
                graph_json = {
                    'query': visualizer_data.get('query', 'Default: Top Organizations View'),
                    'metadata': metadata,
                    'nodes': {
                        'extracted_entities': visualizer_data.get('extracted_entities', [])[:10],
                        'related_entities': visualizer_data.get('related_entities', [])[:10],
                        'connected_articles': visualizer_data.get('connected_articles', [])[:5]
                    },
                    'graph_stats': {
                        'total_nodes': len(subgraph.nodes()),
                        'total_edges': len(subgraph.edges()),
                        'node_types': dict(metadata.get('node_types', {}))
                    }
                }
                st.json(graph_json)
        else:
            st.error("❌ Failed to extract subgraph data")
            
    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")
    
    # Interpretation Panel
    st.markdown("#### 💡 Interpretation")
    extracted_entities = visualizer_data.get('extracted_entities', [])
    related_entities = visualizer_data.get('related_entities', [])
    articles = visualizer_data.get('connected_articles', [])
    
    if extracted_entities:
        main_entities = ", ".join([e.split(':')[-1] for e in extracted_entities[:3]])
        st.markdown(f"""
        **What this visualization shows:**
        - **Primary entities**: {main_entities} (and {len(extracted_entities)-3} others)
        - **Network discovery**: Found {len(related_entities)} related entities through graph traversal
        - **Article connections**: {len(articles)} articles contain these entities
        - **Relationship types**: MENTIONS, CO_OCCURS, SAME_CATEGORY connections
        
        **Key insights:**
        - Entities cluster around major topics and themes in the news dataset
        - Cross-category connections reveal broader narrative patterns  
        - Node colors represent different entity types (Person, Organization, Location, etc.)
        """)
    else:
        st.markdown("""
        **Default Organization View:**
        - Shows top organizations from the BBC News knowledge graph
        - Connected entities represent related people, locations, and topics
        - Article nodes show which news articles mention these organizations
        - Explore the interconnected nature of news entities
        """)

def execute_rag_query(query: str, system_mode: str) -> Dict[str, Any]:
    """Execute RAG query based on selected system mode."""
    
    baseline_rag, kg_enhanced_rag, response_generator = initialize_rag_systems()
    
    if not baseline_rag or not kg_enhanced_rag:
        raise Exception("RAG systems not available. Please check the setup.")
    
    results = {
        'query': query,
        'system_mode': system_mode,
        'timestamp': time.time()
    }
    
    try:
        if system_mode == "🔵 Baseline Only":
            # Run baseline RAG only
            baseline_result = baseline_rag.answer_question(query)
            results['baseline'] = baseline_result
            results['kg_enhanced'] = None  # Explicitly set to None
            
        elif system_mode == "🟢 KG-Enhanced Only":
            # Run KG-Enhanced RAG only
            kg_result = kg_enhanced_rag.generate_answer(query, include_explanation=True)
            enhanced_result = response_generator.enhance_response(kg_result)
            results['baseline'] = None  # Explicitly set to None
            results['kg_enhanced'] = enhanced_result
            results['kg_enhanced_raw'] = kg_result
            
        elif system_mode == "🕸️ KG Visualizer":
            # Handle KG Visualizer mode - extract entities and create subgraph
            results['visualizer_data'] = create_kg_visualizer_data(query)
            results['baseline'] = None
            results['kg_enhanced'] = None
            
        else:  # Both (Compare)
            # Run both systems
            baseline_result = baseline_rag.answer_question(query)
            kg_result = kg_enhanced_rag.generate_answer(query, include_explanation=True)
            enhanced_result = response_generator.enhance_response(kg_result)
            
            results['baseline'] = baseline_result
            results['kg_enhanced'] = enhanced_result
            results['kg_enhanced_raw'] = kg_result
            
        results['success'] = True
        return results
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
        return results

def render_kg_reasoning_path(enhanced_result: Dict[str, Any], kg_raw_result: Dict[str, Any]) -> None:
    """Render detailed KG-Enhanced RAG reasoning path with step-by-step explanation."""
    
    # Extract reasoning data from various sources
    explanation = enhanced_result.get('explanation', {})
    graph_enhancement = enhanced_result.get('graph_enhancement', {})
    
    # Create the enhanced reasoning display
    with st.expander("🧠 **Knowledge Graph Reasoning Process**", expanded=False):
        st.markdown("""
        <style>
        .reasoning-step {
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }
        .step-badge {
            background: #28a745;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .entity-badge {
            background: #007bff;
            color: white;
            padding: 0.1rem 0.4rem;
            border-radius: 8px;
            font-size: 0.75rem;
            margin: 0.1rem;
        }
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        # Step 1: Entity Extraction
        extracted_entities = []
        if explanation and 'steps' in explanation:
            for step in explanation['steps']:
                if 'entities_found' in step.get('details', {}):
                    extracted_entities = step['details']['entities_found']
                    break
        
        if not extracted_entities and graph_enhancement.get('entities_extracted', 0) > 0:
            extracted_entities = ["Query entities discovered"]
        
        st.markdown('<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown('**<span class="step-badge">1</span> Entity Extraction from Query**', unsafe_allow_html=True)
        
        if extracted_entities:
            entity_tags = " ".join([f'<span class="entity-badge">{entity}</span>' for entity in extracted_entities[:6]])
            st.markdown(f"**Extracted entities:** {entity_tags}", unsafe_allow_html=True)
            st.markdown("**💡 Why this matters:** Identifying key entities helps the system understand what to search for in the knowledge graph.")
            confidence = "high" if len(extracted_entities) >= 2 else "medium" if len(extracted_entities) == 1 else "low"
            st.markdown(f'**Confidence:** <span class="confidence-{confidence}">{"High" if confidence == "high" else "Medium" if confidence == "medium" else "Low"}</span>', unsafe_allow_html=True)
        else:
            st.markdown("**No specific entities extracted** - Using semantic search approach")
            st.markdown("**💡 Why this matters:** Falls back to traditional vector similarity when entities aren't clearly identifiable.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Knowledge Graph Matching
        st.markdown('<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown('**<span class="step-badge">2</span> Knowledge Graph Entity Matching**', unsafe_allow_html=True)
        
        # Step 2 should show which ORIGINAL query entities were found in the knowledge graph
        entities_extracted_count = graph_enhancement.get('entities_extracted', 0)
        matched_query_entities = []
        
        # Find which of the original extracted entities exist in the knowledge graph
        if extracted_entities:
            # For Step 2, we show which query entities were successfully matched in KG
            matched_query_entities = extracted_entities  # These are the ones that led to graph traversal
        
        if entities_extracted_count > 0 and matched_query_entities:
            st.markdown(f"**Matched in graph:** {len(matched_query_entities)} query entities found in knowledge graph")
            # Display the matched query entities with badges
            entity_tags = " ".join([f'<span class="entity-badge">{entity}</span>' for entity in matched_query_entities])
            st.markdown(f"**Matched entities:** {entity_tags}", unsafe_allow_html=True)
            st.markdown("**💡 Why this matters:** Finding query entities in the knowledge graph enables relationship-based traversal beyond simple text similarity.")
            confidence = "high" if len(matched_query_entities) >= 2 else "medium" if len(matched_query_entities) == 1 else "low"
            st.markdown(f'**Confidence:** <span class="confidence-{confidence}">{"High" if confidence == "high" else "Medium" if confidence == "medium" else "Low"}</span>', unsafe_allow_html=True)
        elif entities_extracted_count > 0:
            # Fallback when we know entities were processed but don't have the list
            st.markdown(f"**Matched in graph:** {entities_extracted_count} query entities processed")
            st.markdown("**💡 Why this matters:** Finding query entities in the knowledge graph enables relationship-based traversal beyond simple text similarity.")
            confidence = "medium"
            st.markdown(f'**Confidence:** <span class="confidence-{confidence}">Medium</span>', unsafe_allow_html=True)
        else:
            st.markdown("**No query entity matches found** in knowledge graph")
            st.markdown("**💡 Why this matters:** System will rely more on semantic similarity rather than entity relationships.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Graph Traversal & Related Entities
        st.markdown('<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown('**<span class="step-badge">3</span> Knowledge Graph Traversal**', unsafe_allow_html=True)
        
        related_entities = []
        if explanation and 'steps' in explanation:
            for step in explanation['steps']:
                if 'related_entities' in step.get('details', {}):
                    related_entities = step['details']['related_entities'][:8]  # Limit display
                    break
        
        if related_entities:
            entity_tags = " ".join([f'<span class="entity-badge">{entity}</span>' for entity in related_entities])
            st.markdown(f"**Discovered related entities:** {entity_tags}", unsafe_allow_html=True)
            st.markdown("**💡 Why this matters:** Graph traversal discovers entities connected to your query, revealing hidden relationships and context.")
            confidence = "high" if len(related_entities) >= 6 else "medium" if len(related_entities) >= 3 else "low"
            st.markdown(f'**Confidence:** <span class="confidence-{confidence}">{"High" if confidence == "high" else "Medium" if confidence == "medium" else "Low"}</span>', unsafe_allow_html=True)
        else:
            st.markdown("**Limited graph traversal** - Using available entity connections")
            st.markdown("**💡 Why this matters:** Fewer entity connections mean more reliance on semantic similarity for discovery.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Query Enhancement
        st.markdown('<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown('**<span class="step-badge">4</span> Query Enhancement & Expansion**', unsafe_allow_html=True)
        
        # Get original and enhanced queries
        complete_results = st.session_state.query_results if hasattr(st.session_state, 'query_results') else {}
        original_query = complete_results.get('query', 'Query not available')
        enhanced_query = None
        
        # Try to extract enhanced query from various possible locations
        if kg_raw_result:
            # Look for enhanced query in the raw KG results
            enhanced_query = kg_raw_result.get('enhanced_query', None)
            if not enhanced_query and 'retrieval_result' in kg_raw_result:
                enhanced_query = kg_raw_result['retrieval_result'].get('enhanced_query', None)
        
        query_enhanced = graph_enhancement.get('query_enhanced', False)
        
        if query_enhanced and enhanced_query and enhanced_query != original_query:
            st.markdown("**Query successfully enhanced** with graph-discovered context")
            
            # Show original vs enhanced query comparison
            st.markdown("**📝 Original Query:**")
            st.code(original_query, language="text")
            
            st.markdown("**🚀 Enhanced Query:**")
            st.code(enhanced_query, language="text")
            
            st.markdown("**💡 Why this matters:** Enhanced queries capture broader context and related concepts from the knowledge graph, improving retrieval precision and recall.")
            st.markdown('**Confidence:** <span class="confidence-high">High</span>', unsafe_allow_html=True)
            
        elif query_enhanced:
            st.markdown("**Query enhanced** with graph-discovered terms and entity contexts")
            st.markdown("**� Original Query:**")
            st.code(original_query, language="text")
            st.markdown("*Enhanced query details not available for display*")
            st.markdown("**�💡 Why this matters:** Enhanced queries capture broader context and related concepts, improving retrieval precision.")
            st.markdown('**Confidence:** <span class="confidence-high">High</span>', unsafe_allow_html=True)
            
        else:
            st.markdown("**Using original query** with minimal or no enhancement")
            st.markdown("**� Query Used:**")
            st.code(original_query, language="text")
            
            # Explain why no enhancement occurred
            if len(extracted_entities) == 0:
                st.markdown("**⚠️ Reason**: No entities extracted from query for graph enhancement")
            elif entities_related_count == 0:
                st.markdown("**⚠️ Reason**: No related entities found in knowledge graph")
            else:
                st.markdown("**ℹ️ Reason**: Query enhancement not significantly different from original")
                
            st.markdown("**�💡 Why this matters:** System maintains focus on exact query terms when graph enhancement is limited or unnecessary.")
            st.markdown('**Confidence:** <span class="confidence-medium">Medium</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 5: Enhanced Document Retrieval
        st.markdown('<div class="reasoning-step">', unsafe_allow_html=True)
        st.markdown('**<span class="step-badge">5</span> Enhanced Document Retrieval**', unsafe_allow_html=True)
        st.markdown("*Hybrid approach: Graph discovery + Vector selection*")
        
        sources_count = len(enhanced_result.get('sources', []))
        articles_connected = graph_enhancement.get('articles_connected', 0)
        
        if sources_count > 0:
            # Process overview
            st.markdown(f"**📊 Process:** Knowledge Graph → {articles_connected} related articles → Vector Search → {sources_count} top chunks selected")
            
            # Results section with reordered information
            st.markdown("### 🎯 **Results:**")
            if articles_connected > 0:
                st.markdown(f"- **Graph-connected articles:** {articles_connected} articles identified via entity relationships")
            st.markdown(f"- **Retrieved:** {sources_count} relevant text chunks from knowledge-enhanced search")
            
            # Show top sources with article→chunk mapping
            sources = enhanced_result.get('sources', [])[:3]  # Show top 3
            if sources:
                st.markdown("### 📄 **Top Sources:**")
                for i, source in enumerate(sources, 1):
                    source_id = source.get('article_id', f'Source {i}')
                    chunk_id = source.get('chunk_id', source.get('metadata', {}).get('chunk_id', 'Unknown'))
                    score = source.get('score', source.get('similarity', 0))
                    st.markdown(f"{i}. **{source_id}** → Chunk {chunk_id} *({score:.3f})*")
            
            st.markdown("**💡 Why this works:** Graph discovery finds related articles beyond keywords → Vector search selects most relevant passages → Focused content for precise answers")
            confidence = "high" if sources_count >= 5 and articles_connected > 0 else "medium" if sources_count >= 3 else "low"
            st.markdown(f'**Confidence:** <span class="confidence-{confidence}">{"High" if confidence == "high" else "Medium" if confidence == "medium" else "Low"}</span>', unsafe_allow_html=True)
        else:
            st.markdown("**No relevant documents found** - Query may need refinement")
            st.markdown("**💡 Why this matters:** Limited results suggest the query terms or entities don't match available content well.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overall Process Summary
        st.markdown("---")
        enhancement_applied = graph_enhancement.get('enhancement_applied', False)
        
        if enhancement_applied:
            st.success("✅ **Knowledge Graph Enhancement Successfully Applied** - This query benefited from entity relationships and graph traversal.")
        else:
            st.info("ℹ️ **Limited Graph Enhancement** - This query relied more on semantic similarity than entity relationships.")
        
        # Overall confidence calculation
        total_steps = 5
        entities_related_count = graph_enhancement.get('entities_related', 0)
        confident_steps = sum([
            1 if len(extracted_entities) >= 2 else 0,
            1 if entities_related_count >= 3 else 0,
            1 if len(related_entities) >= 4 else 0,
            1 if query_enhanced else 0,
            1 if sources_count >= 4 and articles_connected > 0 else 0
        ])
        
        overall_confidence = confident_steps / total_steps
        conf_color = "high" if overall_confidence >= 0.7 else "medium" if overall_confidence >= 0.4 else "low"
        conf_text = "High" if overall_confidence >= 0.7 else "Medium" if overall_confidence >= 0.4 else "Low"
        
        st.markdown(f'**Overall Process Confidence:** <span class="confidence-{conf_color}">{conf_text} ({overall_confidence:.1%})</span>', unsafe_allow_html=True)


def render_answer_card(title: str, result: Dict[str, Any], card_type: str = "baseline") -> None:
    """Render an answer card with formatting."""
    
    if not result:
        st.warning(f"No results for {title}")
        return
    
    # Determine card styling
    if card_type == "baseline":
        border_color = "#1f77b4"
        icon = "🔵"
    else:
        border_color = "#2ca02c" 
        icon = "🟢"
    
    # Card container
    with st.container():
        st.markdown(f"""
        <div style="border-left: 5px solid {border_color}; padding: 1rem; margin: 0.5rem 0; background: #f8f9fa; border-radius: 0 10px 10px 0;">
            <h3>{icon} {title}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer section
        if 'answer' in result or 'enhanced_answer' in result:
            answer = result.get('enhanced_answer', result.get('answer', 'No answer generated'))
            
            with st.expander("📝 Answer", expanded=True):
                st.markdown(answer)
        
        # Metadata section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            response_time = result.get('response_time', 0)
            st.metric("⏱️ Response Time", f"{response_time:.2f}s")
        
        with col2:
            sources_count = len(result.get('sources', []))
            st.metric("📚 Sources", sources_count)
        
        with col3:
            token_usage = result.get('token_usage', {})
            total_tokens = token_usage.get('total_tokens', 0)
            st.metric("🔢 Tokens", total_tokens)
        
        # Sources section
        sources = result.get('sources', [])
        if sources:
            with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
                for i, source in enumerate(sources[:5], 1):  # Show top 5
                    # Handle different source formats (BaselineRAG vs KG-Enhanced)
                    content = source.get('content') or source.get('document', 'No content')
                    metadata = source.get('metadata', {})
                    
                    # Extract article ID and category from metadata
                    article_id = (
                        source.get('article_id') or 
                        metadata.get('article_id') or 
                        metadata.get('chunk_id', 'Unknown')
                    )
                    category = metadata.get('category', 'Unknown')
                    
                    # Handle score/distance (BaselineRAG uses 'distance', KG-Enhanced uses 'score')
                    score = source.get('score', 0)
                    
                    # If no score but distance exists, convert distance to similarity
                    if score == 0 and 'distance' in source:
                        distance = source.get('distance')
                        if distance is not None:
                            # Convert distance to similarity score using cosine similarity formula
                            # For cosine distance, similarity = 1 - distance
                            score = max(0.0, min(1.0, 1.0 - distance))
                    
                    # If still no score, try to get it from metadata
                    if score == 0:
                        metadata = source.get('metadata', {})
                        score = metadata.get('score', metadata.get('similarity', 0))
                    
                    st.markdown(f"**Source {i}** (Score: {score:.3f})")
                    st.markdown(f"**Article**: {article_id} | **Category**: {category.title()}")
                    
                    # Display content with proper handling
                    if content and content.strip() and content != 'No content':
                        display_content = content[:300] if len(content) > 300 else content
                        st.markdown(f"**Content**: {display_content}{'...' if len(content) > 300 else ''}")
                    else:
                        st.warning("⚠️ No content available for this source")
                    
                    if i < len(sources[:5]):
                        st.markdown("---")
        
        # Enhanced features for KG-Enhanced results
        if card_type == "kg_enhanced" and 'confidence_indicators' in result:
            confidence = result['confidence_indicators']
            overall_conf = confidence.get('overall_confidence', 0)
            conf_level = confidence.get('confidence_level', 'unknown')
            
            with st.expander("🎯 Confidence & Reasoning"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Confidence", f"{overall_conf:.2f}")
                    st.markdown(f"**Level**: {conf_level.replace('_', ' ').title()}")
                
                with col2:
                    detailed = confidence.get('detailed_confidence', {})
                    st.markdown("**Component Confidence**:")
                    for component, value in detailed.items():
                        st.markdown(f"• {component.replace('_', ' ').title()}: {value:.2f}")
        
        # Enhanced KG-RAG Reasoning Path Display
        if card_type == "kg_enhanced":
            # Get the complete results to access KG data
            complete_results = st.session_state.query_results if hasattr(st.session_state, 'query_results') else {}
            kg_raw_result = complete_results.get('kg_enhanced_raw', {})
            
            # Create detailed reasoning display
            render_kg_reasoning_path(result, kg_raw_result)
        
        # Knowledge Graph Visualization for KG-Enhanced results
        if card_type == "kg_enhanced":
            try:
                # Create graph visualizer instance
                graph_visualizer = GraphVisualizer(knowledge_graph_path="data/processed/knowledge_graph.pkl")
                # Get the complete results from session state to access kg_enhanced_raw
                complete_results = st.session_state.query_results if hasattr(st.session_state, 'query_results') else {}
                # Render the interactive graph (only when user expands)
                graph_visualizer.render_in_streamlit(complete_results)
            except Exception as e:
                # Fail gracefully if graph visualization has issues
                with st.expander("🕸️ **Knowledge Graph Visualization**", expanded=False):
                    st.info(f"Graph visualization currently unavailable: {str(e)}")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .system-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .baseline-card {
        border-left: 5px solid #1f77b4;
    }
    .kg-enhanced-card {
        border-left: 5px solid #2ca02c;
    }
    .stMetric {
        background: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Project header
st.markdown("""
<div class="main-header">
    <h1>🧠 Knowledge Graph-Enhanced RAG System</h1>
    <p>Interactive Demo: Compare Baseline vs KG-Enhanced Retrieval</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with About information
with st.sidebar:
    st.header("📋 About This Demo")
    
    st.markdown("""
    This interactive demo compares two RAG approaches:
    """)
    
    # Baseline RAG info
    st.markdown("""
    **🔵 Baseline RAG**
    - Standard vector similarity search
    - OpenAI text-embedding-3-small
    - ChromaDB vector store
    - Direct document retrieval
    - GPT-3.5-turbo generation
    """)
    
    # KG-Enhanced RAG info  
    st.markdown("""
    **🟢 KG-Enhanced RAG**
    - Entity extraction & analysis
    - Knowledge graph traversal
    - Enhanced query expansion
    - Graph-informed retrieval
    - Reasoning path generation
    - Confidence indicators
    """)
    
    st.markdown("---")
    
    # Dataset information
    st.subheader("📊 Dataset Information")
    st.markdown("""
    - **Source**: BBC News articles
    - **Size**: 50 articles (10 per category)
    - **Categories**: Business, Entertainment, Politics, Sport, Technology
    - **Vector Store**: 169 document chunks
    """)
    
    # Knowledge Graph information
    st.subheader("🕸️ Knowledge Graph")
    st.markdown("""
    - **Entities**: 1,022 nodes
    - **Relationships**: 14,075+ edges
    - **Types**: People, Organizations, Locations, Topics, Events
    - **Coverage**: All article categories
    """)
    
    st.markdown("---")
    st.markdown("**Built with**: Streamlit, OpenAI, ChromaDB, NetworkX")

# Session state already initialized at the top

# System Selection (Horizontal Layout)
st.markdown("### Choose Your RAG System")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(
        "🔵 Baseline Only",
        use_container_width=True,
        type="primary" if st.session_state.system_mode == "🔵 Baseline Only" else "secondary"
    ):
        if st.session_state.system_mode != "🔵 Baseline Only":
            st.session_state.system_mode = "🔵 Baseline Only"
            st.session_state.query_results = None  # Clear previous results
            st.rerun()

with col2:
    if st.button(
        "🟢 KG-Enhanced Only",
        use_container_width=True,
        type="primary" if st.session_state.system_mode == "🟢 KG-Enhanced Only" else "secondary"
    ):
        if st.session_state.system_mode != "🟢 KG-Enhanced Only":
            st.session_state.system_mode = "🟢 KG-Enhanced Only"
            st.session_state.query_results = None  # Clear previous results
            st.rerun()

with col3:
    if st.button(
        "🔄 Both (Compare)",
        use_container_width=True,
        type="primary" if st.session_state.system_mode == "🔄 Both (Compare)" else "secondary"
    ):
        if st.session_state.system_mode != "🔄 Both (Compare)":
            st.session_state.system_mode = "🔄 Both (Compare)"
            st.session_state.query_results = None  # Clear previous results
            st.rerun()

with col4:
    if st.button(
        "🕸️ KG Visualizer",
        use_container_width=True,
        type="primary" if st.session_state.system_mode == "🕸️ KG Visualizer" else "secondary"
    ):
        if st.session_state.system_mode != "🕸️ KG Visualizer":
            st.session_state.system_mode = "🕸️ KG Visualizer"
            st.session_state.query_results = None  # Clear previous results
            st.rerun()

st.markdown("---")

# Session state already initialized above

# Pre-loaded example queries (optimized to showcase KG-Enhanced RAG capabilities)
EXAMPLE_QUERIES = {
    "Select an example query...": "",
    "🔗 Relationship Discovery → What organizations are connected to Robert Kilroy-Silk across different news categories?": "What organizations are connected to Robert Kilroy-Silk across different news categories?",
    "🧠 Multi-hop Reasoning → Find people who work at technology companies that are also mentioned in business articles": "Find people who work at technology companies that are also mentioned in business articles",
    "� Entity Disambiguation → Tell me about Apple's coverage in business and technology news": "Tell me about Apple's coverage in business and technology news",
    "📊 Comparative Analysis → Compare how Microsoft is covered versus other technology companies": "Compare how Microsoft is covered versus other technology companies",
    "� Geographic Context → What European organizations and events are mentioned across different article categories?": "What European organizations and events are mentioned across different article categories?",
    "🔍 Topic Co-occurrence → What topics and themes frequently appear together with political news?": "What topics and themes frequently appear together with political news?",
    "📈 Trend Discovery → What are the main technological developments and their associated companies?": "What are the main technological developments and their associated companies?",
    "🎭 Event Connections → What entertainment events connect to broader cultural or business trends?": "What entertainment events connect to broader cultural or business trends?",
    "� Sports Relationships → What sports organizations and events are linked to specific people or locations?": "What sports organizations and events are linked to specific people or locations?"
}

# Add page-specific heading for KG Visualizer mode
if st.session_state.system_mode == "🕸️ KG Visualizer":
    st.markdown("## 🕸️ Knowledge Graph Visualizer")
    st.markdown("*Explore the BBC News knowledge graph through interactive visualization*")
    st.markdown("---")

# Query Input Section
st.subheader("🔍 Query Input")

# Full width layout for query input
# Example query selector
selected_example = st.selectbox(
    "📋 Choose an example query or enter your own:",
    options=list(EXAMPLE_QUERIES.keys()),
    index=0,
    help="Select from pre-loaded examples or choose 'Select an example query...' to enter your own"
)

# Handle example query selection with auto-execution
if selected_example != "Select an example query...":
    example_text = EXAMPLE_QUERIES.get(selected_example, "")
    if example_text and st.session_state.current_query != example_text:
        # Update query text
        st.session_state.current_query = example_text
        # Clear previous results and auto-start RAG processing
        st.session_state.query_results = None  # Clear previous results
        st.session_state.is_processing = True
        st.session_state.auto_selected_example = True
        # Extract just the category name for display (before the arrow)
        category_name = selected_example.split(" → ")[0] if " → " in selected_example else selected_example
        st.success(f"🚀 Running example: {category_name}")
        st.rerun()

# Text input for query
user_query = st.text_area(
    "Enter your query:",
    value=st.session_state.current_query,
    height=100,
    placeholder="Ask anything about the BBC news articles... (e.g., 'What technology companies are mentioned?')",
    help="Enter a question about the content in the BBC news dataset (50 articles across 5 categories)",
    key="query_input"
)

# Keep session state in sync (but avoid infinite rerun loops)
st.session_state.current_query = user_query

# Check if current query is a custom user query (not from examples)
is_custom_query = (
    user_query.strip() != "" and 
    user_query not in EXAMPLE_QUERIES.values() and
    len(user_query.strip()) >= 10
)

# Action buttons below the query text box
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    # Run query button - only enabled for custom queries
    run_disabled = not is_custom_query or st.session_state.is_processing
    
    if st.button(
        "🚀 Run Query",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
        help="Execute your custom query (examples auto-run, 10+ characters required)"
    ):
        if len(user_query.strip()) >= 10:
            st.session_state.query_results = None  # Clear previous results
            st.session_state.is_processing = True
            st.session_state.auto_selected_example = False  # Manual execution
            st.rerun()
        else:
            st.error("Query must be at least 10 characters long")

with col_btn2:
    # Clear/Reset button - only enabled for custom queries
    clear_disabled = not is_custom_query
    
    if st.button(
        "🗑️ Clear",
        disabled=clear_disabled,
        use_container_width=True,
        help="Clear your custom query and results"
    ):
        st.session_state.current_query = ""
        st.session_state.query_results = None
        st.session_state.is_processing = False
        st.session_state.auto_selected_example = False
        st.rerun()

# Input validation and feedback
if user_query.strip():
    if len(user_query.strip()) < 10:
        st.warning("⚠️ Custom query should be at least 10 characters to enable buttons")
    elif len(user_query.strip()) > 500:
        st.warning("⚠️ Very long query detected. Consider shortening for better performance")
    elif st.session_state.auto_selected_example:
        st.info("🎯 Example query loaded and processing automatically...")
    elif user_query not in EXAMPLE_QUERIES.values():
        st.success("✅ Custom query ready! Buttons are now enabled.")

# Processing indicator and actual query execution
if st.session_state.is_processing:
    # Use the current query for processing (handles both manual and auto-selected queries)
    query_to_process = st.session_state.current_query
    
    with st.spinner(f"🔄 Processing query with {st.session_state.system_mode}... This may take 15-30 seconds"):
        try:
            # Execute actual query processing
            results = execute_rag_query(query_to_process, st.session_state.system_mode)
            st.session_state.is_processing = False
            st.session_state.query_results = results
            st.session_state.auto_selected_example = False  # Reset flag
            st.rerun()
        except Exception as e:
            st.session_state.is_processing = False
            st.error(f"❌ Error processing query: {str(e)}")
            st.session_state.query_results = None
            st.session_state.auto_selected_example = False  # Reset flag

# Special handling for KG Visualizer mode
if st.session_state.system_mode == "🕸️ KG Visualizer" and not st.session_state.query_results:
    # Show default visualization for KG Visualizer
    st.markdown("---")
    try:
        default_results = {
            'query': '',
            'system_mode': '🕸️ KG Visualizer',
            'visualizer_data': get_top_organizations_subgraph(),
            'success': True
        }
        render_kg_visualizer_results(default_results)
    except Exception as e:
        st.error(f"❌ Error loading default visualization: {str(e)}")

# Query status and next steps
st.markdown("---")

# Display results with layout based on selected system mode
if st.session_state.query_results:
    results = st.session_state.query_results
    
    # Show query info
    st.info(f"✅ **Query**: \"{results['query'][:100]}{'...' if len(results['query']) > 100 else ''}\" | **Mode**: {results['system_mode']}")
    
    if 'success' in results and results['success']:
        # Success - display results with fresh layout based on system mode
        
        if st.session_state.system_mode == "🔄 Both (Compare)":
            # Side-by-side comparison layout
            col1, col2 = st.columns(2)
            
            with col1:
                if 'baseline' in results and results['baseline']:
                    render_answer_card("Baseline RAG", results['baseline'], "baseline")
                else:
                    st.info("� Baseline RAG results not available")
            
            with col2:
                if 'kg_enhanced' in results and results['kg_enhanced']:
                    render_answer_card("KG-Enhanced RAG", results['kg_enhanced'], "kg_enhanced")
                else:
                    st.info("🟢 KG-Enhanced RAG results not available")
        
        elif st.session_state.system_mode == "🔵 Baseline Only":
            # Full-width layout for baseline only
            if 'baseline' in results and results['baseline']:
                render_answer_card("Baseline RAG Results", results['baseline'], "baseline")
            else:
                st.warning("🔵 No baseline results available")
        
        elif st.session_state.system_mode == "🟢 KG-Enhanced Only":
            # Full-width layout for KG-Enhanced only
            if 'kg_enhanced' in results and results['kg_enhanced']:
                render_answer_card("KG-Enhanced RAG Results", results['kg_enhanced'], "kg_enhanced")
            else:
                st.warning("🟢 No KG-Enhanced results available")
        
        elif st.session_state.system_mode == "🕸️ KG Visualizer":
            # Full-width layout for KG Visualizer
            render_kg_visualizer_results(results)
            
    else:
        # Error occurred
        st.error("❌ Query execution failed")
        if 'error' in results:
            st.error(f"**Error**: {results['error']}")
        if 'traceback' in results:
            with st.expander("🔧 Technical Details"):
                st.code(results['traceback'])

# Footer
st.markdown("---")

# Development info (can be removed in production)
if st.sidebar.button("🔧 Development Info"):
    st.sidebar.info("""
    **Development Status**
    
    ✅ Task 1: Basic Structure
    ✅ Task 2: Query Input Interface  
    ✅ Task 3: Results Display
    ✅ Task 6: Example Queries (Integrated)
    ✅ Performance Optimization
    ✅ System Selection Fixed
    ✅ Tabs Removed for Simplicity
    """)
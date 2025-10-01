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
sys.path.insercol1, col2 = st.columns([3, 1])

with col1:
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
            # Clear previous results and auto-start RAG processingoot))

# Import RAG systems
try:
    from src.retrieval.baseline_rag import BaselineRAG
    from src.retrieval.kg_enhanced_rag import KGEnhancedRAG
    from src.retrieval.response_generator import ResponseGenerator
    from src.retrieval.response_tracker import ResponseTracker
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
                    if score == 0 and 'distance' in source:
                        # Convert distance to similarity score (lower distance = higher similarity)
                        distance = source.get('distance', 1.0)
                        score = max(0, 1.0 - distance)  # Simple conversion
                    
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
        
        # Reasoning path for enhanced results
        if card_type == "kg_enhanced" and 'reasoning_path' in result:
            reasoning = result['reasoning_path']
            steps = reasoning.get('steps', [])
            
            if steps:
                with st.expander(f"🔍 Reasoning Path ({len(steps)} steps)"):
                    for step in steps:
                        step_num = step.get('step', 0)
                        description = step.get('description', 'No description')
                        confidence = step.get('confidence', 0)
                        details = step.get('details', '')
                        
                        st.markdown(f"**Step {step_num}**: {description}")
                        st.markdown(f"*Confidence: {confidence:.2f}*")
                        if details:
                            st.markdown(f"*{details}*")
                        st.markdown("")

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

st.markdown("### Experience the power of knowledge graph-enhanced retrieval augmented generation")

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
    - GPT-4-turbo-preview generation
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

# Main content area
st.header("Interactive Query Interface")

# Initialize session state for query management
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'auto_selected_example' not in st.session_state:
    st.session_state.auto_selected_example = False

# Pre-loaded example queries (optimized for BBC news dataset)
EXAMPLE_QUERIES = {
    "Select an example query...": "",
    "🏢 Technology Companies → What technology companies like Microsoft, AOL, and Yahoo are discussed?": "What technology companies like Microsoft, AOL, and Yahoo are discussed?",
    "🏛️ Political Developments → What political parties and politicians like Robert Kilroy-Silk are mentioned?": "What political parties and politicians like Robert Kilroy-Silk are mentioned?",
    "🎾 Sports & Tennis → What sports news and tennis developments are covered in the articles?": "What sports news and tennis developments are covered in the articles?",
    "🎮 Gaming Industry → What gaming companies and trends are discussed at events like CES?": "What gaming companies and trends are discussed at events like CES?",
    "🏦 Business & Finance → What business developments and financial institutions are mentioned?": "What business developments and financial institutions are mentioned?",
    "🎭 Entertainment News → What entertainment events like the Oscars and BBC shows are discussed?": "What entertainment events like the Oscars and BBC shows are discussed?",
    "🔒 Security & Technology → What security patches and software vulnerabilities are reported by Microsoft?": "What security patches and software vulnerabilities are reported by Microsoft?",
    "🌍 International Relations → What European Union policies and international events are covered?": "What European Union policies and international events are covered?"
}

# Query Input Section
st.subheader("🔍 Query Input")

col1, col2 = st.columns([3, 1])
    
    with col1:
        # Example query selector
        selected_example = st.selectbox(
            "� Choose an example query or enter your own:",
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
        
        # Show current selection status
        if selected_example != "Select an example query..." and st.session_state.current_query:
            category_name = selected_example.split(" → ")[0] if " → " in selected_example else selected_example
            st.info(f"📋 **Current selection**: {category_name}")
        
        # Keep session state in sync (but avoid infinite rerun loops)
        st.session_state.current_query = user_query
    
    with col2:
        # System selector
        st.markdown("### 🎛️ System Selection")
        
        system_mode = st.radio(
            "Choose RAG system(s):",
            options=[
                "🔵 Baseline Only", 
                "🟢 KG-Enhanced Only", 
                "🔄 Both (Compare)"
            ],
            index=2,  # Default to "Both"
            help="""
            • Baseline: Standard vector-based RAG
            • KG-Enhanced: Graph-aware RAG with reasoning
            • Both: Side-by-side comparison
            """
        )
        
        st.markdown("---")
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            # Run query button
            run_disabled = len(user_query.strip()) < 10 or st.session_state.is_processing
            
            if st.button(
                "🚀 Run Query",
                disabled=run_disabled,
                use_container_width=True,
                type="primary",
                help="Execute the query with selected system(s) (examples auto-run)"
            ):
                if len(user_query.strip()) >= 10:
                    st.session_state.query_results = None  # Clear previous results
                    st.session_state.is_processing = True
                    st.session_state.auto_selected_example = False  # Manual execution
                    st.rerun()
                else:
                    st.error("Query must be at least 10 characters long")
        
        with col_btn2:
            # Clear/Reset button
            if st.button(
                "🗑️ Clear",
                use_container_width=True,
                help="Clear the current query and results"
            ):
                st.session_state.current_query = ""
                st.session_state.query_results = None
                st.session_state.is_processing = False
                st.session_state.auto_selected_example = False
                st.rerun()
    
    # Input validation and feedback
    if user_query.strip():
        if len(user_query.strip()) < 10:
            st.warning("⚠️ Query should be at least 10 characters for best results")
        elif len(user_query.strip()) > 500:
            st.warning("⚠️ Very long query detected. Consider shortening for better performance")
        elif st.session_state.auto_selected_example:
            st.info("🎯 Example query loaded and processing automatically...")
        else:
            st.success("✅ Query looks good! Click 'Run Query' or select an example above")
    
    # Processing indicator and actual query execution
    if st.session_state.is_processing:
        # Use the current query for processing (handles both manual and auto-selected queries)
        query_to_process = st.session_state.current_query
        
        with st.spinner(f"🔄 Processing query with {system_mode}... This may take 15-30 seconds"):
            try:
                # Execute actual query processing
                results = execute_rag_query(query_to_process, system_mode)
                st.session_state.is_processing = False
                st.session_state.query_results = results
                st.session_state.auto_selected_example = False  # Reset flag
                st.rerun()
            except Exception as e:
                st.session_state.is_processing = False
                st.error(f"❌ Error processing query: {str(e)}")
                st.session_state.query_results = None
                st.session_state.auto_selected_example = False  # Reset flag
    
    # Query status and next steps
    st.markdown("---")
    
    # Display results if available
    if st.session_state.query_results:
        results = st.session_state.query_results
        
        # Show query info
        st.info(f"✅ **Query**: \"{results['query'][:100]}{'...' if len(results['query']) > 100 else ''}\" | **Mode**: {results['system_mode']}")
        
        if 'success' in results and results['success']:
            # Success - display the results using render cards
            
            # Debug information (can be removed later)
            st.write(f"🔧 Debug - System Mode: {results.get('system_mode', 'Unknown')}")
            st.write(f"🔧 Debug - Has baseline: {'baseline' in results and results['baseline'] is not None}")
            st.write(f"🔧 Debug - Has kg_enhanced: {'kg_enhanced' in results and results['kg_enhanced'] is not None}")
            
            # Show baseline results
            if 'baseline' in results and results['baseline']:
                render_answer_card("Baseline RAG", results['baseline'], "baseline")
            
            # Show KG-enhanced results  
            if 'kg_enhanced' in results and results['kg_enhanced']:
                render_answer_card("KG-Enhanced RAG", results['kg_enhanced'], "kg_enhanced")
                
        else:
            # Error occurred
            st.error("❌ Query execution failed")
            if 'error' in results:
                st.error(f"**Error**: {results['error']}")
            if 'traceback' in results:
                with st.expander("🔧 Technical Details"):
                    st.code(results['traceback'])
    else:
        st.info("📝 **Ready**: Enter a query above and click 'Run Query' to see results")
    
    # Task completion status
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Task 1**: Basic Structure Complete")
        st.success("✅ **Task 2**: Query Interface Complete")
    with col2:
        st.success("✅ **Task 3**: Results Display Complete")
        st.info("⏳ **Task 4**: Graph Visualization (Next)")



# Footer
st.markdown("---")
st.markdown("**Demo Status**: Core functionality complete - Compare RAG systems with interactive queries")

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
    """)
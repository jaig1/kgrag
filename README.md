# KG-Enhanced RAG System Setup Guide

Complete setup instructions for the Knowledge Graph-Enhanced Retrieval-Augmented Generation system.

## 🎯 Prerequisites

- **Python 3.8+** (Check with `python --version`)
- **OpenAI API Key** (Required for entity extraction and RAG functionality)
- **8GB+ RAM** (Recommended for processing knowledge graphs)

## 🚀 Quick Start (Automatic Setup)

For a complete automated setup:

```bash
# 1. Clone and navigate to project
cd kgrag

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure OpenAI API key
cp .env.example .env
# Edit .env file and add: OPENAI_API_KEY=your_key_here

# 5. Run complete setup
python setup/complete_setup.py

# 6. Launch application
python -m streamlit run src/interface/demo_app.py
```

## 🔧 Manual Setup (Step-by-Step)

If you prefer manual control or need to troubleshoot:

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd kgrag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify critical packages are installed
pip show streamlit openai chromadb networkx plotly
```

### Step 2: Configure OpenAI API Key

```bash
# Copy environment template
cp .env.example .env

# Add your OpenAI API key (REQUIRED)
echo "OPENAI_API_KEY=your_actual_api_key_here" >> .env
# OR edit .env file manually with a text editor
```

⚠️ **CRITICAL**: Replace `your_actual_api_key_here` with your real OpenAI API key.

### Step 3: Validate Environment

```bash
# Verify project structure and dependencies
python setup/verify_environment_setup.py
```

**Expected Result**: `5/5 validations passed` ✅

### Step 4: Download BBC Dataset  

```bash
# Download and prepare 50 BBC News articles
python setup/setup_bbc_dataset.py
```

**Expected Result**: 
- ✅ Creates `data/raw/bbc_news_subset.json` (50 articles, ~125KB)
- ✅ Shows "Successfully processed BBC News dataset!"

### Step 5: Process Documents into Chunks

```bash
# Process articles into searchable chunks  
python setup/setup_vector_database.py
```

**Expected Result**:
- ✅ Creates `data/processed/processed_articles.json` (169 chunks, ~500KB)
- ✅ Shows "Processing pipeline completed successfully!"

### Step 6: Create Vector Database

```bash
# Generate embeddings and populate ChromaDB
python setup/setup_vector_store.py
```

**Expected Result**:
- ✅ Creates `data/chroma_db/` directory (ChromaDB files)
- ✅ Creates `data/processed/embeddings_cache.pkl`
- ✅ Shows "Vector database populated with 169 documents"

### Step 7: Build Knowledge Graph

```bash
# Extract entities and build knowledge graph (takes 2-5 minutes)
python setup/setup_knowledge_graph.py
```

**Expected Result**:
- ✅ Creates `data/processed/knowledge_graph.pkl` (~145KB, 1000+ nodes)
- ✅ Creates `data/processed/extracted_entities.json` (~100KB)
- ✅ Creates `data/processed/resolved_entities.json` (~165KB)
- ✅ Shows "Phase 3 Complete! Knowledge graph ready"

### Step 8: Verify Complete Setup

```bash
# Verify all critical files exist
python -c "
import os
files = [
    'data/raw/bbc_news_subset.json',
    'data/processed/processed_articles.json', 
    'data/processed/knowledge_graph.pkl',
    'data/processed/extracted_entities.json',
    'data/chroma_db'
]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print('❌ Missing files:', missing)
    exit(1)
else:
    print('✅ All required files present!')
    print('🎉 Setup complete - ready to launch app!')
"
```

**Expected Result**: `✅ All required files present!` and `🎉 Setup complete`

## 🚀 **Quick Setup (One Command)**

For a complete fresh setup, run the master setup script:

```bash
# Activate virtual environment and create directories first
source .venv/bin/activate
python setup/setup_bbc_dataset.py
python setup/complete_setup.py
```

This will automatically run all setup steps in the correct order and validate the system.

## 🧪 System Validation (Critical Before Launch)

**IMPORTANT**: Test both systems to ensure Streamlit app will work:

### Test 1: Baseline RAG System
```bash
source .venv/bin/activate
python setup/test_baseline_rag.py
```
**Expected Result**: `✅ Baseline RAG testing complete!` with 100% success rate

### Test 2: KG-Enhanced RAG System  
```bash
source .venv/bin/activate
python setup/test_kg_enhanced_rag.py
```
**Expected Result**: `🎉 SCRIPT 5 (KG-ENHANCED RAG) VALIDATION: ✅ PASSED`

### Test 3: Knowledge Graph Operations
```bash
source .venv/bin/activate
python setup/test_graph_traversal.py
```
**Expected Result**: `🎉 SCRIPT 2 (GRAPH TRAVERSAL) VALIDATION: ✅ PASSED`

## � Launch Streamlit Application

Once all tests pass, launch the app:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Then launch Streamlit (recommended)
python -m streamlit run src/interface/demo_app.py

# Alternative: Use full path (if activation doesn't work)
.venv/bin/python -m streamlit run src/interface/demo_app.py
```

**Expected Result**: 
- ✅ App starts on `http://localhost:8501`
- ✅ No import errors or missing files
- ✅ All 4 app modes load successfully:
  - Baseline RAG
  - KG-Enhanced RAG  
  - Side-by-Side Comparison
  - KG Visualizer

## � Troubleshooting

### Common Issues & Solutions

**❌ Error: "No module named 'openai'" or similar**
```bash
# Ensure virtual environment is activated and dependencies installed
source .venv/bin/activate
pip install -r requirements.txt
```

**❌ Error: "OpenAI API key not found"** 
```bash
# Verify API key is configured
cat .env | grep OPENAI_API_KEY
# Should show: OPENAI_API_KEY=sk-...
```

**❌ Error: "FileNotFoundError: data/processed/knowledge_graph.pkl"**
```bash
# Rerun knowledge graph construction
source .venv/bin/activate
python setup/setup_knowledge_graph.py
```

**❌ Error: "ChromaDB collection not found" or "0 documents"**
```bash
# Rerun vector database population
source .venv/bin/activate
python setup/setup_vector_store.py
```

**❌ Streamlit app fails to start or shows errors**
```bash
# Test individual systems first
source .venv/bin/activate
python setup/test_baseline_rag.py
python setup/test_kg_enhanced_rag.py
# All tests should show ✅ PASSED
```

**❌ Port 8501 already in use**
```bash
# Use different port
python -m streamlit run src/interface/demo_app.py --server.port=8502
```

### Performance Notes
- **Setup time**: 5-10 minutes total (depends on OpenAI API speed)
- **Baseline RAG**: ~3-8 seconds per query  
- **KG-Enhanced RAG**: ~4-12 seconds per query
- **Memory usage**: ~2-4GB during operation

## ✅ Setup Summary

### Quick Setup (Recommended)
```bash
source .venv/bin/activate
python setup/setup_bbc_dataset.py
python setup/complete_setup.py
python -m streamlit run src/interface/demo_app.py
```

### Manual Setup Order  
```bash
# 1. Activate virtual environment first
source .venv/bin/activate

# 2. Create data directories (must be first!)
python setup/setup_bbc_dataset.py          # Download 50 BBC articles

# 3. Run remaining setup steps
python setup/verify_environment_setup.py   # Verify dependencies (5/5 pass)
python setup/setup_vector_database.py      # Process into 169 chunks
python setup/setup_vector_store.py         # Populate ChromaDB with embeddings
python setup/setup_knowledge_graph.py      # Build 1,000+ node knowledge graph
python setup/test_baseline_rag.py          # Test baseline system (100% success)
python setup/test_kg_enhanced_rag.py       # Test enhanced system (100% success)
python -m streamlit run src/interface/demo_app.py  # Launch app
```

### Expected Final State
- ✅ `data/raw/bbc_news_subset.json` (50 articles, ~125KB)
- ✅ `data/processed/processed_articles.json` (169 chunks, ~500KB)  
- ✅ `data/chroma_db/` (vector database)
- ✅ `data/processed/knowledge_graph.pkl` (1,000+ nodes, ~145KB)
- ✅ App accessible at `http://localhost:8501` with 4 working modes

**Total setup time: 5-10 minutes**

## 📄 License

This project is for educational and research purposes.
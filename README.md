# KG-Enhanced RAG System

A Knowledge Graph-Enhanced Retrieval Augmented Generation system that combines vector databases with knowledge graphs for improved accuracy and explainability.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd kgrag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file and add your OpenAI API key
```

### 2. Download Dataset

```bash
# Run the dataset loader
python run_dataset_loader.py

# Or run directly
python -m src.ingestion.dataset_loader
```

### 3. Process Documents

```bash
# Run the document processing pipeline
python run_document_processor.py

# Or run directly
python -m src.ingestion.processor
```

### 4. Explore Data

```bash
# Launch Jupyter and open the exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 5. Validate Implementation

```bash
# Run Phase 1 validation
python validate_phase1.py

# Run Phase 2 validation
python validate_phase2.py
```

## 📁 Project Structure

```
kgrag/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── ingestion/                # Data ingestion module
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # BBC News dataset loader
│   │   ├── preprocessor.py       # Text preprocessing and cleaning
│   │   ├── chunker.py            # Document chunking with overlaps
│   │   └── processor.py          # Complete processing pipeline
│   ├── storage/                  # Vector DB and KG storage
│   │   └── __init__.py
│   ├── retrieval/                # Query processing and retrieval
│   │   └── __init__.py
│   ├── evaluation/               # System evaluation
│   │   └── __init__.py
│   └── interface/                # User interface components
│       └── __init__.py
├── data/                         # Data storage
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data and indices
├── notebooks/                    # Jupyter notebooks
│   └── 01_data_exploration.ipynb # Data exploration notebook
├── tests/                        # Test suite
│   └── __init__.py
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── run_dataset_loader.py         # Dataset loader runner script
├── run_document_processor.py     # Document processor runner script  
├── validate_phase1.py            # Phase 1 validation script
├── validate_phase2.py            # Phase 2 validation script
└── README.md                     # This file
```

## 🛠 Tech Stack

- **Python 3.8+** - Core language
- **ChromaDB** - Vector database for embeddings
- **NetworkX** - Knowledge graph construction and analysis
- **OpenAI API** - Embeddings and text generation
- **Datasets (HuggingFace)** - BBC News dataset access
- **Streamlit** - Web interface (future phases)
- **Jupyter** - Interactive development and exploration

## 📊 Dataset

The system uses the BBC News dataset from HuggingFace (`SetFit/bbc-news`):
- **50 articles total** (10 per category)
- **5 categories**: business, entertainment, politics, sport, tech
- **Balanced distribution** for fair evaluation
- **Unique article IDs** in format: `bbc_{category}_{index}`

## ⚙️ Configuration

Configuration is managed through environment variables and `src/config.py`:

### Key Parameters
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `CHAT_MODEL`: OpenAI chat model (default: gpt-4-turbo-preview)
- `CHUNK_SIZE`: Text chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `TOP_K_RESULTS`: Number of top results to retrieve (default: 5)

## 🔧 Development Phases

### Phase 1: Environment Setup & Data Acquisition ✅
- [x] Project structure creation
- [x] Dependency management
- [x] Configuration system
- [x] Dataset download and processing
- [x] Data exploration notebook

### Phase 2: Document Processing & Chunking ✅
- [x] Text preprocessing and normalization
- [x] Sentence-aware chunking with overlaps
- [x] Document processing pipeline
- [x] Chunk metadata and statistics

### Phase 3: Vector Database Implementation (Next)
- [ ] ChromaDB setup and configuration
- [ ] Embedding generation and storage
- [ ] Vector similarity search
- [ ] Chunk indexing and retrieval

### Phase 3: Knowledge Graph Construction (Future)
- [ ] Entity extraction and linking
- [ ] Relationship identification
- [ ] Graph storage and querying
- [ ] Graph visualization

### Phase 4: Enhanced Retrieval System (Future)
- [ ] Hybrid vector + graph search
- [ ] Query understanding and expansion
- [ ] Result ranking and fusion
- [ ] Context-aware retrieval

### Phase 5: Generation & Interface (Future)
- [ ] Response generation with context
- [ ] Explanation generation
- [ ] Streamlit web interface
- [ ] Evaluation metrics

## 📝 Usage Examples

### Load and Process Dataset
```python
from src.ingestion.dataset_loader import BBCDatasetLoader

# Initialize loader
loader = BBCDatasetLoader()

# Download and process
loader.download_dataset()
loader.create_balanced_subset()
loader.save_subset()

# View statistics
loader.display_statistics()
```

### Configuration Access
```python
from src.config import config

print(f"Embedding model: {config.EMBEDDING_MODEL}")
print(f"Dataset size: {config.DATASET_SIZE}")
print(f"Output path: {config.get_bbc_subset_path()}")
```

## 🧪 Testing

```bash
# Run all tests (when available)
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_dataset_loader.py
```

## 📚 Dependencies

Key packages and their purposes:
- `openai>=1.12.0` - OpenAI API access
- `chromadb>=0.4.22` - Vector database
- `networkx>=3.2.1` - Knowledge graph operations
- `datasets>=2.16.1` - HuggingFace datasets
- `pandas>=2.1.4` - Data manipulation
- `plotly>=5.17.0` - Interactive visualizations
- `streamlit>=1.31.1` - Web interface
- `spacy>=3.7.2` - NLP processing

## 🤝 Contributing

1. Ensure all dependencies are installed
2. Follow PEP 8 style guidelines
3. Add tests for new functionality
4. Update documentation as needed

## 📄 License

This project is for educational and research purposes.
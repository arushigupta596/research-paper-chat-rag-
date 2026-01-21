# File Index

Complete index of all files in the Research Paper Chat Assistant project.

## Root Directory

### Configuration Files
- **requirements.txt**: Python package dependencies
- **.env.example**: Environment variable template
- **.gitignore**: Git ignore rules for version control

### Application Files
- **app.py**: Main Streamlit application entry point

### Documentation
- **README.md**: Comprehensive user documentation
- **QUICKSTART.md**: 5-minute quick start guide
- **PROJECT_SUMMARY.md**: Technical project overview
- **ARCHITECTURE.md**: System architecture diagrams and explanations
- **FILE_INDEX.md**: This file - complete file listing

## Source Code (`src/`)

### Core Module
- **src/__init__.py**: Package initialization
- **src/config.py**: Centralized configuration management
  - Environment variable loading
  - Path configuration
  - Model configuration
  - Processing parameters

### Document Processing (`src/document_processing/`)
- **src/document_processing/__init__.py**: Module initialization
- **src/document_processing/ocr_engine.py**: PaddleOCR wrapper
  - PDF to image conversion
  - Text extraction with bounding boxes
  - Confidence filtering
  - Page-by-page processing
- **src/document_processing/layout_detector.py**: LayoutParser integration
  - Region detection (text, table, figure, etc.)
  - Bounding box extraction
  - Confidence thresholding
  - IoU calculations
- **src/document_processing/reading_order.py**: Reading order detection
  - Column detection algorithm
  - Region ordering (top-to-bottom, left-to-right)
  - Type-based priority
- **src/document_processing/vlm_extractor.py**: Vision-Language Model extraction
  - Table structure extraction
  - Chart analysis
  - OpenRouter API integration
  - JSON response parsing
- **src/document_processing/document_processor.py**: Main processing pipeline
  - Orchestrates all processing steps
  - OCR + layout merging
  - JSON serialization
  - Batch processing
- **src/document_processing/chunker.py**: Semantic chunking
  - Token-based chunking
  - Overlap handling
  - Metadata enrichment
  - VLM data formatting

### Retrieval (`src/retrieval/`)
- **src/retrieval/__init__.py**: Module initialization
- **src/retrieval/vector_store.py**: ChromaDB wrapper
  - Vector indexing
  - Metadata filtering
  - Similarity search
  - Persistence management
  - Statistics and analytics
- **src/retrieval/rag_retriever.py**: RAG retrieval layer
  - Cross-paper synthesis
  - Diversity sampling (MMR)
  - Evidence grouping
  - Context assembly
  - Result ranking

### LLM Orchestration (`src/llm_orchestration/`)
- **src/llm_orchestration/__init__.py**: Module initialization
- **src/llm_orchestration/answer_engine.py**: Answer synthesis
  - LangChain integration
  - OpenRouter LLM calls
  - Evidence-based prompting
  - Citation generation
  - Multi-hop reasoning

### UI Components (`src/ui/`)
- **src/ui/__init__.py**: Module initialization (reserved for future components)

## Scripts (`scripts/`)

### Setup & Utilities
- **scripts/setup.sh**: Automated installation script
  - Virtual environment setup
  - Dependency installation
  - Directory creation
  - Configuration file setup
- **scripts/test_installation.py**: Installation verification
  - Package import tests
  - Configuration validation
  - Directory checks
  - Component initialization tests

### Processing
- **scripts/process_documents.py**: Document processing CLI
  - Batch PDF processing
  - OCR and layout detection
  - VLM extraction
  - Vector store indexing
  - Command-line argument parsing

## Data Directories

### Input
- **Data/**: PDF research papers (15 files)
  - User-provided research papers
  - Input for document processing pipeline

### Output
- **data/processed/**: Processed document JSON files
  - One JSON per PDF with extracted data
  - Layout regions and reading order
  - VLM extraction results
- **data/embeddings/**: Reserved for future use

### Database
- **chroma_db/**: ChromaDB vector store persistence
  - Vector embeddings
  - Document chunks
  - Metadata index

### Logs
- **logs/**: Application and processing logs
  - **logs/processing.log**: Document processing events
  - **logs/app.log**: Application runtime logs

## File Sizes (Approximate)

### Source Code
- Total Python code: ~3,500 lines
- Configuration: ~100 lines
- Scripts: ~500 lines

### Documentation
- README.md: ~600 lines
- QUICKSTART.md: ~200 lines
- PROJECT_SUMMARY.md: ~400 lines
- ARCHITECTURE.md: ~600 lines

### Data (After Processing)
- Processed JSON: ~1-5MB per paper
- Vector store: ~50-100MB per paper
- Logs: ~1-10MB depending on processing

## File Dependencies

### Core Dependencies
```
app.py
  └── src/
      ├── config.py
      ├── retrieval/
      │   ├── vector_store.py
      │   └── rag_retriever.py
      └── llm_orchestration/
          └── answer_engine.py
```

### Processing Dependencies
```
scripts/process_documents.py
  └── src/
      ├── config.py
      ├── document_processing/
      │   ├── ocr_engine.py
      │   ├── layout_detector.py
      │   ├── reading_order.py
      │   ├── vlm_extractor.py
      │   ├── document_processor.py
      │   └── chunker.py
      └── retrieval/
          └── vector_store.py
```

## Configuration Files

### .env (User-created)
```
OPENROUTER_API_KEY=xxx
LLM_MODEL=openai/gpt-4-turbo-preview
VLM_MODEL=qwen/qwen-vl-max
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=10
```

## Generated Files

### During Processing
1. **data/processed/*.json**: Processed documents
2. **data/processed/*.chunks.json**: Chunk files
3. **chroma_db/**: Vector store files
4. **logs/*.log**: Log files

### Streamlit Cache
- **.streamlit/**: Streamlit configuration (auto-generated)

## File Permissions

### Executable
- scripts/setup.sh: 755 (rwxr-xr-x)

### Read-Only (Recommended)
- Source code: 644 (rw-r--r--)
- Configuration: 644 (rw-r--r--)
- Documentation: 644 (rw-r--r--)

### Sensitive
- .env: 600 (rw-------)

## Backup Recommendations

### Critical Files (Version Control)
- All source code (src/)
- Scripts (scripts/)
- Documentation (*.md)
- Configuration templates (.env.example)

### Critical Data (Manual Backup)
- .env (API keys)
- chroma_db/ (vector store)
- data/processed/ (processed documents)

### Regenerable (Can Skip)
- logs/
- __pycache__/
- .streamlit/

## Future Additions (Planned)

### Documentation
- [ ] API documentation (if adding REST API)
- [ ] Performance benchmarks
- [ ] Cost analysis spreadsheet

### Code
- [ ] Unit tests (tests/)
- [ ] Integration tests
- [ ] CI/CD configuration (.github/)

### Utilities
- [ ] Backup script
- [ ] Migration script
- [ ] Deployment scripts

---

**Total Files**: ~25 Python files, ~10 documentation files
**Total Lines of Code**: ~4,000+ lines
**Last Updated**: 2026-01-19

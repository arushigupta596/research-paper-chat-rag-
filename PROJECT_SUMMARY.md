# Project Summary

## Overview

**Research Paper Chat Assistant** - A comprehensive document understanding and evidence-backed chat system built with Streamlit, enabling natural language interaction with 15 research papers using advanced RAG techniques.

## Key Features

### 1. Layout-Aware Document Understanding
- **OCR with PaddleOCR**: Text extraction with bounding boxes and confidence scores
- **Layout Detection with LayoutParser**: Identify document regions (text, tables, charts, figures)
- **Reading Order Detection**: Custom heuristic-based algorithm for multi-column layouts
- **Region-Level Metadata**: Page numbers, bounding boxes, confidence scores

### 2. VLM-Based Structured Extraction
- **Table Extraction**: Extract headers, rows, units, footnotes from table regions
- **Chart Analysis**: Identify chart types, axes, data series, trends, anomalies
- **Structured JSON Output**: Normalized format for downstream processing
- **OpenRouter Integration**: Uses Vision-Language Models via API

### 3. Semantic Chunking & Knowledge Representation
- **Structure-Aware Chunking**: Respects reading order and region boundaries
- **Metadata Enrichment**: Paper name, page, region type, bounding box, reading order
- **Configurable Chunk Size**: Token-based with overlap for context continuity
- **VLM Data Integration**: Tables/charts formatted as readable text

### 4. Vector Search & RAG
- **ChromaDB Vector Store**: Fast semantic search with persistence
- **Sentence Transformers**: SOTA embedding models
- **Metadata Filtering**: Filter by paper name, region type, etc.
- **Cross-Paper Synthesis**: Retrieve across all indexed documents
- **Diversity Sampling**: MMR-based diversification for varied results

### 5. LLM Orchestration
- **LangChain Integration**: Structured prompting and chain management
- **OpenRouter Backend**: Access to multiple LLM providers
- **Evidence-Based Answering**: Strict grounding in retrieved passages
- **Citation Generation**: Automatic source attribution
- **Multi-Hop Reasoning**: Complex questions requiring multiple steps

### 6. Streamlit UI
- **Chat Interface**: Natural conversation flow with message history
- **Evidence Display**: Expandable passages with source information
- **Advanced Filters**: Paper selection, region type, retrieval parameters
- **Retrieval Statistics**: Transparency into search results
- **Real-time Processing**: Immediate feedback and streaming responses

## Technical Architecture

### Processing Pipeline
```
PDF → OCR (PaddleOCR) → Layout (LayoutParser) → Reading Order →
VLM (Tables/Charts) → Chunking → Embeddings → ChromaDB
```

### Query Pipeline
```
User Query → Embedding → Vector Search (ChromaDB) →
RAG Retrieval → Context Assembly → LLM (OpenRouter) →
Answer + Evidence → UI Display
```

## Technology Stack

### Core Framework
- **Python 3.9+**: Main programming language
- **Streamlit**: Web application framework

### Document Processing
- **PaddleOCR**: OCR engine with GPU support
- **LayoutParser + Detectron2**: Layout analysis
- **pdf2image**: PDF to image conversion
- **Pillow**: Image processing

### Machine Learning
- **Sentence Transformers**: Text embeddings
- **ChromaDB**: Vector database
- **LangChain**: LLM orchestration
- **OpenRouter**: LLM API gateway

### VLM & LLM
- **OpenRouter API**: Access to GPT-4, Claude, Qwen-VL, etc.
- **Configurable Models**: Easy switching between providers

## Project Structure

```
RAG on Research ADI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── README.md                       # Comprehensive documentation
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_SUMMARY.md              # This file
│
├── Data/                           # PDF input directory (15 papers)
│
├── data/
│   ├── processed/                  # Processed JSON documents
│   └── embeddings/                 # (Reserved)
│
├── chroma_db/                      # Vector store persistence
├── logs/                           # Application logs
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   │
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py          # PaddleOCR wrapper
│   │   ├── layout_detector.py     # LayoutParser wrapper
│   │   ├── reading_order.py       # Reading order detection
│   │   ├── vlm_extractor.py       # VLM for tables/charts
│   │   ├── document_processor.py  # Main pipeline orchestrator
│   │   └── chunker.py             # Semantic chunking
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   └── rag_retriever.py       # RAG retrieval layer
│   │
│   ├── llm_orchestration/
│   │   ├── __init__.py
│   │   └── answer_engine.py       # LangChain + OpenRouter
│   │
│   └── ui/
│       └── __init__.py             # UI components
│
└── scripts/
    ├── setup.sh                    # Automated setup script
    ├── process_documents.py        # Document processing CLI
    └── test_installation.py        # Installation verification
```

## Implementation Highlights

### 1. OCR Engine (ocr_engine.py)
- Configurable language and GPU support
- Confidence-based filtering
- Page-by-page processing with bounding boxes
- Text extraction by region

### 2. Layout Detector (layout_detector.py)
- Detectron2-based model (PubLayNet)
- Confidence thresholding
- Region type classification
- IoU calculations for region overlap

### 3. Reading Order Detection (reading_order.py)
- Column detection via horizontal clustering
- Type-based priority (title > text > table > figure)
- Top-to-bottom, left-to-right ordering
- Multi-column support

### 4. VLM Extractor (vlm_extractor.py)
- OpenRouter API integration
- Structured prompts for tables and charts
- JSON response parsing
- Error handling and fallbacks

### 5. Document Processor (document_processor.py)
- Orchestrates all processing steps
- Merges OCR with layout regions
- JSON serialization for persistence
- Handles images and PDF conversion

### 6. Semantic Chunker (chunker.py)
- Token-based chunking with overlap
- Respects region boundaries
- VLM data formatting
- Metadata preservation

### 7. Vector Store (vector_store.py)
- ChromaDB persistence
- Batch embedding generation
- Metadata filtering
- Statistics and analytics

### 8. RAG Retriever (rag_retriever.py)
- Cross-paper synthesis
- Diversity sampling (MMR)
- Evidence grouping by paper/region type
- Context formatting for LLM

### 9. Answer Engine (answer_engine.py)
- Evidence-based answer synthesis
- Citation extraction
- Multi-hop reasoning support
- Deterministic outputs

### 10. Streamlit App (app.py)
- Chat interface with history
- Sidebar filters and controls
- Evidence display with expandable sections
- Real-time statistics

## Configuration

### Environment Variables (.env)
```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

LLM_MODEL=openai/gpt-4-turbo-preview
VLM_MODEL=qwen/qwen-vl-max
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

CHROMA_PERSIST_DIR=./chroma_db
MAX_WORKERS=4
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=10
```

### Customizable Parameters
- Chunk size and overlap
- OCR confidence threshold
- Layout confidence threshold
- Number of retrieval results
- Diversity factor
- Embedding model
- LLM model

## Usage Workflow

### 1. Setup
```bash
./scripts/setup.sh
# Edit .env with API key
```

### 2. Process Documents
```bash
# With VLM extraction
python scripts/process_documents.py

# Without VLM (faster)
python scripts/process_documents.py --no-vlm

# Clear and reprocess
python scripts/process_documents.py --clear
```

### 3. Launch Application
```bash
streamlit run app.py
```

### 4. Query Papers
- Use natural language questions
- Apply filters (papers, region types)
- Adjust retrieval parameters
- Enable multi-hop reasoning for complex queries

## Performance Characteristics

### Processing Time (per paper)
- **Without VLM**: 2-3 minutes
- **With VLM**: 3-5 minutes (depends on table/chart count)

### Query Response Time
- **Vector Search**: <1 second
- **LLM Generation**: 2-5 seconds (depends on model and context size)
- **Total**: 3-6 seconds per query

### Memory Requirements
- **Processing**: 8-16GB RAM (GPU recommended)
- **Runtime**: 4-8GB RAM
- **Vector Store**: ~100MB per paper

## Cost Analysis

### One-Time Processing
- **VLM Extraction**: $0.15-0.30 for 15 papers (~$0.01-0.02 per table/chart)
- **Without VLM**: Free

### Per-Query Costs
- **GPT-4 Turbo**: $0.02-0.04 per query
- **GPT-3.5 Turbo**: $0.002-0.005 per query
- **Claude 3 Haiku**: $0.003-0.008 per query

### Cost Optimization
- Use `--no-vlm` for processing
- Reduce `TOP_K_RETRIEVAL`
- Use cheaper models
- Cache frequently asked questions

## Limitations & Considerations

1. **PDF Quality**: OCR accuracy depends on document quality
2. **VLM Accuracy**: Table/chart extraction may have errors
3. **Context Length**: Very long papers may exceed LLM limits
4. **API Costs**: VLM and LLM calls incur charges
5. **GPU Requirement**: Recommended for faster processing
6. **English Only**: Currently optimized for English papers

## Future Enhancements

### Short-term
- [ ] Batch question processing
- [ ] Export functionality (PDF, Markdown)
- [ ] Citation graph visualization
- [ ] Query history persistence

### Medium-term
- [ ] Multi-language support
- [ ] Custom embedding fine-tuning
- [ ] Integration with Zotero/Mendeley
- [ ] Advanced filtering (date ranges, authors)

### Long-term
- [ ] Support for more document formats
- [ ] Cross-modal search (images, equations)
- [ ] Collaborative features (shared annotations)
- [ ] API endpoint for programmatic access

## Testing

### Installation Test
```bash
python scripts/test_installation.py
```

Verifies:
- Package imports
- Configuration
- Directory structure
- PDF file detection
- Component initialization

## Deployment Considerations

### Local Deployment
- Suitable for personal use
- GPU recommended
- API key required

### Server Deployment
- Use Docker for containerization
- Configure reverse proxy (nginx)
- Set up SSL/TLS
- Implement rate limiting
- Monitor API usage

### Cloud Deployment
- AWS/GCP/Azure compatible
- Use managed vector databases (Pinecone, Weaviate)
- Implement authentication
- Set up monitoring and logging

## Maintenance

### Regular Tasks
- Monitor API costs
- Update embedding models
- Clean up old processed documents
- Backup vector store
- Update dependencies

### Troubleshooting
- Check logs in `logs/` directory
- Verify API key validity
- Ensure sufficient disk space
- Monitor memory usage
- Test with smaller document sets

## Documentation

- **README.md**: Comprehensive user guide
- **QUICKSTART.md**: 5-minute setup guide
- **PROJECT_SUMMARY.md**: Technical overview (this file)
- **Code Comments**: Inline documentation in all modules
- **Type Hints**: Full type annotations for better IDE support

## Support & Contribution

### Getting Help
1. Check documentation
2. Review logs for errors
3. Test with `test_installation.py`
4. Search existing issues

### Contributing
- Follow existing code style
- Add tests for new features
- Update documentation
- Submit pull requests

## License

Research and educational purposes.

## Acknowledgments

Special thanks to the open-source community for:
- PaddleOCR (OCR engine)
- LayoutParser (document layout analysis)
- ChromaDB (vector database)
- LangChain (LLM orchestration)
- Streamlit (web framework)
- OpenRouter (LLM API gateway)

---

**Project Status**: Production-ready for research use cases
**Last Updated**: 2026-01-19
**Version**: 1.0.0

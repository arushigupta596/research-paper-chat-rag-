# Research Paper Chat Assistant

A Streamlit-based document understanding and evidence-backed chat system that enables users to chat with research papers using advanced RAG techniques with layout-aware document processing.

## Features

### Document Processing Pipeline
- **Tesseract OCR**: Extract text with bounding boxes from PDF pages
- **LayoutParser**: Detect document regions (text, tables, charts, figures, titles, lists)
- **Reading Order Detection**: Determine correct reading flow across multi-column layouts
- **VLM-based Extraction**: Extract structured data from tables and charts using Qwen2-VL Vision-Language Model
- **Poppler Integration**: High-quality PDF-to-image conversion for VLM processing

### Knowledge Representation
- **Semantic Chunking**: Respect document structure and reading order
- **Metadata Enrichment**: Paper name, page number, region type, bounding boxes, paper topics
- **ChromaDB Vector Store**: Fast semantic search with metadata filtering
- **Paper Metadata System**: Track paper titles, topics, and keywords for contextual referencing

### RAG & Reasoning
- **Cross-paper Synthesis**: Retrieve and synthesize information across all papers
- **Evidence-backed Answers**: Natural conversational answers with inline evidence citations
- **Multi-hop Reasoning**: Complex questions requiring multiple reasoning steps
- **LangChain + OpenRouter**: Orchestrated LLM reasoning with Qwen2.5-72B-Instruct
- **Paper-aware Context**: Answers include paper topics and contextual information

### User Interface
- **Streamlit Chat Interface**: Natural language conversations with persistent chat history
- **Evidence Display**: Clear distinction between evidence passages and paper sources
- **Retrieval Statistics**: Transparency into search results by paper and region type
- **Smart Filters**: Filter by papers and region types
- **Sample Questions**: 30 curated questions showcasing different content types (tables, figures, text, etc.)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                            │
│  - Chat Interface    - Evidence Display    - Sample Questions       │
│  - Paper Filters     - Region Type Filters - Retrieval Stats        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Answer Engine (LangChain + OpenRouter)                 │
│  - Natural Answer Generation  - Inline Evidence Citations           │
│  - Paper Topic Integration    - Multi-hop Reasoning                 │
│  - Qwen2.5-72B-Instruct LLM   - Paper Metadata Loading              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG Retriever (rag_retriever.py)                 │
│  - Semantic Search           - Cross-paper Retrieval                │
│  - Metadata Filtering         - Evidence Formatting                 │
│  - Paper & Region Filters     - Score-based Ranking                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Vector Store (ChromaDB - chroma_db/)                   │
│  - 876 Embedded Chunks        - all-mpnet-base-v2 Embeddings        │
│  - Metadata: paper, page, region_type, bbox, reading_order          │
│  - Fast Similarity Search     - Persistent Storage                  │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│         Document Processing Pipeline (process_documents.py)         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ PDF → Images │→ │ Tesseract    │→ │   LayoutParser          │  │
│  │  (Poppler)   │  │ OCR Engine   │  │  (Region Detection)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                                                  │                  │
│                                                  ▼                  │
│                                   ┌──────────────────────────┐      │
│                                   │   Reading Order          │      │
│                                   │   Detection              │      │
│                                   └──────────────────────────┘      │
│                                                  │                  │
│              ┌───────────────────────────────────┴─────────┐        │
│              ▼                                             ▼        │
│  ┌──────────────────────┐                    ┌──────────────────┐  │
│  │  VLM Extractor       │                    │ Text Regions     │  │
│  │  (Qwen2-VL)          │                    │ (text, title,    │  │
│  │  - Tables → JSON     │                    │  list)           │  │
│  │  - Charts → JSON     │                    └──────────────────┘  │
│  │  - Rate Limit Retry  │                             │            │
│  └──────────────────────┘                             │            │
│              │                                         │            │
│              └─────────────────┬─────────────────────┘            │
│                                ▼                                    │
│                  ┌──────────────────────────────┐                   │
│                  │   Semantic Chunker           │                   │
│                  │   - Structure-aware Chunking │                   │
│                  │   - Metadata Enrichment      │                   │
│                  │   - Reading Order Preserved  │                   │
│                  └──────────────────────────────┘                   │
│                                │                                    │
│                                ▼                                    │
│                  Processed JSON Files (data/processed/)             │
│                  + Paper Metadata (data/paper_metadata.json)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Processing (Offline)**
   - PDFs converted to images using Poppler
   - Tesseract OCR extracts text with bounding boxes
   - LayoutParser detects regions (text, table, figure, title, list)
   - Reading order algorithm determines correct sequence
   - VLM extracts structured data from tables/charts (with retry logic)
   - Chunker creates semantic chunks preserving structure
   - Chunks embedded and stored in ChromaDB

2. **Query Processing (Runtime)**
   - User asks question via Streamlit UI
   - RAG Retriever searches ChromaDB with filters
   - Top-K relevant chunks retrieved with metadata
   - Answer Engine loads paper metadata
   - LLM generates natural answer with inline citations
   - Evidence and sources displayed with paper topics

## Installation

### Prerequisites
- Python 3.9+
- **Poppler** (required for PDF-to-image conversion)
- **Tesseract OCR** (for text extraction)
- 8GB+ RAM recommended
- OpenRouter API key

### Setup

1. **Install System Dependencies**

   **macOS:**
   ```bash
   # Install Poppler and Tesseract
   brew install poppler tesseract
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install poppler-utils tesseract-ocr
   ```

   **Windows:**
   - Install Poppler: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Install Tesseract: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add both to system PATH

2. **Clone and Setup Project**
   ```bash
   cd "RAG on Research ADI"
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: Detectron2 (for LayoutParser) requires special installation:
   ```bash
   # For CUDA 11.8
   pip install 'git+https://github.com/facebookresearch/detectron2.git'

   # For CPU only (macOS/Linux)
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

   ⚠️ **Security Note**: Never commit your `.env` file to Git! It's already in `.gitignore` for protection.

### Verify Installation

Test that all dependencies are installed correctly:
```bash
# Check Poppler
pdftoppm -h

# Check Tesseract
tesseract --version

# Test Python imports
python -c "import layoutparser, chromadb, langchain; print('All imports successful')"
```

## Usage

### 1. Process Documents

Place your PDF research papers in the `Data/` directory, then run:

```bash
python scripts/process_documents.py
```

Options:
```bash
# Skip VLM extraction (faster but no table/chart understanding)
python scripts/process_documents.py --no-vlm

# Use Tesseract instead of PaddleOCR (recommended - more stable)
python scripts/process_documents.py --use-tesseract

# Clear existing vector store before processing
python scripts/process_documents.py --clear

# Specify custom data directory
python scripts/process_documents.py --data-dir /path/to/pdfs

# Limit number of documents to process
python scripts/process_documents.py --limit 5
```

**What happens during processing:**
1. PDFs converted to images (Poppler at 200 DPI)
2. Text extracted with Tesseract OCR + bounding boxes
3. Layout regions detected (LayoutParser with Detectron2)
4. Reading order determined across multi-column layouts
5. VLM extracts structured data from tables/charts (Qwen2-VL)
6. Semantic chunks created preserving document structure
7. Chunks embedded and indexed in ChromaDB

**Processing Time:** ~4-5 minutes per paper
- Depends on: number of pages, tables/figures, VLM API rate limits
- VLM uses exponential backoff retry for rate limits (2s, 4s, 8s)

**Output:**
- `data/processed/*.json`: Processed document files
- `data/processed/*.chunks.json`: Chunk files
- `chroma_db/`: Vector store with embedded chunks
- `data/paper_metadata.json`: Paper titles and topics

### 2. (Optional) Reprocess VLM Extractions

If you already processed documents but want to re-extract tables/figures:

```bash
python scripts/reprocess_vlm.py
```

This will update existing processed files with new VLM extractions without re-running OCR and layout detection.

### 3. Launch Chat Interface

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 4. Chat with Papers

**Sample Questions** (30 curated questions available in sidebar):

**Table Data Questions:**
- "Show me experimental results comparing different models or methods in tables."
- "What performance metrics and benchmarks are reported in the tables?"

**Figure & Chart Questions:**
- "What trends or patterns are shown in the charts and graphs?"
- "Describe the visualizations and their key insights from the figures."

**Methodology Questions:**
- "What are the main methodologies and approaches described?"
- "How were the experiments designed and conducted?"

**Comprehensive Questions:**
- "What are the key findings from both quantitative results and qualitative analysis?"
- "Summarize the main contributions combining text, tables, and figures."

**Technical Details:**
- "What hyperparameters, configurations, or settings are specified?"
- "What are the model architectures and their component details?"

## Features Guide

### Search Filters (Sidebar)

**Filter by Papers**: Select specific papers to search within
- Leave empty to search all indexed papers
- Useful for comparing specific papers or focusing on a subset

**Filter by Region Type**: Focus on specific content types:
- `text`: Regular text passages and paragraphs
- `table`: Data tables (VLM-extracted structured data)
- `figure`: Charts, graphs, and diagrams (VLM-extracted)
- `title`: Section and subsection titles
- `list`: Bullet points, numbered lists, and enumerations

**Number of Evidence Chunks**: Control retrieval (3-20)
- More chunks = more context but slower generation
- Default: 10 chunks

### Sample Questions (Sidebar)

**6 Categories of Questions:**
1. **Table Data Questions**: Showcase VLM table extractions
2. **Figure & Chart Questions**: Showcase VLM figure/chart extractions
3. **Methodology & Text Questions**: Focus on textual content
4. **Comprehensive Questions**: Cross-region and multi-document retrieval
5. **Technical Details**: Mixed region types
6. **Structural Content**: Lists and titles

Click any question to automatically ask it.

### Answer Display

Each answer includes:

1. **Natural Answer**: Conversational response with inline citations `[Evidence N]`
   - Includes paper topics when mentioning findings
   - Synthesizes information across multiple papers
   - Clear, direct style addressing the question

2. **Retrieval Statistics** (expandable):
   - Total evidence chunks retrieved
   - Number of papers searched
   - Region types found
   - Chunk distribution by paper

3. **Key Evidence** (expandable):
   - Full text of each evidence passage
   - Source: paper name, page number, region type
   - Relevance score (0-1)
   - Direct excerpts from papers

4. **Research Paper Sources**:
   - Paper title with topic
   - Page numbers referenced
   - Unique list of all papers cited

### Chat History

- Persistent within session
- View all previous Q&A
- Clear chat button in sidebar

## Project Structure

```
.
├── app.py                      # Streamlit chat application
├── requirements.txt            # Python dependencies
├── .env                        # Configuration (create from .env.example)
├── README.md                   # This file
│
├── Data/                       # PDF documents (your research papers)
│
├── data/
│   ├── processed/              # Processed document JSON files
│   │   ├── *.json              # Document structure, regions, VLM data
│   │   └── *.chunks.json       # Chunked data ready for embedding
│   └── paper_metadata.json     # Paper titles, topics, keywords
│
├── chroma_db/                  # ChromaDB vector store (876 chunks)
│   └── (persistent ChromaDB files)
│
├── logs/                       # Application logs
│   └── app.log                 # Runtime logs
│
├── src/
│   ├── config.py               # Configuration management
│   │
│   ├── document_processing/
│   │   ├── ocr_engine.py       # Tesseract OCR wrapper
│   │   ├── layout_detector.py  # LayoutParser + Detectron2
│   │   ├── reading_order.py    # Reading order detection algorithm
│   │   ├── vlm_extractor.py    # Qwen2-VL extractor (tables/charts)
│   │   ├── document_processor.py # Main processing pipeline
│   │   └── chunker.py          # Semantic chunking with structure
│   │
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB wrapper with metadata
│   │   └── rag_retriever.py    # RAG retrieval with filtering
│   │
│   └── llm_orchestration/
│       └── answer_engine.py    # LangChain + OpenRouter (Qwen2.5-72B)
│
└── scripts/
    ├── process_documents.py    # Main document processing script
    ├── reprocess_vlm.py        # Re-extract VLM data only
    ├── test_vlm_fix.py         # Test VLM extraction
    └── test_rag.py             # Test RAG retrieval
```

### Key Files

**`app.py`**: Streamlit UI with chat interface, filters, and sample questions

**`src/llm_orchestration/answer_engine.py`**:
- Natural answer generation with inline citations
- Paper metadata integration
- Multi-hop reasoning support

**`src/retrieval/rag_retriever.py`**:
- Semantic search over ChromaDB
- Paper and region type filtering
- Evidence formatting with metadata

**`src/document_processing/vlm_extractor.py`**:
- Qwen2-VL API integration
- Structured table/chart extraction
- Retry logic with exponential backoff

**`data/paper_metadata.json`**:
- Paper titles, topics, and keywords
- Used for contextual referencing in answers

## Configuration

Edit `src/config.py` or `.env` file to customize:

### API Keys
```python
OPENROUTER_API_KEY = "your_api_key_here"
```

### Paths
- `DATA_DIR`: Directory containing PDF files (default: `Data/`)
- `PROCESSED_DATA_DIR`: Output directory for processed documents (default: `data/processed/`)
- `CHROMA_PERSIST_DIR`: Vector store persistence directory (default: `chroma_db/`)
- `LOGS_DIR`: Application logs directory (default: `logs/`)

### Models

**LLM for Answer Synthesis:**
```python
LLM_MODEL = "qwen/qwen-2.5-72b-instruct"  # Main reasoning model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

**Vision-Language Model (VLM):**
```python
VLM_MODEL = "qwen/qwen-2-vl-7b-instruct"  # For table/chart extraction
```

**Embedding Model:**
```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # For ChromaDB
```

### Processing Parameters
- `CHUNK_SIZE`: Maximum chunk size in tokens (default: `512`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)
- `TOP_K_RETRIEVAL`: Default number of chunks to retrieve (default: `10`)
- `OCR_CONFIDENCE_THRESHOLD`: Minimum OCR confidence (default: `0.5`)
- `LAYOUT_CONFIDENCE_THRESHOLD`: Minimum layout detection confidence (default: `0.6`)
- `OCR_USE_GPU`: Use GPU for OCR (default: `False` for Tesseract)

### Model Selection

**Alternative LLMs** (change in `.env`):
- `qwen/qwen-2.5-72b-instruct` (default - best quality)
- `openai/gpt-4-turbo-preview` (OpenAI GPT-4)
- `anthropic/claude-3-sonnet` (Claude 3)
- `google/gemini-pro-1.5` (Google Gemini)

**Alternative VLMs**:
- `qwen/qwen-2-vl-7b-instruct` (default - good balance)
- `qwen/qwen-vl-max` (better quality, higher cost)
- `anthropic/claude-3-opus` (vision-capable)

## Performance Optimization

### For Faster Processing
- **Skip VLM**: Use `--no-vlm` flag (sacrifices table/chart understanding)
- **Limit Documents**: Use `--limit N` to process fewer documents
- **Reduce Chunk Size**: Lower `CHUNK_SIZE` in config for faster embedding
- **Reduce Retrieval**: Lower `TOP_K_RETRIEVAL` to retrieve fewer chunks

### For Better Accuracy
- **More Evidence**: Increase `TOP_K_RETRIEVAL` (10-20)
- **More Sensitive OCR**: Lower `OCR_CONFIDENCE_THRESHOLD` (0.3-0.5)
- **Better VLM**: Use `qwen/qwen-vl-max` instead of `qwen-2-vl-7b-instruct`
- **Better LLM**: Use `openai/gpt-4-turbo-preview` or `anthropic/claude-3-opus`

### Cost Optimization
- **Use Tesseract**: Already default (free OCR vs. PaddleOCR)
- **Skip VLM**: `--no-vlm` saves ~$0.01-0.02 per table/chart
- **Cheaper LLM**: Use `qwen/qwen-2.5-7b-instruct` instead of 72B
- **Reduce Retrieval**: Lower `TOP_K_RETRIEVAL` reduces LLM context size

## Troubleshooting

### Installation Issues

**Poppler not found:**
```bash
# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils
```

**Tesseract not found:**
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr
```

**Detectron2 fails to install:**
```bash
# Try direct GitHub installation
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Or CPU-only version
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

### Processing Issues

**`pdftoppm not found` error:**
- Install Poppler (see Installation section above)
- Ensure `pdftoppm` is in system PATH

**VLM extraction 100% failure:**
- Check OpenRouter API key in `.env`
- Verify VLM model is available: `qwen/qwen-2-vl-7b-instruct`
- Check API rate limits (script has retry logic but may fail if persistent)
- Check OpenRouter account has credits

**Out of memory during processing:**
- Process fewer documents: `--limit 5`
- Disable VLM: `--no-vlm`
- Close other applications
- Reduce image DPI in config (default: 200)

**JSON parsing errors in VLM extraction:**
- Already handled with `_clean_json_response` method
- Check `logs/app.log` for specific errors
- May indicate VLM model issues - try different VLM model

### Runtime Issues

**Streamlit shows "No papers indexed":**
- Verify `chroma_db/` directory exists and has data
- Check sidebar shows chunk count > 0
- Run `python scripts/process_documents.py` first

**No results found for queries:**
- Check filters - clear paper/region filters
- Try broader search terms
- Verify ChromaDB has indexed chunks (check sidebar)

**Slow answer generation:**
- Reduce `TOP_K_RETRIEVAL` in sidebar slider
- Use faster LLM model in config
- Check internet connection (OpenRouter API calls)

**LangChain import errors:**
- Upgrade LangChain packages:
  ```bash
  pip install --upgrade langchain langchain-core langchain-openai langchain-community
  ```

## API Costs

### OpenRouter Pricing (Approximate)

**Document Processing** (one-time per paper):
- VLM extraction: ~$0.005-0.02 per table/chart
- Example: 10 papers with 15 tables/charts each = ~$0.75-3.00 total

**Query Runtime** (per question):
- Qwen-2.5-72B-Instruct: ~$0.01-0.02 per question
- GPT-4-Turbo: ~$0.05-0.10 per question
- Claude-3-Sonnet: ~$0.03-0.08 per question

**Typical Session:**
- Process 10 papers: ~$1-3 (one-time)
- 20 questions: ~$0.20-0.40 (Qwen) or ~$1-2 (GPT-4)
- **Total: ~$1.20-5.00** for complete session

### Cost Optimization
- **Use Tesseract**: Free OCR (already default)
- **Skip VLM**: `--no-vlm` flag (no table/chart extraction cost)
- **Cheaper LLM**: Use Qwen-2.5-7B instead of 72B
- **Reduce Retrieval**: Lower `TOP_K_RETRIEVAL` to reduce context size
- **Cache Answers**: Reuse answers from chat history

## Current System Stats

Based on processed data:
- **Total Papers**: 10
- **Total Chunks**: 876
- **VLM Extractions**: 157 (88 tables + 69 figures)
- **Success Rate**: 100%
- **Region Types**: Text (473), Title (124), List (122), Table (88), Figure (69)
- **Embedding Model**: all-mpnet-base-v2 (768-dim)
- **Vector Store**: ChromaDB (persistent)

## Limitations

1. **PDF Quality**: OCR accuracy depends on PDF quality (native text vs. scanned)
2. **VLM Accuracy**: Table/chart extraction depends on VLM model capabilities
3. **Layout Complexity**: Very complex multi-column layouts may have reading order errors
4. **Language**: Currently optimized for English papers only
5. **Mathematical Formulas**: Complex equations may not be extracted accurately
6. **Context Window**: Very long papers may require chunking strategies
7. **API Rate Limits**: VLM extraction has retry logic but may slow down with limits

## Future Enhancements

**Document Processing:**
- [ ] Support for more document formats (DOCX, HTML, LaTeX)
- [ ] Better mathematical formula extraction (MathPix, LaTeX OCR)
- [ ] Multi-language support (multilingual embeddings)
- [ ] Image caption generation for figures

**RAG & Retrieval:**
- [ ] Hybrid search (dense + sparse/BM25)
- [ ] Re-ranking with cross-encoders
- [ ] Query expansion and reformulation
- [ ] Citation graph awareness

**User Experience:**
- [ ] Export answers to PDF/Markdown
- [ ] Batch question processing
- [ ] Answer caching for sample questions
- [ ] Citation graph visualization
- [ ] Integration with reference managers (Zotero, Mendeley)
- [ ] Question suggestion based on document content

## License

This project is for research and educational purposes.

## Acknowledgments

- **Tesseract**: OCR engine
- **Poppler**: PDF rendering
- **LayoutParser**: Document layout analysis
- **Detectron2**: Layout detection model
- **ChromaDB**: Vector database
- **LangChain**: LLM orchestration framework
- **Streamlit**: UI framework
- **OpenRouter**: LLM API gateway
- **Qwen Team**: Qwen2-VL and Qwen2.5 models

## Support

For issues, questions, or contributions:
- Check `logs/app.log` for detailed error messages
- Review this README's Troubleshooting section
- Open an issue with error details and system info

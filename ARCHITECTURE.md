# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│                      (Streamlit Web App)                            │
│  - Chat Interface        - Evidence Display                         │
│  - Filters & Controls    - Statistics Dashboard                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ANSWER ENGINE (LangChain)                        │
│                                                                     │
│  ┌──────────────────┐    ┌─────────────────┐   ┌────────────────┐   │
│  │ Query Processing │───▶│Answer Synthesis │──▶│ Citation       │   │
│  │                  │    │                 │   │ Extraction     │   │
│  └──────────────────┘    └─────────────────┘   └────────────────┘   │
│           │                       │                      │          │
│           │                       ▼                      │          │
│           │           ┌──────────────────────┐          │           │
│           └──────────▶│  OpenRouter LLM API  │◀─────────┘           │
│                       │  (Claude)│                      │
│                       └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG RETRIEVAL LAYER                            │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Vector Search Engine                         ││
│  │                                                                  │ │
│  │  Query Embedding ───▶ ChromaDB Search ───▶ Result Ranking      │ │
│  │       │                     │                      │             │ │
│  │       ▼                     ▼                      ▼             │ │
│  │  [SentenceTransformer] [Similarity]  [Diversity Sampling]       ││
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   Evidence Processing                          │ │
│  │                                                                  │ │
│  │  - Cross-paper synthesis  - Metadata filtering                 │ │
│  │  - Source grouping        - Context assembly                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VECTOR STORE (ChromaDB)                       │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  Embeddings  │  │   Metadata   │  │   Documents  │               │
│  │              │  │              │  │              │               │
│  │ - Dense      │  │ - Paper name │  │ - Text       │               │
│  │   vectors    │  │ - Page num   │  │   chunks     │               │ 
│  │ - HNSW       │  │ - Region type│  │ - Tables     │               │
│  │   index      │  │ - Bbox       │  │ - Charts     │               │ 
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                   DOCUMENT PROCESSING PIPELINE                       │
│                                                                       │
│  Input: PDF Files                                                    │
│     │                                                                 │
│     ▼                                                                 │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              PDF to Image Conversion                  │           │
│  │                 (pdf2image)                           │           │
│  └──────────────────────────────────────────────────────┘           │
│     │                                                                 │
│     ▼                                                                 │
│  ┌──────────────────────────────────────────────────────┐           │
│  │             OCR Text Extraction                       │           │
│  │               (PaddleOCR)                             │           │
│  │                                                       │           │
│  │  Output: Text + Bounding Boxes + Confidence          │           │
│  └──────────────────────────────────────────────────────┘           │
│     │                                                                 │
│     ▼                                                                 │
│  ┌──────────────────────────────────────────────────────┐           │
│  │           Layout Region Detection                     │           │
│  │             (LayoutParser)                            │           │
│  │                                                       │           │
│  │  Output: Regions (text, table, figure, title, list)  │           │
│  └──────────────────────────────────────────────────────┘           │
│     │                                                                 │
│     ├──────────────────────────┬────────────────────────┐           │
│     ▼                          ▼                        ▼           │
│  ┌──────────┐        ┌────────────────┐     ┌──────────────────┐   │
│  │ Reading  │        │  VLM Table     │     │  VLM Chart       │   │
│  │  Order   │        │  Extraction    │     │  Extraction      │   │
│  │Detection │        │                │     │                  │   │
│  │          │        │  (OpenRouter)  │     │  (OpenRouter)    │   │
│  │- Columns │        │                │     │                  │   │
│  │- Top→Bot │        │Output: Headers,│     │Output: Type,     │   │
│  │- Priority│        │ Rows, Footnotes│     │ Axes, Trends     │   │
│  └──────────┘        └────────────────┘     └──────────────────┘   │
│     │                          │                        │           │
│     └──────────────────────────┴────────────────────────┘           │
│                                │                                     │
│                                ▼                                     │
│              ┌─────────────────────────────────┐                    │
│              │      Region Text Assembly       │                    │
│              │  (Merge OCR + VLM + Layout)     │                    │
│              └─────────────────────────────────┘                    │
│                                │                                     │
│                                ▼                                     │
│              ┌─────────────────────────────────┐                    │
│              │     Semantic Chunking           │                    │
│              │                                 │                    │
│              │  - Respect region boundaries    │                    │
│              │  - Token-based splitting        │                    │
│              │  - Overlap for context          │                    │
│              │  - Metadata enrichment          │                    │
│              └─────────────────────────────────┘                    │
│                                │                                     │
│                                ▼                                     │
│              ┌─────────────────────────────────┐                    │
│              │      Embedding Generation       │                    │
│              │   (Sentence Transformers)       │                    │
│              └─────────────────────────────────┘                    │
│                                │                                     │
│                                ▼                                     │
│                    ┌────────────────────┐                           │
│                    │   Vector Store     │                           │
│                    │    Indexing        │                           │
│                    └────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Document Ingestion Flow

```
PDF File
  │
  ├──▶ pdf2image ──▶ [Page Images]
  │                        │
  │                        ├──▶ PaddleOCR ──▶ [OCR Results]
  │                        │                      │
  │                        │                      ├─ Text
  │                        │                      ├─ Bounding Boxes
  │                        │                      └─ Confidence
  │                        │
  │                        └──▶ LayoutParser ──▶ [Layout Regions]
  │                                                   │
  │                                                   ├─ Text blocks
  │                                                   ├─ Tables
  │                                                   ├─ Figures
  │                                                   ├─ Titles
  │                                                   └─ Lists
  │
  ├──▶ Merge OCR + Layout ──▶ [Enriched Regions]
  │                                │
  │                                ├─ Text content
  │                                ├─ Region type
  │                                ├─ Position (bbox)
  │                                └─ Confidence
  │
  ├──▶ Reading Order Detection ──▶ [Ordered Regions]
  │
  ├──▶ VLM Processing (tables/charts) ──▶ [Structured Data]
  │                                              │
  │                                              ├─ Table: headers, rows
  │                                              └─ Chart: type, axes, trends
  │
  ├──▶ Semantic Chunking ──▶ [Chunks with Metadata]
  │                              │
  │                              ├─ Paper name
  │                              ├─ Page number
  │                              ├─ Region ID/type
  │                              ├─ Reading order
  │                              └─ Bounding box
  │
  └──▶ Embedding Generation ──▶ [Vector Embeddings]
                                     │
                                     └──▶ ChromaDB Index
```

### 2. Query Processing Flow

```
User Query
  │
  ├──▶ Query Embedding (SentenceTransformer)
  │         │
  │         └──▶ Vector Search (ChromaDB)
  │                   │
  │                   ├─ Similarity matching
  │                   ├─ Metadata filtering
  │                   └─ Top-K selection
  │                         │
  │                         └──▶ [Retrieved Chunks]
  │                                   │
  │                                   ├─ Chunk text
  │                                   ├─ Source metadata
  │                                   └─ Relevance score
  │
  ├──▶ Diversity Sampling (MMR)
  │         │
  │         └──▶ [Diversified Results]
  │
  ├──▶ Evidence Grouping
  │         │
  │         ├─ By paper
  │         └─ By region type
  │
  ├──▶ Context Assembly
  │         │
  │         └──▶ [Formatted Context]
  │                   │
  │                   ├─ Evidence passages
  │                   ├─ Source citations
  │                   └─ Metadata
  │
  └──▶ LLM Processing (OpenRouter)
          │
          ├─ System prompt (evidence-based answering)
          ├─ Context (retrieved passages)
          └─ User query
              │
              └──▶ [Generated Answer]
                      │
                      ├─ Synthesized text
                      ├─ Evidence citations [Evidence N]
                      └─ Source list
                          │
                          └──▶ UI Display
```

## Component Interactions

### Document Processing Components

```
┌────────────────────┐
│  DocumentProcessor │ (Orchestrator)
└────────────────────┘
         │
         ├──▶ OCREngine
         │      └─ PaddleOCR library
         │
         ├──▶ LayoutDetector
         │      └─ LayoutParser + Detectron2
         │
         ├──▶ ReadingOrderDetector
         │      └─ Column detection + sorting
         │
         ├──▶ VLMExtractor
         │      └─ OpenRouter API
         │
         └──▶ SemanticChunker
                └─ Token-based chunking
```

### Retrieval Components

```
┌────────────────┐
│  AnswerEngine  │ (Query Handler)
└────────────────┘
         │
         ├──▶ RAGRetriever
         │      │
         │      ├─ VectorStore
         │      │    └─ ChromaDB
         │      │
         │      ├─ Evidence grouping
         │      └─ Context formatting
         │
         └──▶ LangChain LLM
                └─ OpenRouter API
```

## Module Dependencies

```
app.py
  │
  ├── src.retrieval.vector_store
  │     └── chromadb
  │     └── sentence_transformers
  │
  ├── src.retrieval.rag_retriever
  │     └── vector_store
  │
  ├── src.llm_orchestration.answer_engine
  │     └── rag_retriever
  │     └── langchain
  │     └── openai (via OpenRouter)
  │
  └── src.config

scripts/process_documents.py
  │
  ├── src.document_processing.document_processor
  │     ├── ocr_engine
  │     │    └── paddleocr
  │     ├── layout_detector
  │     │    └── layoutparser
  │     ├── reading_order
  │     └── vlm_extractor
  │          └── openai (via OpenRouter)
  │
  ├── src.document_processing.chunker
  │
  └── src.retrieval.vector_store
```

## Database Schema

### ChromaDB Collection

```
Collection: "research_papers"
├── Embeddings: [float] vector (dimension: 768 for all-mpnet-base-v2)
├── Documents: [string] chunk text
└── Metadata:
      ├── paper_name: [string]
      ├── page_num: [int]
      ├── region_id: [string]
      ├── region_type: [string] (text|table|figure|title|list)
      ├── reading_order: [int]
      ├── chunk_index: [int]
      └── bbox: [string] "[x1, y1, x2, y2]"
```

### Processed Document JSON

```json
{
  "filename": "paper_name.pdf",
  "num_pages": 10,
  "pages": [
    {
      "page_num": 1,
      "width": 595,
      "height": 842,
      "regions": [
        {
          "region_id": "page1_region0",
          "region_type": "text",
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "text": "extracted text...",
          "reading_order": 0
        }
      ]
    }
  ],
  "ordered_regions": [...],
  "vlm_extractions": {
    "page1_region3": {
      "type": "table",
      "headers": [...],
      "rows": [...],
      "summary": "..."
    }
  },
  "metadata": {
    "total_regions": 50,
    "total_tables": 5,
    "total_figures": 3,
    "vlm_extracted": 8
  }
}
```

## API Integrations

### OpenRouter API

```
POST https://openrouter.ai/api/v1/chat/completions

Headers:
  Authorization: Bearer $OPENROUTER_API_KEY
  Content-Type: application/json

Body (LLM):
{
  "model": "openai/gpt-4-turbo-preview",
  "messages": [...],
  "temperature": 0.1,
  "max_tokens": 2000
}

Body (VLM):
{
  "model": "qwen/qwen-vl-max",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

## Performance Characteristics

### Processing Pipeline
- **OCR**: ~10-30s per page (GPU) / ~30-60s (CPU)
- **Layout Detection**: ~5-15s per page (GPU) / ~15-30s (CPU)
- **VLM Extraction**: ~2-5s per table/chart (API latency)
- **Chunking**: <1s per document
- **Embedding**: ~1-3s per 100 chunks (GPU) / ~5-10s (CPU)

### Query Pipeline
- **Query Embedding**: <100ms
- **Vector Search**: <500ms for 10K chunks
- **Diversity Sampling**: <100ms
- **LLM Generation**: 2-5s (depends on model)
- **Total**: 3-6s per query

### Scalability
- **Documents**: Tested up to 100 papers
- **Chunks**: Efficient up to 100K chunks
- **Concurrent Users**: 5-10 (single Streamlit instance)
- **Memory**: 4-8GB runtime, 8-16GB processing

## Security Considerations

1. **API Keys**: Stored in .env, never committed
2. **User Input**: Sanitized before LLM processing
3. **File Upload**: Not implemented (security consideration)
4. **Data Privacy**: All processing done locally
5. **Vector Store**: No sensitive data in embeddings

## Monitoring & Logging

```
logs/
├── processing.log  # Document processing events
└── app.log         # Application runtime logs

Log Levels:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Failures requiring attention
- DEBUG: Detailed troubleshooting info
```

## Error Handling

### Processing Pipeline
- OCR failures: Skip page with warning
- Layout detection failures: Fallback to OCR-only
- VLM failures: Store error in metadata, continue
- Chunking failures: Log and skip document

### Query Pipeline
- Embedding failures: Return error message
- Vector search failures: Show system error
- LLM failures: Display error with retry option
- Network errors: Retry with exponential backoff

---

**Architecture Version**: 1.0.0
**Last Updated**: 2026-01-19

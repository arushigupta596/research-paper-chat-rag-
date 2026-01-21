# Quick Start Guide

Get up and running with the Research Paper Chat Assistant in 5 minutes.

## Prerequisites

- Python 3.9+
- 16GB RAM (recommended)
- OpenRouter API key ([Get one here](https://openrouter.ai/))

## Installation

### Option 1: Automated Setup (Linux/Mac)

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

## Configuration

Edit `.env` file:

```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional - use defaults or customize
LLM_MODEL=openai/gpt-4-turbo-preview
VLM_MODEL=qwen/qwen-vl-max
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

## Process Documents

Your PDF files should already be in the `Data/` directory (15 papers detected).

### Quick Processing (No VLM)
Faster, but won't extract structured table/chart data:

```bash
python scripts/process_documents.py --no-vlm
```

Expected time: ~2-3 minutes per paper (~30-45 minutes total for 15 papers)

### Full Processing (With VLM)
Slower, but includes table/chart understanding:

```bash
python scripts/process_documents.py
```

Expected time: ~3-5 minutes per paper (~45-75 minutes total for 15 papers)

**Note**: VLM extraction requires OpenRouter API calls and will incur costs (~$0.01-0.02 per table/chart).

## Launch Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## First Query

Try asking:
- "What are the main research topics covered in these papers?"
- "Summarize the key findings about neural networks"
- "Compare the methodologies used in different papers"

## Features at a Glance

### Sidebar Controls
- **Search Filters**: Filter by specific papers or region types
- **Evidence Chunks**: Adjust number of results (3-20)
- **Advanced Options**:
  - Diversity Factor: Balance relevance vs diversity
  - Multi-hop Reasoning: For complex questions

### Answer Display
Each answer includes:
1. **Answer**: Synthesized response with citations
2. **Statistics**: Retrieval transparency
3. **Evidence**: Expandable passages with sources
4. **Sources**: List of cited papers and pages

## Troubleshooting

### "No papers indexed" error
Run the processing script first:
```bash
python scripts/process_documents.py --no-vlm
```

### "OpenRouter API key not found"
Edit `.env` and add your API key:
```bash
OPENROUTER_API_KEY=your_key_here
```

### Slow processing
1. Use `--no-vlm` flag to skip table/chart extraction
2. Ensure GPU is available (check with `nvidia-smi`)
3. Reduce image DPI in `src/document_processing/ocr_engine.py`

### Out of memory
1. Process fewer documents at once
2. Reduce `CHUNK_SIZE` in `.env`
3. Close other applications

## Common Commands

```bash
# Clear existing data and reprocess
python scripts/process_documents.py --clear

# Process with custom data directory
python scripts/process_documents.py --data-dir /path/to/pdfs

# Skip VLM extraction
python scripts/process_documents.py --no-vlm

# Run application
streamlit run app.py

# Check logs
tail -f logs/processing.log  # Processing logs
tail -f logs/app.log        # Application logs
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore advanced filtering options in the sidebar
3. Try multi-hop reasoning for complex questions
4. Experiment with different models in `.env`

## Cost Estimation

### One-time Processing Costs (VLM extraction)
- ~$0.15-0.30 total for 15 papers (assuming ~10-20 tables/charts per paper)

### Per-query Costs
- ~$0.01-0.03 per question (depends on LLM model and context size)

### Cost Optimization
- Use `--no-vlm` to skip VLM extraction (free, but no table/chart understanding)
- Reduce `TOP_K_RETRIEVAL` to minimize context size
- Use cheaper models (e.g., `anthropic/claude-3-haiku`)

## Support

Having issues? Check:
1. Logs in `logs/` directory
2. [README.md](README.md) for detailed troubleshooting
3. GitHub issues for known problems

Happy researching! 📚

# Answer Caching Guide

## Overview

The answer caching system pre-computes and stores answers to all suggested questions in the sidebar. This provides:

- ⚡ **Instant responses** - No waiting for LLM generation
- 💰 **Cost savings** - No API calls for cached questions
- 🎯 **Consistent quality** - Pre-vetted answers
- 🔄 **Easy updates** - Refresh cache when documents change

## How It Works

1. **Suggested Questions**: All questions in the sidebar expanders are defined as "suggested questions"
2. **Cache Storage**: Answers are stored in `data/answer_cache.json`
3. **Cache Check**: When a suggested question is clicked, the app checks the cache first
4. **Instant Display**: Cached answers are displayed immediately without calling the LLM
5. **Fallback**: Custom user questions still generate fresh answers using the LLM

## Setup Instructions

### Step 1: Index Your Documents

First, make sure your documents are indexed:

```bash
python scripts/process_documents.py
```

### Step 2: Populate the Cache

Run the cache population script to pre-compute all suggested question answers:

```bash
python scripts/populate_cache.py
```

This will:
- Process all 30 suggested questions
- Generate high-quality answers with evidence
- Save them to `data/answer_cache.json`
- Take approximately 5-10 minutes depending on your API speed

### Step 3: Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Now when you click any suggested question, you'll see:
- 📦 "Using cached answer (instant response)" indicator
- Immediate display of the answer
- All evidence and sources intact

## Cache Management

### View Cache Statistics

The sidebar shows:
- Number of cached answers vs total suggested questions
- Example: "Cached Answers: 30/30"

### Refresh the Cache

To update cached answers (e.g., after adding new documents):

```bash
python scripts/populate_cache.py
```

This will regenerate all cached answers with the latest data.

### Clear the Cache

To manually clear the cache:

```bash
rm data/answer_cache.json
```

Or in Python:

```python
from src.llm_orchestration.answer_cache import AnswerCache
from src.config import config

cache = AnswerCache(config.DATA_DIR / 'answer_cache.json')
cache.clear()
```

## Cache File Format

The cache is stored as JSON:

```json
{
  "What video generation models are discussed in the papers?": {
    "question": "What video generation models are discussed in the papers?",
    "answer": "The papers discuss several video generation models...",
    "evidence": [...],
    "sources": [...],
    "has_evidence": true,
    "retrieval_stats": {...},
    "cached_at": "2024-01-15T10:30:00"
  }
}
```

## Benefits

### For Users
- **Instant Responses**: Click and see the answer immediately
- **Reliable Quality**: Pre-generated answers are carefully crafted
- **Explore Efficiently**: Quickly browse through all suggested topics

### For Developers
- **Reduced API Costs**: Suggested questions don't consume API tokens
- **Better UX**: No waiting time for common questions
- **Consistent Results**: Same question always returns same answer

### For Demonstrations
- **Professional Demos**: No delays during presentations
- **Predictable Behavior**: Know exactly what users will see
- **Offline Capability**: Works even if API is slow or rate-limited

## Advanced Usage

### Partial Cache Updates

Update only specific questions:

```python
from src.llm_orchestration.answer_cache import AnswerCache
from src.llm_orchestration.answer_engine import AnswerEngine

cache = AnswerCache(config.DATA_DIR / 'answer_cache.json')
answer_engine = AnswerEngine(retriever)

# Update specific questions
questions = [
    "What video generation models are discussed in the papers?",
    "How does HunyuanVideo achieve state-of-the-art video generation?"
]

for question in questions:
    answer = answer_engine.answer_question(question, top_k=10)
    cache.set(question, answer)
```

### Custom Cache Location

Use a different cache file:

```python
from pathlib import Path

cache = AnswerCache(Path('/custom/path/my_cache.json'))
```

## Troubleshooting

### Cache Not Loading

**Issue**: Cached answers not showing up

**Solutions**:
1. Check if `data/answer_cache.json` exists
2. Run `python scripts/populate_cache.py`
3. Check logs for errors: `tail -f logs/app.log`

### Outdated Answers

**Issue**: Cached answers are stale after adding new papers

**Solution**: Regenerate the cache:
```bash
python scripts/populate_cache.py
```

### Missing Questions

**Issue**: Some suggested questions aren't cached

**Solution**:
1. Verify the question text matches exactly between `app.py` and `populate_cache.py`
2. Regenerate cache with the script

## Best Practices

1. **Regenerate After Updates**: Always refresh cache after:
   - Adding new papers
   - Updating document processing
   - Changing retrieval parameters

2. **Version Control**: Commit `answer_cache.json` to git for consistent team experience

3. **Monitor Cache Size**: Large caches may slow down app startup
   - Current cache: ~30 questions ≈ 500KB
   - Acceptable up to ~100 questions ≈ 2MB

4. **Cache Validation**: Periodically review cached answers for quality

## Performance Metrics

### Without Caching
- Question → LLM API call → Wait 3-8 seconds → Display answer
- Cost: ~$0.01-0.05 per question
- Rate limits: 10-60 requests per minute

### With Caching
- Question → Load from cache → Display instantly (< 100ms)
- Cost: $0
- No rate limits

### One-Time Setup Cost
- 30 questions × 5 seconds average = 2.5 minutes
- 30 questions × $0.02 average = $0.60
- **Saves money after ~30-60 user interactions**

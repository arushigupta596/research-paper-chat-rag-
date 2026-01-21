# Streamlit Cloud Deployment Guide

## ✅ What's Fixed

The deployment errors were caused by:
1. **Detectron2 dependency**: Required `torch` at build time, causing ModuleNotFoundError
2. **Heavy processing libraries**: PaddleOCR, layoutparser, torchvision (not needed for runtime)
3. **Missing pre-processed data**: ChromaDB vector store wasn't in repository

## 🚀 Deployment Steps

### 1. Repository is Ready
✅ Pre-processed vector store (`chroma_db/`) is now tracked in git
✅ Streamlined `requirements.txt` (only runtime dependencies)
✅ System dependencies added in `packages.txt`
✅ Streamlit config added (`.streamlit/config.toml`)
✅ All changes pushed to GitHub

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Repository: `arushigupta596/research-paper-chat-rag-`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Add API Key Secret**
   - While app is deploying, click on "Settings" (gear icon)
   - Go to "Secrets" section
   - Add the following:
   ```toml
   OPENROUTER_API_KEY = "your_actual_openrouter_api_key_here"
   ```
   - Click "Save"

4. **Wait for Deployment**
   - Initial deployment: 3-5 minutes
   - The app will automatically restart after adding secrets

### 3. Verify Deployment

Once deployed, check:
- ✅ App loads without errors
- ✅ Sidebar shows "Total Chunks: 876" and "Indexed Papers: 10"
- ✅ Sample questions are clickable
- ✅ Dark green theme is applied
- ✅ Chat interface is responsive

## 📦 What's Included in Deployment

**Runtime Dependencies (requirements.txt):**
- Streamlit 1.31.0
- LangChain + OpenAI integration
- ChromaDB (vector store)
- Sentence Transformers (embeddings)
- NumPy, Pandas, Pydantic

**Pre-processed Data:**
- `chroma_db/` - Vector store with 876 embedded chunks (~15MB)
- `Data/paper_metadata.json` - Paper titles, topics, keywords

**System Packages (packages.txt):**
- poppler-utils, tesseract-ocr (for potential future use)

## ❌ What's NOT Included

**Document processing dependencies** (not needed for runtime):
- torch, torchvision, detectron2
- paddleocr, paddlepaddle
- layoutparser
- VLM tools (transformers, qwen-vl-utils)

These are only needed for local document processing, not for the deployed chat app.

## 🔧 Troubleshooting

### Deployment Still Failing?

**Check requirements.txt:**
```bash
# Verify detectron2 is commented out or removed
cat requirements.txt | grep detectron2
# Should return nothing or a commented line
```

**Check chroma_db is committed:**
```bash
git ls-files chroma_db/
# Should show multiple .bin and .sqlite3 files
```

**Check secrets are added:**
- Go to app Settings → Secrets
- Verify `OPENROUTER_API_KEY` is present
- Make sure there are no syntax errors in TOML format

### App Loads But Shows "No Papers Indexed"

This means ChromaDB couldn't find the vector store:
- Check that `chroma_db/` directory is in the repository
- Verify `chroma_db/chroma.sqlite3` exists
- Check app logs for ChromaDB errors

### API Key Errors

If you see "Invalid API key" or similar:
- Verify the secret name is exactly `OPENROUTER_API_KEY` (case-sensitive)
- Ensure there are no quotes issues in secrets.toml format
- Test your API key locally first

## 🎯 Expected Behavior

**On successful deployment:**
1. App starts in ~30 seconds
2. Sidebar shows system stats immediately
3. Sample questions are interactive
4. Chat responds with evidence-backed answers
5. Dark green EMB theme is applied

**Performance:**
- First query: ~3-5 seconds (cold start)
- Subsequent queries: ~1-2 seconds
- Retrieval: Instant (local ChromaDB)
- LLM response: Depends on OpenRouter API

## 📊 Resource Usage

**Streamlit Cloud (Free Tier):**
- Memory: ~500MB (well within 1GB limit)
- Storage: ~20MB total
- CPU: Minimal (only LLM API calls)

**API Costs:**
- Qwen-2.5-72B: ~$0.01-0.02 per question
- 100 questions: ~$1-2
- Set spending limits in OpenRouter dashboard

## 🔐 Security Notes

**API Key Protection:**
- ✅ `.env` file excluded from git
- ✅ Secrets stored in Streamlit Cloud dashboard
- ✅ Not exposed in logs or UI

**Public vs Private:**
- Repository is public (code visible)
- Streamlit app URL is public (anyone can access)
- API usage is tied to your OpenRouter account
- Consider enabling authentication if needed

## 📞 Support

If deployment fails after these fixes:
1. Check Streamlit Cloud logs (bottom of deploy page)
2. Verify all files are pushed: `git status`
3. Test locally: `streamlit run app.py`
4. Open GitHub issue with error details

## ✨ Next Steps After Deployment

1. **Test all features**: Try sample questions
2. **Monitor API usage**: Check OpenRouter dashboard
3. **Share with team**: Send Streamlit app URL
4. **Add authentication** (optional): Use Streamlit auth component
5. **Custom domain** (optional): Configure in Streamlit Cloud settings

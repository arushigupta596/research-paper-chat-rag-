#!/bin/bash
# Document Processing Script
# This script processes all PDF documents in the Data/ directory

cd "/Users/arushigupta/Desktop/EMB/Demos/RAG on Research ADI"
source venv/bin/activate
export PATH="/opt/homebrew/Cellar/poppler/25.05.0/bin:$PATH"

echo "Starting document processing..."
echo "Processing 14 PDF files without VLM extraction"
echo "This may take 10-30 minutes depending on system performance"
echo ""

python scripts/process_documents.py --no-vlm

echo ""
echo "Processing complete!"
echo "Check the output above for results"

"""
Script to populate answer cache with all suggested questions.
Run this after indexing documents to pre-compute answers for instant responses.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_retriever import RAGRetriever
from src.llm_orchestration.answer_engine import AnswerEngine
from src.llm_orchestration.answer_cache import AnswerCache
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# All suggested questions from the app
SUGGESTED_QUESTIONS = [
    # Video Generation Models
    "What video generation models are discussed in the papers?",
    "How does HunyuanVideo achieve state-of-the-art video generation?",
    "What are the main components of the video generation pipeline?",
    "What resolutions and durations can the video generation models produce?",
    "How is video super-resolution implemented in these models?",

    # Training & Optimization
    "What training procedures and optimization strategies are used?",
    "How does the Muon optimizer compare to AdamW?",
    "What is the role of Reinforcement Learning in video generation?",
    "How are the models trained for multi-task learning?",
    "What data acquisition and filtering methods are described?",

    # Model Architecture
    "What are the model architectures and parameter counts?",
    "How does the DiT transformer architecture work?",
    "What is the role of the Video Super-Resolution Network?",
    "How are spatial resolution and temporal length scaled?",
    "What pre-training stages are used for the foundation model?",

    # Data & Quality
    "How is training data quality ensured in video generation?",
    "What filtering mechanisms are applied to raw video data?",
    "How are aesthetic scores used to evaluate videos?",
    "What dimensions are used to assess video quality?",
    "How much video data is used for training?",

    # Technical Innovations
    "What novel techniques are introduced for video captioning?",
    "How is the richness-hallucination trade-off addressed?",
    "What reward models are used for reinforcement learning?",
    "How does flow matching-based training work?",
    "What strategies are used for training stability?",

    # Performance & Capabilities
    "What are the main contributions and capabilities of these models?",
    "How do open-source models compare to closed-source alternatives?",
    "What improvements are achieved through supervised fine-tuning?",
    "What tasks can the models perform besides text-to-video?",
    "How is motion quality and temporal consistency improved?"
]


def main():
    """Main script execution."""
    logger.info("Starting cache population...")

    try:
        # Initialize system
        logger.info("Initializing system components...")
        vector_store = VectorStore()
        retriever = RAGRetriever(vector_store)
        answer_engine = AnswerEngine(retriever)

        # Check if documents are indexed
        stats = vector_store.get_stats()
        if stats['total_chunks'] == 0:
            logger.error("No documents indexed. Please run process_documents.py first.")
            sys.exit(1)

        logger.info(f"Found {stats['total_chunks']} indexed chunks from {len(stats['papers'])} papers")

        # Initialize cache
        cache_file = config.DATA_DIR / 'answer_cache.json'
        answer_cache = AnswerCache(cache_file)

        logger.info(f"Current cache has {answer_cache.get_stats()['total_cached']} answers")

        # Populate cache
        logger.info(f"Processing {len(SUGGESTED_QUESTIONS)} suggested questions...")
        answer_cache.update_all(
            questions=SUGGESTED_QUESTIONS,
            answer_engine=answer_engine,
            top_k=10
        )

        # Show final stats
        final_stats = answer_cache.get_stats()
        logger.info(f"✓ Cache population complete!")
        logger.info(f"✓ Total cached answers: {final_stats['total_cached']}")
        logger.info(f"✓ Cache file: {cache_file}")

    except KeyboardInterrupt:
        logger.info("Cache population interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to populate cache: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

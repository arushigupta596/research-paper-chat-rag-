"""
Test RAG system with sample research queries.
"""
import logging
from src.retrieval.vector_store import VectorStore
from src.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_queries():
    """Test the RAG system with sample research queries."""

    # Initialize vector store
    vector_store = VectorStore()

    # Test queries
    queries = [
        "machine learning algorithm performance",
        "neural network architecture design",
        "data preprocessing techniques",
        "optimization methods for deep learning",
        "transformer models attention mechanism"
    ]

    print("\n" + "=" * 80)
    print("Testing RAG System with Research Queries")
    print("=" * 80 + "\n")

    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 80)

        # Search
        results = vector_store.search(query, top_k=5)

        if not results:
            print("❌ No results found")
            continue

        # Display results
        for i, result in enumerate(results, 1):
            relevance = result['score'] * 100
            metadata = result['metadata']

            print(f"\n{i}. 📄 {metadata['paper_name']} (Page {metadata['page_num']})")
            print(f"   Region: {metadata['region_type']} | Relevance: {relevance:.1f}%")

            # Show snippet of text
            text = result['text'][:200].replace('\n', ' ')
            if len(result['text']) > 200:
                text += "..."
            print(f"   Content: {text}")

        print("\n" + "-" * 80)

    # Final stats
    stats = vector_store.get_stats()
    print("\n" + "=" * 80)
    print("ChromaDB Statistics")
    print("=" * 80)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total papers: {len(stats['papers'])}")
    print("\nRegion type distribution:")
    for region_type, count in stats['region_type_counts'].items():
        percentage = (count / stats['total_chunks']) * 100
        print(f"  • {region_type}: {count} ({percentage:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_queries()

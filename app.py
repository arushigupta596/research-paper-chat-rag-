"""
Streamlit application for document understanding and evidence-backed chat.
"""
import streamlit as st
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_retriever import RAGRetriever
from src.llm_orchestration.answer_engine import AnswerEngine
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Research Paper Chat Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - EMB Global Dark-Green Theme
st.markdown("""
<style>
    /* ════════════════════════════════════════════════════════════ */
    /* GLOBAL TYPOGRAPHY SYSTEM */
    /* ════════════════════════════════════════════════════════════ */
    /* Enterprise-grade typography hierarchy:
     *
     * Font Family: Inter, Segoe UI, Roboto, system-ui, sans-serif
     *
     * Size Hierarchy:
     * 1. Page Title: 32px / weight 600 / lh 1.3
     * 2. Section Titles: 22px / weight 500 / lh 1.4
     * 3. Body Text: 16px / weight 400 / lh 1.6-1.65
     * 4. User Questions: 16px / weight 500 / lh 1.5
     * 5. Sample Questions: 15px / weight 500 / lh 1.4
     * 6. Sidebar Headers: 14px / weight 500 / ls 0.04em / lh 1.4
     * 7. Metadata/Labels: 13-14px / weight 400 / lh 1.5
     *
     * Design Principle: Readable for long research answers, calm hierarchy
     */
    * {
        font-family: Inter, 'Segoe UI', Roboto, system-ui, sans-serif;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* GLOBAL BACKGROUND */
    /* ════════════════════════════════════════════════════════════ */
    .stApp {
        background-color: #0c2114;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* SIDEBAR - GRADIENT BACKGROUND */
    /* ════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c2114 0%, #102a1c 60%, #0f2419 100%);
    }

    [data-testid="stSidebar"] .element-container {
        color: #cfe7db;
    }

    /* Sidebar section headers - 14px, weight 500, letter-spacing 0.04em */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #cfe7db !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        line-height: 1.4 !important;
    }

    /* Sidebar body text - lighter than main content */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        font-size: 15px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    /* Sidebar labels and captions */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        font-size: 13px !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* TYPOGRAPHY HIERARCHY */
    /* ════════════════════════════════════════════════════════════ */

    /* 1) Page Title - 32px, weight 600, line-height 1.3 */
    .main-header,
    h1 {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        line-height: 1.3 !important;
        margin-bottom: 1rem;
    }

    /* 2) Section Titles - 22px, weight 500, line-height 1.4 */
    h2 {
        font-size: 22px !important;
        font-weight: 500 !important;
        color: #ffffff !important;
        line-height: 1.4 !important;
    }

    h3 {
        font-size: 22px !important;
        font-weight: 500 !important;
        color: #ffffff !important;
        line-height: 1.4 !important;
    }

    /* Subsection headers */
    h4, h5, h6 {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #cfe7db !important;
        line-height: 1.5 !important;
    }

    /* 3) Body Text - 16px, weight 400, line-height 1.6 */
    p, span, div, li {
        font-size: 16px !important;
        font-weight: 400 !important;
        color: #cfe7db;
        line-height: 1.6 !important;
    }

    /* Answer text - increased line-height for long research answers */
    .main [data-testid="stMarkdownContainer"] > p {
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
    }

    /* 4) User Question Text - 16px, weight 500, line-height 1.5 */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 16px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
    }

    /* Evidence list items */
    ul li, ol li {
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        margin-bottom: 0.5rem;
    }

    /* 7) Metadata / Labels - 14px, weight 400, line-height 1.5 */
    .stCaption,
    [data-testid="stMarkdownContainer"] p em,
    small {
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #8fb3a2 !important;
        line-height: 1.5 !important;
    }

    /* Links */
    a {
        color: #47bf72;
        text-decoration: none;
        font-size: inherit;
    }

    a:hover {
        text-decoration: underline;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* CONTENT SURFACES (CARDS / PANELS) */
    /* ════════════════════════════════════════════════════════════ */
    /* Evidence Box - 16px body text */
    .evidence-box {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
        border-left: 4px solid #47bf72;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        color: #cfe7db;
        font-size: 16px;
        font-weight: 400;
        line-height: 1.65;
    }

    /* Source Citation - 13px metadata */
    .source-citation {
        font-size: 13px;
        font-weight: 400;
        color: #8fb3a2;
        font-style: italic;
        line-height: 1.5;
    }

    /* Stat Box */
    .stat-box {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* CHAT MESSAGES */
    /* ════════════════════════════════════════════════════════════ */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }

    .user-message {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
        border-left: 3px solid #47bf72;
    }

    .assistant-message {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
        border-left: 3px solid #47bf72;
    }

    /* Streamlit native chat messages */
    [data-testid="stChatMessage"] {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* SAMPLE QUESTION BUTTONS - GRADIENT */
    /* ════════════════════════════════════════════════════════════ */
    /* 5) Sample Question Boxes - 15px, weight 500, line-height 1.4 */
    .stButton>button {
        background: linear-gradient(90deg, #0c2114 0%, #346948 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 15px !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #0f2419 0%, #3d7a53 100%);
        transform: translateY(-1px);
    }

    /* ════════════════════════════════════════════════════════════ */
    /* EXPANDERS */
    /* ════════════════════════════════════════════════════════════ */
    /* Expander headers - 15px for sidebar questions */
    .streamlit-expanderHeader {
        background-color: #102a1c;
        border: 1px solid #1f3d2b;
        color: #cfe7db;
        border-radius: 0.5rem;
        font-size: 15px !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: #142f20;
    }

    details[open] > summary {
        border-bottom: 1px solid #1f3d2b;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* METRICS (SIDEBAR STATS) */
    /* ════════════════════════════════════════════════════════════ */
    [data-testid="stMetricValue"] {
        color: #47bf72 !important;
        font-size: 22px !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cfe7db !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* INPUT FIELDS */
    /* ════════════════════════════════════════════════════════════ */
    /* Text Input - 16px body text */
    .stTextInput>div>div>input {
        background-color: #102a1c;
        color: #ffffff;
        border: 1px solid #1f3d2b;
        border-radius: 0.5rem;
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    .stTextInput>div>div>input:focus {
        border-color: #47bf72;
        box-shadow: 0 0 0 1px #47bf72;
    }

    /* Chat Input - 16px body text */
    .stChatInput {
        background-color: #0c2114;
    }

    .stChatInput>div {
        background-color: #0c2114;
    }

    .stChatInput>div>div>input {
        background-color: #0c2114;
        color: #ffffff;
        border: 1px solid #1f3d2b;
        border-radius: 0.5rem;
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    .stChatInput>div>div>input:focus {
        border-color: #47bf72;
        box-shadow: 0 0 0 1px #47bf72;
    }

    /* Chat input container */
    [data-testid="stChatInput"] {
        background-color: #0c2114 !important;
    }

    [data-testid="stBottom"] {
        background-color: #0c2114 !important;
    }

    /* Chat input form wrapper */
    [data-testid="stChatInput"] form {
        background-color: #0c2114 !important;
    }

    /* Main container background */
    .main .block-container {
        background-color: #0c2114;
    }

    /* Bottom area outside chat input */
    section[data-testid="stBottom"]>div {
        background-color: #0c2114 !important;
    }

    /* All chat input parent containers */
    .stChatInputContainer {
        background-color: #0c2114 !important;
    }

    /* Main content area */
    .main {
        background-color: #0c2114 !important;
    }

    /* Content wrapper */
    section.main>div {
        background-color: #0c2114 !important;
    }

    /* Bottom padding area */
    .stBottom {
        background-color: #0c2114 !important;
    }

    /* Chat message container background */
    [data-testid="stVerticalBlock"] {
        background-color: #0c2114;
    }

    /* All block containers */
    .block-container {
        background-color: #0c2114 !important;
        padding-top: 1rem !important;
    }

    /* Remove white background from all containers */
    section[data-testid="stMain"] {
        background-color: #0c2114 !important;
    }

    section[data-testid="stMain"] > div {
        background-color: #0c2114 !important;
    }

    /* Main content wrapper */
    [data-testid="stAppViewContainer"] {
        background-color: #0c2114 !important;
    }

    /* Decoration (top bar) */
    [data-testid="stDecoration"] {
        background-color: #0c2114 !important;
    }

    /* Eliminate all white backgrounds */
    div[class*="element-container"],
    div[class*="stMarkdown"],
    div[class*="stText"] {
        background-color: transparent !important;
    }

    /* Main view container */
    .stMainBlockContainer {
        background-color: #0c2114 !important;
    }

    /* All divs in main content */
    .main > div {
        background-color: #0c2114 !important;
    }

    /* Force all containers to dark green */
    div[data-testid="column"] {
        background-color: #0c2114 !important;
    }

    div[data-testid="stHorizontalBlock"] {
        background-color: #0c2114 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0c2114 !important;
    }

    /* Remove any remaining white/light backgrounds */
    .element-container,
    .stMarkdown,
    .stText {
        background-color: transparent !important;
    }

    /* Main content column */
    [data-testid="stMainBlockContainer"] {
        background-color: #0c2114 !important;
    }

    /* View container */
    .st-emotion-cache-1y4p8pa,
    .st-emotion-cache-z5fcl4 {
        background-color: #0c2114 !important;
    }

    /* All Streamlit containers */
    [class*="st-emotion-cache"] {
        background-color: transparent;
    }

    /* NUCLEAR OPTION - Force ALL backgrounds to match */
    /* Root and app-level containers */
    #root {
        background-color: #0c2114 !important;
    }

    /* All data-testid containers */
    [data-testid*="st"] {
        background-color: transparent !important;
    }

    /* Force main app background */
    [data-testid="stApp"] {
        background-color: #0c2114 !important;
    }

    /* All divs with st- prefix classes */
    [class*="st-"] {
        background-color: transparent !important;
    }

    /* Main content area - force dark green */
    .main, .main > div, .main div {
        background-color: #0c2114 !important;
    }

    /* Streamlit block containers */
    [data-testid="block-container"] {
        background-color: #0c2114 !important;
    }

    /* All divs inside main that might have white background */
    section.main div[class^="st-"],
    section.main div[class*=" st-"] {
        background-color: transparent !important;
    }

    /* Catch-all for any remaining white backgrounds */
    .main [style*="background-color: rgb(255, 255, 255)"],
    .main [style*="background-color: white"],
    .main [style*="background-color:#ffffff"],
    .main [style*="background-color: #ffffff"] {
        background-color: #0c2114 !important;
    }

    /* ULTIMATE FIX - Target the exact visible white area */
    /* This targets the main content area where chat messages appear */
    [data-testid="stVerticalBlock"] > div,
    [data-testid="stVerticalBlock"] > div > div {
        background-color: #0c2114 !important;
    }

    /* All vertical layout blocks */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"] > *,
    div[data-testid="stVerticalBlock"] * {
        background-color: transparent !important;
    }

    /* Ensure stVerticalBlock itself is dark green */
    [data-testid="stVerticalBlock"] {
        background-color: #0c2114 !important;
    }

    /* Main app container and all children */
    .stApp > div,
    .stApp > div > div,
    .stApp > div > div > div {
        background-color: #0c2114 !important;
    }

    /* Section main - the visible content area */
    section.main {
        background-color: #0c2114 !important;
    }

    section.main * {
        background-color: transparent !important;
    }

    /* Override any inline styles */
    div[style*="background"] {
        background-color: #0c2114 !important;
    }

    /* MOST AGGRESSIVE FIX - Force EVERY div to have correct background */
    /* Target the main content column specifically */
    section[data-testid="stMain"] div[data-testid="column"] {
        background-color: #0c2114 !important;
    }

    /* All divs inside main section */
    section[data-testid="stMain"] div {
        background-color: #0c2114 !important;
    }

    /* Override for content that should be transparent */
    section[data-testid="stMain"] [data-testid="stChatMessage"],
    section[data-testid="stMain"] .evidence-box,
    section[data-testid="stMain"] .stat-box {
        background-color: #102a1c !important;
    }

    /* Expander content areas */
    section[data-testid="stMain"] [data-testid="stExpander"] {
        background-color: #0c2114 !important;
    }

    /* Select Boxes - 16px body text */
    .stSelectbox>div>div>div {
        background-color: #102a1c;
        color: #ffffff;
        border: 1px solid #1f3d2b;
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    /* Multiselect - 16px body text */
    .stMultiSelect>div>div>div {
        background-color: #102a1c;
        color: #ffffff;
        border: 1px solid #1f3d2b;
        font-size: 16px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background-color: #47bf72;
        color: #ffffff;
        font-size: 14px !important;
        font-weight: 400 !important;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* SLIDER */
    /* ════════════════════════════════════════════════════════════ */
    .stSlider>div>div>div {
        color: #47bf72;
    }

    .stSlider [role="slider"] {
        background-color: #47bf72;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* DIVIDERS */
    /* ════════════════════════════════════════════════════════════ */
    hr {
        border-color: #1f3d2b;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* SCROLLBAR */
    /* ════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0c2114;
    }

    ::-webkit-scrollbar-thumb {
        background: #47bf72;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3d9e5f;
    }

    /* ════════════════════════════════════════════════════════════ */
    /* HEADER & FOOTER */
    /* ════════════════════════════════════════════════════════════ */
    /* Streamlit header */
    header[data-testid="stHeader"] {
        background-color: #0c2114 !important;
    }

    /* Streamlit footer */
    footer {
        background-color: #0c2114 !important;
    }

    footer[data-testid="stFooter"] {
        background-color: #0c2114 !important;
    }

    /* Hide the "Made with Streamlit" footer if desired */
    footer::after {
        background-color: #0c2114 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the document understanding system."""
    try:
        logger.info("Initializing system...")

        # Initialize vector store
        vector_store = VectorStore()

        # Initialize retriever
        retriever = RAGRetriever(vector_store)

        # Initialize answer engine
        answer_engine = AnswerEngine(retriever)

        logger.info("System initialized successfully")
        return vector_store, retriever, answer_engine

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        st.error(f"System initialization failed: {e}")
        st.stop()


def display_header():
    """Display application header."""
    st.markdown('<div class="main-header">Research Paper Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Chat with 15 research papers using layout-aware document understanding and evidence-backed answers</div>',
        unsafe_allow_html=True
    )


def display_sidebar(vector_store: VectorStore):
    """Display sidebar with system information and filters."""
    with st.sidebar:
        st.header("System Information")

        # Get statistics
        stats = vector_store.get_stats()

        # Display stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Chunks", stats['total_chunks'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Indexed Papers", len(stats['papers']))
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Filters
        st.header("Search Filters")

        # Paper filter
        all_papers = [p['name'] for p in stats['papers']]
        selected_papers = st.multiselect(
            "Filter by Papers",
            options=all_papers,
            default=[],
            help="Leave empty to search all papers"
        )

        # Region type filter
        region_types = list(stats.get('region_type_counts', {}).keys())
        selected_region_types = st.multiselect(
            "Filter by Region Type",
            options=region_types,
            default=[],
            help="Filter by document regions (text, table, figure, etc.)"
        )

        # Number of results
        top_k = st.slider(
            "Number of Evidence Chunks",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of relevant chunks to retrieve"
        )

        st.divider()

        # Sample questions
        st.header("Sample Questions")
        st.caption("Click on any question to ask it")

        # Questions designed to showcase TABLE extractions (VLM-extracted structured data)
        table_questions = [
            "Show me experimental results comparing different models or methods in tables.",
            "What performance metrics and benchmarks are reported in the tables?",
            "What are the quantitative results showing accuracy, precision, or F1 scores?",
            "Compare the numerical results across different datasets or configurations.",
            "What statistical data or measurements are presented in tabular format?"
        ]

        # Questions designed to showcase FIGURE/CHART extractions (VLM-extracted visual data)
        figure_questions = [
            "What trends or patterns are shown in the charts and graphs?",
            "Describe the visualizations and their key insights from the figures.",
            "What does the performance over time or across conditions look like in charts?",
            "What architectural diagrams or model structures are illustrated?",
            "What data distributions or comparative visualizations are presented?"
        ]

        # Questions designed to showcase TEXT content (various text regions)
        methodology_questions = [
            "What are the main methodologies and approaches described?",
            "How were the experiments designed and conducted?",
            "What algorithms or techniques are proposed in the papers?",
            "Explain the training procedures and optimization strategies used.",
            "What preprocessing steps or data preparation methods are mentioned?"
        ]

        # Questions designed to showcase cross-region and multi-document retrieval
        comprehensive_questions = [
            "What are the key findings from both quantitative results and qualitative analysis?",
            "Summarize the main contributions combining text, tables, and figures.",
            "What conclusions can be drawn from the experimental data and visualizations?",
            "Compare the theoretical approach with the empirical results shown.",
            "What evidence supports the main claims across different sections?"
        ]

        # Questions designed to showcase specific technical details from all region types
        technical_details_questions = [
            "What hyperparameters, configurations, or settings are specified?",
            "What evaluation metrics are used and what are their values?",
            "What are the model architectures and their component details?",
            "What datasets are used and what are their characteristics?",
            "What baseline methods are compared and how do they perform?"
        ]

        # Questions designed to showcase LIST and TITLE regions
        structural_questions = [
            "What are the main research contributions or key points listed?",
            "What limitations or future work directions are mentioned?",
            "What related work or background concepts are discussed?",
            "What are the research objectives and hypotheses stated?",
            "What conclusions and recommendations are provided?"
        ]

        with st.expander("Table Data Questions (VLM Extractions)"):
            for i, q in enumerate(table_questions, 1):
                if st.button(f"{i}. {q}", key=f"table_{i}"):
                    st.session_state.selected_question = q

        with st.expander("Figure & Chart Questions (VLM Extractions)"):
            for i, q in enumerate(figure_questions, 1):
                if st.button(f"{i}. {q}", key=f"figure_{i}"):
                    st.session_state.selected_question = q

        with st.expander("Methodology & Text Questions"):
            for i, q in enumerate(methodology_questions, 1):
                if st.button(f"{i}. {q}", key=f"methodology_{i}"):
                    st.session_state.selected_question = q

        with st.expander("Comprehensive Questions (All Region Types)"):
            for i, q in enumerate(comprehensive_questions, 1):
                if st.button(f"{i}. {q}", key=f"comprehensive_{i}"):
                    st.session_state.selected_question = q

        with st.expander("Technical Details (Mixed Regions)"):
            for i, q in enumerate(technical_details_questions, 1):
                if st.button(f"{i}. {q}", key=f"technical_{i}"):
                    st.session_state.selected_question = q

        with st.expander("Structural Content (Lists & Titles)"):
            for i, q in enumerate(structural_questions, 1):
                if st.button(f"{i}. {q}", key=f"structural_{i}"):
                    st.session_state.selected_question = q

        st.divider()

        # System info
        st.header("Indexed Papers")
        with st.expander("View all papers"):
            for paper in stats['papers']:
                st.text(f"• {paper['name']} ({paper['chunks']} chunks)")

        return {
            'selected_papers': selected_papers if selected_papers else None,
            'selected_region_types': selected_region_types if selected_region_types else None,
            'top_k': top_k,
            'diversity_lambda': 0.5,
            'enable_multi_hop': False
        }


def display_answer(answer: Any):
    """
    Display answer with evidence and sources.

    Args:
        answer: Answer object from answer engine
    """
    if not answer.has_evidence:
        st.warning(answer.answer)
        return

    # Display answer
    st.markdown("### Answer")
    st.markdown(answer.answer)

    # Display retrieval stats
    with st.expander("Retrieval Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Evidence Chunks", answer.retrieval_stats['total_chunks'])
        with col2:
            st.metric("Papers Searched", len(answer.retrieval_stats['papers_searched']))
        with col3:
            if 'by_region_type' in answer.retrieval_stats:
                st.metric("Region Types", len(answer.retrieval_stats['by_region_type']))

        # Papers searched
        st.write("**Papers Searched:**")
        for paper in answer.retrieval_stats['papers_searched']:
            chunks = answer.retrieval_stats.get('by_paper', {}).get(paper, 0)
            st.text(f"• {paper} ({chunks} chunks)")

    # Display evidence
    st.markdown("### Key Evidence")
    st.caption("Specific passages and data points that directly support the answer")

    for i, evidence in enumerate(answer.evidence, 1):
        with st.expander(f"Evidence {i}: {evidence['source']['paper']} (Page {evidence['source']['page']})"):
            st.markdown(f"**Region Type:** {evidence['source']['region_type']}")
            st.markdown(f"**Relevance Score:** {evidence['score']:.3f}")
            st.markdown("**Content:**")
            st.markdown(f'<div class="evidence-box">{evidence["text"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="source-citation">Source: {evidence["citation"]}</div>', unsafe_allow_html=True)

    # Display sources
    st.markdown("### Research Paper Sources")
    st.caption("Academic papers from which the evidence was extracted")
    for source in answer.sources:
        st.markdown(f"• {source}")


def main():
    """Main application."""
    # Initialize system
    vector_store, retriever, answer_engine = initialize_system()

    # Display header
    display_header()

    # Display sidebar and get filters
    filters = display_sidebar(vector_store)

    # Check if system has indexed papers
    if vector_store.get_stats()['total_chunks'] == 0:
        st.error(
            "No papers have been indexed yet. "
            "Please run the document processing script first:\n\n"
            "```bash\npython scripts/process_documents.py\n```"
        )
        st.stop()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize selected question
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'])
            else:
                # Display full answer with evidence
                display_answer(message['answer'])

    # Handle selected question from sidebar
    question = None
    if st.session_state.selected_question:
        question = st.session_state.selected_question
        st.session_state.selected_question = None  # Reset after use
    else:
        # Chat input
        question = st.chat_input("Ask a question about the research papers...")

    if question:
        # Add user message to chat
        st.session_state.messages.append({
            'role': 'user',
            'content': question
        })

        # Display user message
        with st.chat_message('user'):
            st.markdown(question)

        # Generate answer
        with st.chat_message('assistant'):
            with st.spinner('Searching papers and generating answer...'):
                try:
                    # Choose reasoning mode
                    if filters['enable_multi_hop']:
                        answer = answer_engine.multi_hop_reasoning(
                            question=question
                        )
                    else:
                        answer = answer_engine.answer_question(
                            question=question,
                            top_k=filters['top_k'],
                            filter_papers=filters['selected_papers'],
                            filter_region_types=filters['selected_region_types']
                        )

                    # Display answer
                    display_answer(answer)

                    # Add to chat history
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'answer': answer
                    })

                except Exception as e:
                    logger.error(f"Error generating answer: {e}")
                    st.error(f"Failed to generate answer: {e}")

    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == '__main__':
    main()

import html
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from retrieval import index_pdf, collection
from generator import generate_answer

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchPilot | AI Research Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
EVALUATION_FILE = Path("evaluation_results.csv")
DATA_DIR.mkdir(exist_ok=True)


def html_block(markup: str) -> str:
    """Collapse HTML onto one line so Markdown never treats indented
    lines as a code block."""
    return "".join(line.strip() for line in markup.strip().splitlines())


# --------------------------------------------------
# DARK THEME STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --rp-bg: #0B0F19;
        --rp-sidebar: #070A12;
        --rp-surface: #121826;
        --rp-surface-2: #182033;
        --rp-border: #243045;
        --rp-ink: #F1F5F9;
        --rp-text: #CBD5E1;
        --rp-muted: #94A3B8;
        --rp-primary: #6366F1;
        --rp-primary-hover: #818CF8;
        --rp-primary-soft: rgba(99, 102, 241, .16);
        --rp-success: #34D399;
        --rp-warning: #FBBF24;
        --rp-radius: 14px;
    }

    html, body, [class*="css"], .stMarkdown, button, input, textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }

    /* ---------- Base ---------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--rp-bg);
        color: var(--rp-text);
    }
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    h1, h2, h3, h4, h5, h6 { color: var(--rp-ink) !important; letter-spacing: -.02em; }
    p, li, label, .stMarkdown { color: var(--rp-text); }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--rp-muted) !important; }
    hr { border-color: var(--rp-border) !important; }
    code { background: var(--rp-surface-2) !important; color: #C7D2FE !important; border-radius: 6px; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: var(--rp-sidebar);
        border-right: 1px solid var(--rp-border);
    }
    [data-testid="stSidebar"] h3 {
        color: var(--rp-muted) !important; font-size: 11.5px; font-weight: 700;
        text-transform: uppercase; letter-spacing: .1em; margin-bottom: 2px;
    }
    [data-testid="stSidebar"] hr { margin: 1.1rem 0; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: var(--rp-surface); border: 1px dashed #334155; border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section * { color: var(--rp-text); }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button {
        background: var(--rp-surface-2); border: 1px solid var(--rp-border); color: var(--rp-ink);
    }

    .rp-brand { display: flex; align-items: center; gap: 12px; padding: 2px 0 4px 0; }
    .rp-brand-icon {
        width: 40px; height: 40px; border-radius: 12px; display: flex;
        align-items: center; justify-content: center; font-size: 20px; color: #fff;
        background: linear-gradient(135deg, #818CF8, #4F46E5);
        box-shadow: 0 6px 18px rgba(99,102,241,.4);
    }
    .rp-brand-name { color: #FFFFFF; font-size: 18px; font-weight: 800; letter-spacing: -.02em; }
    .rp-brand-sub { color: var(--rp-muted); font-size: 12px; }

    .rp-doc-item {
        display: flex; justify-content: space-between; align-items: center; gap: 8px;
        padding: 9px 11px; margin-bottom: 6px; border-radius: 10px;
        background: var(--rp-surface); border: 1px solid var(--rp-border); font-size: 12.5px;
    }
    .rp-doc-name {
        color: var(--rp-ink); overflow: hidden; text-overflow: ellipsis;
        white-space: nowrap; max-width: 150px;
    }
    .rp-doc-count { color: var(--rp-primary-hover); font-weight: 600; font-size: 11.5px; white-space: nowrap; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: var(--rp-surface); color: var(--rp-ink);
        border: 1px solid var(--rp-border); border-radius: 10px;
        font-weight: 600; transition: all .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--rp-primary-hover); color: #FFFFFF; background: var(--rp-surface-2);
    }
    .stButton > button:disabled { opacity: .45; }
    .stButton > button[kind="primary"] {
        background: var(--rp-primary); border-color: var(--rp-primary); color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--rp-primary-hover); border-color: var(--rp-primary-hover);
    }

    /* ---------- Hero ---------- */
    .rp-hero {
        position: relative; overflow: hidden;
        padding: 30px 34px; border-radius: 20px; margin-bottom: 20px;
        background: linear-gradient(125deg, #0F1526 0%, #1A1745 55%, #3730A3 100%);
        border: 1px solid #27305A;
        box-shadow: 0 10px 40px rgba(0,0,0,.35);
    }
    .rp-hero::after {
        content: ""; position: absolute; right: -60px; top: -80px; width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(129,140,248,.35), transparent 65%);
    }
    .rp-eyebrow {
        color: #A5B4FC; font-size: 11px; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: 10px;
    }
    .rp-hero h1 {
        color: #FFFFFF !important; font-size: clamp(26px, 4vw, 36px); font-weight: 800;
        letter-spacing: -.03em; margin: 0 0 10px 0; padding: 0; line-height: 1.15;
    }
    .rp-hero p { color: #C7D2FE; font-size: 15px; line-height: 1.7; max-width: 680px; margin: 0; }
    .rp-chips { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }
    .rp-chip {
        padding: 5px 12px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
        color: #E0E7FF; background: rgba(255,255,255,.08); border: 1px solid rgba(199,210,254,.22);
    }

    /* ---------- Stat cards ---------- */
    .rp-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 22px; }
    .rp-stat {
        background: var(--rp-surface); border: 1px solid var(--rp-border);
        border-radius: var(--rp-radius); padding: 16px 18px;
        display: flex; align-items: center; gap: 14px;
    }
    .rp-stat-icon {
        width: 42px; height: 42px; border-radius: 11px; background: var(--rp-primary-soft);
        display: flex; align-items: center; justify-content: center; font-size: 19px;
    }
    .rp-stat-label { color: var(--rp-muted); font-size: 12px; font-weight: 500; }
    .rp-stat-value { color: var(--rp-ink); font-size: 24px; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--rp-border); background: transparent; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px; font-weight: 600; color: var(--rp-muted);
        border-radius: 8px 8px 0 0; background: transparent;
    }
    .stTabs [aria-selected="true"] { color: var(--rp-primary-hover) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--rp-primary-hover); }
    .stTabs [data-baseweb="tab-border"] { background-color: var(--rp-border); }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: var(--rp-surface); border: 1px solid var(--rp-border);
        border-radius: var(--rp-radius); padding: 16px 18px; margin-bottom: 12px;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--rp-text); }

    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background: var(--rp-bg) !important; }
    [data-testid="stChatInput"] {
        background: var(--rp-surface) !important; border: 1px solid var(--rp-border);
        border-radius: 14px;
    }
    [data-testid="stChatInput"] textarea { background: transparent !important; color: var(--rp-ink) !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--rp-muted); }
    [data-testid="stChatInput"]:focus-within { border-color: var(--rp-primary-hover); }

    /* ---------- Empty state ---------- */
    .rp-empty {
        text-align: center; padding: 34px 20px; border: 1px dashed #334155;
        border-radius: var(--rp-radius); background: var(--rp-surface); margin: 6px 0 18px 0;
    }
    .rp-empty-icon { font-size: 30px; margin-bottom: 6px; }
    .rp-empty-title { color: var(--rp-ink); font-weight: 700; font-size: 16px; margin-bottom: 4px; }
    .rp-empty-text { color: var(--rp-muted); font-size: 13.5px; }

    /* ---------- Source cards ---------- */
    .rp-source {
        border: 1px solid var(--rp-border); border-radius: 12px; padding: 14px 16px;
        margin-bottom: 10px; background: var(--rp-surface-2);
    }
    .rp-source-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 8px; }
    .rp-source-title { color: var(--rp-ink); font-weight: 600; font-size: 13.5px; }
    .rp-badge {
        display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px;
        font-weight: 600; background: var(--rp-primary-soft); color: #A5B4FC; margin-left: 6px;
    }
    .rp-badge-high { background: rgba(52,211,153,.14); color: var(--rp-success); }
    .rp-badge-mid { background: rgba(251,191,36,.14); color: var(--rp-warning); }
    .rp-badge-low { background: rgba(148,163,184,.14); color: var(--rp-muted); }
    .rp-source-text {
        color: var(--rp-text); font-size: 13px; line-height: 1.65;
        border-left: 3px solid #4F46E5; padding-left: 12px; margin-top: 6px;
    }

    /* ---------- Expander, tables, metrics, alerts ---------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--rp-border); border-radius: 12px; background: var(--rp-surface);
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--rp-ink); }
    [data-testid="stDataFrame"] { border: 1px solid var(--rp-border); border-radius: 12px; overflow: hidden; }
    [data-testid="stMetric"] {
        background: var(--rp-surface); border: 1px solid var(--rp-border);
        padding: 16px 18px; border-radius: var(--rp-radius);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--rp-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--rp-ink) !important; font-weight: 800; }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--rp-border); }
    [data-testid="stSpinner"] * { color: var(--rp-text) !important; }

    .rp-footer { text-align: center; padding: 6px 0 14px 0; color: var(--rp-muted); font-size: 12px; }
    .rp-footer strong { color: var(--rp-text); }

    @media (max-width: 768px) {
        .rp-stats { grid-template-columns: 1fr; }
        .rp-hero { padding: 22px 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_indexed_file" not in st.session_state:
    st.session_state.last_indexed_file = None

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

if "confirm_clear_all" not in st.session_state:
    st.session_state.confirm_clear_all = False

# --------------------------------------------------
# HELPERS
# --------------------------------------------------


def get_indexed_documents():
    """Return indexed document names and their chunk counts."""
    try:
        records = collection.get(include=["metadatas"])
        documents = {}
        for metadata in records.get("metadatas") or []:
            if metadata and metadata.get("source"):
                name = metadata["source"]
                documents[name] = documents.get(name, 0) + 1
        return documents
    except Exception:
        return {}



def sha256_bytes(file_bytes: bytes) -> str:
    """Return a stable fingerprint for the exact contents of a PDF."""
    return hashlib.sha256(file_bytes).hexdigest()


def find_duplicate_pdf(file_bytes: bytes):
    """Find identical PDFs among saved files and any indexed hash metadata."""
    uploaded_hash = sha256_bytes(file_bytes)

    # Compare against PDFs already saved locally. This also detects files
    # uploaded under a different filename, as long as the original PDF exists.
    for existing_pdf in DATA_DIR.glob("*.pdf"):
        try:
            if sha256_bytes(existing_pdf.read_bytes()) == uploaded_hash:
                return existing_pdf.name
        except OSError:
            continue

    # Also support hashes already stored in ChromaDB by newer indexers.
    try:
        records = collection.get(include=["metadatas"])
        for metadata in records.get("metadatas") or []:
            if metadata and metadata.get("file_hash") == uploaded_hash:
                return metadata.get("source", "an existing document")
    except Exception:
        pass

    return None


def clear_all_documents():
    """Delete all ChromaDB records and saved PDFs used by this app."""
    records = collection.get()
    ids = records.get("ids") or []
    if ids:
        # Delete in batches to avoid very large delete requests.
        for start in range(0, len(ids), 500):
            collection.delete(ids=ids[start:start + 500])

    for pdf_path in DATA_DIR.glob("*.pdf"):
        pdf_path.unlink(missing_ok=True)


def similarity_badge(similarity):
    """Return an HTML badge describing retrieval similarity."""
    try:
        score = max(0.0, min(1.0, float(similarity)))
    except (ValueError, TypeError):
        return ""
    if score >= 0.65:
        css, label = "rp-badge-high", "Strong match"
    elif score >= 0.45:
        css, label = "rp-badge-mid", "Relevant"
    else:
        css, label = "rp-badge-low", "Weak match"
    return f'<span class="rp-badge {css}">{label} · {score:.2f}</span>'


def render_sources(sources):
    """Display retrieved evidence for an answer."""
    if not sources:
        return

    with st.expander(f"📄 Supporting evidence ({len(sources)} passages)", expanded=False):
        for number, source in enumerate(sources, start=1):
            name = html.escape(str(source.get("source", "Unknown document")))
            page = html.escape(str(source.get("page", "Unknown")))
            text = html.escape(" ".join(str(source.get("text", "No passage text available.")).split()))
            badge = similarity_badge(source.get("similarity"))

            st.markdown(
                html_block(
                    f"""
                    <div class="rp-source">
                        <div class="rp-source-head">
                            <div class="rp-source-title">{number}. {name}
                                <span class="rp-badge">Page {page}</span>
                            </div>
                            <div>{badge}</div>
                        </div>
                        <div class="rp-source-text">{text}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


def stat_card(icon, label, value):
    return html_block(
        f"""
        <div class="rp-stat">
            <div class="rp-stat-icon">{icon}</div>
            <div>
                <div class="rp-stat-label">{label}</div>
                <div class="rp-stat-value">{value}</div>
            </div>
        </div>
        """
    )


def empty_state(icon, title, text):
    st.markdown(
        html_block(
            f"""
            <div class="rp-empty">
                <div class="rp-empty-icon">{icon}</div>
                <div class="rp-empty-title">{title}</div>
                <div class="rp-empty-text">{text}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

indexed_documents = get_indexed_documents()

with st.sidebar:
    st.markdown(
        html_block(
            """
            <div class="rp-brand">
                <div class="rp-brand-icon">✦</div>
                <div>
                    <div class="rp-brand-name">ResearchPilot</div>
                    <div class="rp-brand-sub">AI research workspace</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Add a paper")
    st.caption("Upload a PDF to make it searchable.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_version}",
    )

    if st.button(
        "Upload and index",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        pdf_name = Path(uploaded_file.name).name
        pdf_path = DATA_DIR / pdf_name
        file_bytes = uploaded_file.getvalue()

        duplicate_source = find_duplicate_pdf(file_bytes)

        if duplicate_source:
            st.warning(
                f"This PDF has already been added as **{duplicate_source}**. "
                "Duplicate upload was rejected."
            )
        else:
            try:
                # Do not silently overwrite an indexed file with the same name.
                if pdf_path.exists():
                    st.error(
                        f"A file named **{pdf_name}** already exists. "
                        "Clear the existing library or rename the new PDF."
                    )
                else:
                    pdf_path.write_bytes(file_bytes)

                    with st.spinner("Extracting text, embedding, and indexing..."):
                        result = index_pdf(pdf_path)

                    # A retrieval implementation may return a status. Keep
                    # compatibility with versions where index_pdf returns None.
                    if isinstance(result, dict) and result.get("status") == "duplicate":
                        pdf_path.unlink(missing_ok=True)
                        st.warning(
                            f"This PDF is already indexed as "
                            f"**{result.get('source', 'an existing document')}**."
                        )
                    else:
                        # Persist the fingerprint in ChromaDB metadata so duplicate
                        # detection still works if the local PDF is unavailable later.
                        indexed_records = collection.get(include=["metadatas"])
                        update_ids = []
                        update_metadatas = []

                        for record_id, metadata in zip(
                            indexed_records.get("ids") or [],
                            indexed_records.get("metadatas") or [],
                        ):
                            if metadata and metadata.get("source") == pdf_name:
                                updated_metadata = dict(metadata)
                                updated_metadata["file_hash"] = sha256_bytes(file_bytes)
                                update_ids.append(record_id)
                                update_metadatas.append(updated_metadata)

                        if update_ids:
                            collection.update(
                                ids=update_ids,
                                metadatas=update_metadatas,
                            )

                        st.session_state.last_indexed_file = pdf_name
                        st.success(f"Indexed {pdf_name}")
                        st.rerun()

            except Exception as error:
                # Avoid leaving an unindexed PDF behind after an indexing error.
                try:
                    if pdf_path.exists():
                        pdf_path.unlink()
                except OSError:
                    pass
                st.error(f"Indexing failed: {error}")

    st.divider()
    st.markdown("### Library")

    if indexed_documents:
        for doc_name, chunk_count in sorted(indexed_documents.items()):
            safe_name = html.escape(doc_name)
            st.markdown(
                html_block(
                    f"""
                    <div class="rp-doc-item">
                        <span class="rp-doc-name" title="{safe_name}">📄 {safe_name}</span>
                        <span class="rp-doc-count">{chunk_count} chunks</span>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.markdown("### Session")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Data management")
    st.caption("Remove all indexed documents and saved PDF files.")

    clear_confirmation = st.checkbox(
        "I understand this permanently deletes all documents",
        key=f"confirm_clear_all_{st.session_state.uploader_version}",
    )

    if st.button(
        "🗑️ Clear all data",
        use_container_width=True,
        disabled=not clear_confirmation,
    ):
        try:
            clear_all_documents()
            st.session_state.messages = []
            st.session_state.last_indexed_file = None
            st.session_state.uploader_version += 1
            st.success("All indexed documents and saved PDFs have been deleted.")
            st.rerun()
        except Exception as error:
            st.error(f"Could not clear all data: {error}")

    st.caption("Answers are generated from retrieved document evidence.")

# --------------------------------------------------
# HEADER AND STATS
# --------------------------------------------------

total_documents = len(indexed_documents)
total_chunks = sum(indexed_documents.values())

st.markdown(
    html_block(
        """
        <section class="rp-hero">
            <div class="rp-eyebrow">Your AI research workspace</div>
            <h1>Turn research papers into evidence-backed answers</h1>
            <p>Upload papers, ask focused questions, and inspect the exact passages behind every answer.</p>
            <div class="rp-chips">
                <span class="rp-chip">Retrieval-Augmented Generation</span>
                <span class="rp-chip">Source-aware answers</span>
                <span class="rp-chip">RAGAS evaluated</span>
            </div>
        </section>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="rp-stats">'
    + stat_card("📚", "Indexed documents", total_documents)
    + stat_card("🧩", "Indexed chunks", total_chunks)
    + stat_card("💬", "Chat messages", len(st.session_state.messages))
    + "</div>",
    unsafe_allow_html=True,
)

chat_tab, library_tab, evaluation_tab = st.tabs(
    ["💬  Research Chat", "📁  Document Library", "📊  Evaluation"]
)

# --------------------------------------------------
# TAB 1: RESEARCH CHAT
# --------------------------------------------------

with chat_tab:
    if total_chunks == 0:
        empty_state(
            "📥",
            "Your library is empty",
            "Upload a PDF from the sidebar to start asking questions.",
        )

    if not st.session_state.messages:
        st.markdown("##### What would you like to investigate?")
        st.caption("Pick a starter question or type your own below.")

        examples = [
            "What is the main objective of this study?",
            "What methodology did the researchers use?",
            "What were the key findings?",
            "What limitations did the authors mention?",
        ]

        example_columns = st.columns(2)
        for index, example in enumerate(examples):
            with example_columns[index % 2]:
                if st.button(
                    example,
                    key=f"example_{index}",
                    use_container_width=True,
                    disabled=total_chunks == 0,
                ):
                    st.session_state.pending_question = example
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about your indexed research papers...")

    if question is None:
        question = pending_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching document evidence and generating an answer..."):
                    result = generate_answer(
                        question,
                        top_k=5,
                        return_sources=True,
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as error:
                st.error(f"Unable to generate an answer: {error}")

        st.rerun()

# --------------------------------------------------
# TAB 2: DOCUMENT LIBRARY
# --------------------------------------------------

with library_tab:
    st.markdown("#### Document library")
    st.caption("Documents currently indexed for retrieval.")

    if not indexed_documents:
        empty_state(
            "🗂️",
            "No indexed documents",
            "Upload a PDF from the sidebar to get started.",
        )
    else:
        library_df = pd.DataFrame(
            [
                {
                    "Document": filename,
                    "Indexed chunks": chunks,
                    "File available": "Yes" if (DATA_DIR / filename).is_file() else "No",
                }
                for filename, chunks in sorted(indexed_documents.items())
            ]
        )

        st.dataframe(
            library_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Indexed chunks": st.column_config.NumberColumn(format="%d"),
            },
        )

        st.caption(
            "\"File available\" reflects PDFs present in the current environment. "
            "On hosted deployments, local files may not persist across restarts."
        )

    st.divider()
    st.markdown("#### Files in the data folder")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if pdf_files:
        for pdf in pdf_files:
            size_kb = pdf.stat().st_size / 1024
            st.markdown(f"📄 `{pdf.name}` · {size_kb:,.0f} KB")
    else:
        st.caption("No PDF files currently exist in the data folder.")

# --------------------------------------------------
# TAB 3: EVALUATION
# --------------------------------------------------

with evaluation_tab:
    st.markdown("#### RAG evaluation")
    st.caption("Saved RAGAS metrics and per-question results.")

    if not EVALUATION_FILE.exists():
        st.warning(
            "evaluation_results.csv was not found. Run `python evaluate.py` locally "
            "to create it, then commit the results file only if you intend to publish it."
        )
    else:
        try:
            results_df = pd.read_csv(EVALUATION_FILE)

            if results_df.empty:
                st.warning("The evaluation file contains no results.")
            else:
                score_columns = [
                    c for c in ["faithfulness", "answer_relevancy"] if c in results_df.columns
                ]

                if score_columns:
                    score_layout = st.columns(len(score_columns))
                    for index, column in enumerate(score_columns):
                        scores = pd.to_numeric(results_df[column], errors="coerce").dropna()
                        with score_layout[index]:
                            title = column.replace("_", " ").title()
                            if not scores.empty:
                                st.metric(title, f"{scores.mean():.3f}")
                                st.caption(f"{len(scores)} valid score(s) out of {len(results_df)} rows")
                            else:
                                st.metric(title, "N/A")
                                st.caption("No valid scores available.")

                st.divider()
                st.markdown("##### Per-question results")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download evaluation CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="researchpilot_evaluation_results.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(f"Could not load evaluation results: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.markdown(
    html_block(
        """
        <div class="rp-footer">
            <strong>ResearchPilot</strong> · Python · Streamlit · ChromaDB ·
            Sentence Transformers · Gemini · RAGAS
        </div>
        """
    ),
    unsafe_allow_html=True,
)
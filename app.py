
import streamlit as st
import pandas as pd
from pathlib import Path

from retrieval import index_pdf, collection
from generator import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchPilot | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path("data")
EVALUATION_FILE = Path("evaluation_results.csv")
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_indexed_file" not in st.session_state:
    st.session_state.last_indexed_file = None


# --------------------------------------------------
# HELPER FUNCTIONS
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


def render_sources(sources):
    """Display retrieved evidence for an answer."""
    if not sources:
        st.info("No source passages were returned.")
        return

    with st.expander("📄 View retrieved sources", expanded=False):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**Source {number}: {source.get('source', 'Unknown')}**"
            )
            st.caption(f"Page {source.get('page', 'Unknown')}")

            similarity = source.get("similarity")
            if similarity is not None:
                st.progress(
                    max(0.0, min(1.0, float(similarity))),
                    text=f"Cosine similarity: {similarity:.3f}"
                )

            st.markdown("**Retrieved passage**")
            st.write(source.get("text", "No passage text available."))
            st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("📚 ResearchPilot")
    st.caption("Your AI-powered research workspace")

    st.divider()
    st.subheader("Upload a research paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a research paper to extract and index its text."
    )

    if st.button(
        "⬆️ Upload and Index",
        disabled=uploaded_file is None,
        use_container_width=True
    ):
        pdf_name = Path(uploaded_file.name).name
        pdf_path = DATA_DIR / pdf_name

        try:
            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            with st.spinner(
                "Extracting text, generating embeddings and indexing..."
            ):
                index_pdf(pdf_path)

            st.session_state.last_indexed_file = pdf_name
            st.success(f"Processed {pdf_name}")
            st.rerun()

        except Exception as error:
            st.error(f"Indexing failed: {error}")

    if st.session_state.last_indexed_file:
        st.caption(
            f"Last processed: {st.session_state.last_indexed_file}"
        )

    st.divider()
    st.subheader("Chat controls")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "ResearchPilot uses your indexed documents as evidence "
        "for AI-generated answers."
    )


# --------------------------------------------------
# HEADER AND METRICS
# --------------------------------------------------

st.title("ResearchPilot")
st.markdown(
    "### Research smarter. Find evidence. Get cited answers."
)
st.write(
    "Upload academic papers, ask research questions, inspect "
    "retrieved passages, and evaluate answer quality."
)

indexed_documents = get_indexed_documents()
total_documents = len(indexed_documents)
total_chunks = sum(indexed_documents.values())

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Indexed Documents", total_documents)
metric2.metric("Indexed Chunks", total_chunks)
metric3.metric(
    "Chat Messages",
    len(st.session_state.messages)
)

st.divider()


# --------------------------------------------------
# MAIN WORKSPACE
# --------------------------------------------------

chat_tab, library_tab, evaluation_tab = st.tabs([
    "💬 Research Chat",
    "📁 Document Library",
    "📊 Evaluation Dashboard"
])


# --------------------------------------------------
# TAB 1: RESEARCH CHAT
# --------------------------------------------------

with chat_tab:
    st.subheader("Ask your research papers")

    if total_chunks == 0:
        st.info(
            "Your document library is empty. Upload a PDF using "
            "the sidebar to get started."
        )

    if not st.session_state.messages:
        st.markdown("**Try asking questions like:**")

        examples = [
            "What is the main objective of this study?",
            "What methodology did the researchers use?",
            "What were the key findings?",
            "What limitations did the authors mention?"
        ]

        example_columns = st.columns(2)

        for index, example in enumerate(examples):
            with example_columns[index % 2]:
                if st.button(
                    example,
                    key=f"example_{index}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = example
                    st.rerun()

    # Display conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    pending_question = st.session_state.pop(
        "pending_question", None
    )

    question = st.chat_input(
        "Ask a question about your indexed research papers..."
    )

    if question is None:
        question = pending_question

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching evidence and generating your answer..."
                ):
                    result = generate_answer(
                        question,
                        top_k=5,
                        return_sources=True
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Unable to generate an answer: {error}")

        st.rerun()


# --------------------------------------------------
# TAB 2: DOCUMENT LIBRARY
# --------------------------------------------------

with library_tab:
    st.subheader("Your research library")
    st.write(
        "Review indexed PDFs and the number of chunks stored "
        "in the vector database."
    )

    if not indexed_documents:
        st.info(
            "No indexed documents found. Upload a PDF from the sidebar."
        )
    else:
        library_df = pd.DataFrame([
            {
                "Document": filename,
                "Indexed Chunks": chunks,
                "PDF Available Locally": (
                    DATA_DIR / filename
                ).is_file()
            }
            for filename, chunks in sorted(indexed_documents.items())
        ])

        st.dataframe(
            library_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "PDFs and their embeddings are stored locally in your "
            "'data' and 'chroma_db' folders."
        )

    st.divider()
    st.subheader("Local PDF files")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if pdf_files:
        for pdf in pdf_files:
            st.markdown(f"📄 `{pdf.name}`")
    else:
        st.caption("No PDFs currently exist in the data folder.")


# --------------------------------------------------
# TAB 3: EVALUATION DASHBOARD
# --------------------------------------------------

with evaluation_tab:
    st.subheader("RAG evaluation")
    st.write(
        "Review previously saved evaluation results from your "
        "RAGAS evaluation run."
    )

    if not EVALUATION_FILE.exists():
        st.warning(
            "evaluation_results.csv was not found. Run "
            "'python evaluate.py' in your terminal to create it."
        )

    else:
        try:
            results_df = pd.read_csv(EVALUATION_FILE)

            if results_df.empty:
                st.warning("The evaluation file contains no results.")

            else:
                score_columns = [
                    column for column in [
                        "faithfulness",
                        "answer_relevancy"
                    ]
                    if column in results_df.columns
                ]

                if score_columns:
                    score_columns_layout = st.columns(
                        len(score_columns)
                    )

                    for index, column in enumerate(score_columns):
                        scores = pd.to_numeric(
                            results_df[column],
                            errors="coerce"
                        ).dropna()

                        with score_columns_layout[index]:
                            if not scores.empty:
                                st.metric(
                                    column.replace("_", " ").title(),
                                    f"{scores.mean():.3f}"
                                )
                                st.caption(
                                    f"{len(scores)} valid scores "
                                    f"out of {len(results_df)} questions"
                                )
                            else:
                                st.metric(column.title(), "N/A")
                                st.caption("No valid scores available")

                st.divider()
                st.markdown("**Per-question evaluation results**")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Evaluation CSV",
                    data=csv_data,
                    file_name="researchpilot_evaluation_results.csv",
                    mime="text/csv",
                    use_container_width=False
                )

        except Exception as error:
            st.error(f"Could not load evaluation results: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "ResearchPilot | Python · Streamlit · ChromaDB · "
    "Sentence Transformers · Gemini · RAGAS"
)

# 📚 ResearchPilot — AI-Powered Research Assistant

**ResearchPilot** is an AI-powered research assistant built using Retrieval-Augmented Generation (RAG). It helps users explore research papers by uploading PDF documents, asking questions in natural language, and receiving answers grounded in retrieved document evidence.

🚀 **Live Demo:** [Try ResearchPilot](https://researchpilot-amwnierbx9cw8fjgpwquyy.streamlit.app/)

## ✨ Features

- **PDF Upload & Indexing:** Upload research papers and index their contents for semantic search.
- **Retrieval-Augmented Generation:** Retrieve relevant document passages before generating answers.
- **AI-Powered Question Answering:** Ask questions about research papers using natural language.
- **Source Inspection:** Review retrieved passages and supporting document information.
- **Document Library:** View indexed documents and their associated text chunks.
- **RAG Evaluation Dashboard:** Review saved evaluation results, including faithfulness and answer relevancy.
- **Interactive Web Interface:** Use a Streamlit dashboard to upload documents, chat with research papers, and inspect results.
- **Evaluation Export:** Download evaluation results as a CSV file.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| ChromaDB | Vector storage and retrieval |
| Sentence Transformers | Text embeddings |
| Google Gemini API | AI-powered answer generation |
| Pandas | Evaluation data handling |
| RAGAS | RAG evaluation |

## 🧠 How It Works

ResearchPilot follows a Retrieval-Augmented Generation workflow.

1. **Upload:** The user uploads a research paper in PDF format.
2. **Text Processing:** The application extracts and processes document content.
3. **Embedding & Indexing:** Text segments are converted into embeddings and stored in ChromaDB.
4. **Question Processing:** The user submits a natural-language question.
5. **Semantic Retrieval:** Relevant passages are retrieved from the indexed documents.
6. **Answer Generation:** Gemini generates an answer using the retrieved context.
7. **Evidence Inspection:** The user can inspect the supporting passages returned by retrieval.
8. **Evaluation:** RAGAS evaluation results can be reviewed through the dashboard when available.

## 📂 Project Structure

```text
ResearchPilot/
├── app.py                  # Streamlit application and interface
├── api.py                  # API-related functionality
├── evaluate.py             # RAG evaluation workflow
├── evaluation_dataset.json # Evaluation questions and data
├── generator.py            # Answer generation using Gemini
├── ingest.py               # Document ingestion utilities
├── retrieval.py            # Retrieval and vector database logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Excluded files and directories
├── README.md               # Project documentation
├── data/                   # Uploaded PDF documents (local)
├── chroma_db/              # Local vector database (if configured)
└── evaluation_results.csv  # Saved evaluation results (when generated)
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/CodeWithAshmit/ResearchPilot.git
cd ResearchPilot
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a `.env` file in the project root if your local `generator.py` loads environment variables from it:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Never commit your real API key to GitHub.

For Streamlit Community Cloud, configure the key in **App Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_actual_gemini_api_key"
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

## 📊 RAG Evaluation

ResearchPilot includes an evaluation workflow using RAGAS.

Run the evaluation script from your project directory:

```bash
python evaluate.py
```

If configured successfully, the workflow can produce evaluation results for inspection in the dashboard.

The dashboard supports these metrics when present in the results file:

- **Faithfulness:** Measures how well an answer is supported by its retrieved context.
- **Answer Relevancy:** Measures how relevant the generated answer is to the user's question.

Evaluation results depend on the dataset, configuration, and evaluation run. They should not be interpreted as a guarantee that every answer is correct.

## 🔐 Security & Data

- Keep API keys and credentials private.
- Do not commit `.env` files.
- Exclude private PDFs and generated vector database files from version control unless intentionally sharing them.
- Uploaded files and locally stored indexes may not persist across hosted application restarts.
- Avoid uploading confidential or sensitive research documents unless you understand how your hosting and AI providers process data.

## 🚧 Future Improvements

Potential enhancements include:

- Multi-document comparison and synthesis.
- More detailed citations with page-level references.
- Conversation memory for follow-up questions.
- Streaming AI responses.
- Improved retrieval using reranking and hybrid search.
- Expanded evaluation datasets and automated benchmarking.
- Persistent document storage for hosted deployments.
- Exporting research summaries and reports.

## 👨‍💻 Author

**Ashmit Srivastava**  
B.Tech, Computer Science and Engineering  
Dr. B. R. Ambedkar National Institute of Technology, Jalandhar

- GitHub: [@CodeWithAshmit](https://github.com/CodeWithAshmit)
- Project: [ResearchPilot](https://github.com/CodeWithAshmit/ResearchPilot)
- Live Application: [ResearchPilot Demo](https://researchpilot-amwnierbx9cw8fjgpwquyy.streamlit.app/)

---

*Research faster. Verify evidence. Stay curious.*

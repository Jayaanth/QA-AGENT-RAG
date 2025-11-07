# Legal RAG Agent (Gemini)

This repository contains a small RAG (Retrieval-Augmented Generation) demo that uses Google (Gemini) LLM embeddings and chat together with a FAISS vector store and a LangGraph pipeline to answer legal questions from a set of PDF documents.

This README explains the approach, how to set up and run the project, and where to look in the code (`app.py`).

[Click here to interact with the hosted space on Hugging Face and interact with Gradio UI](https://huggingface.co/spaces/timefullytrue/LegalRagAgent)

## Project contract

- Inputs: a short text query (string) from the user.
- Outputs: a rendered result string containing the planner decision, retrieved context (if any), generated answer, a reference answer produced for evaluation, and automatic metrics (BLEU, ROUGE-L, BERTScore).
- Error modes: missing environment variables cause immediate failure; missing PDFs will be skipped but a missing merged vectorstore will raise a RuntimeError.

## How it works (high level)

1. app.py loads environment variables and dependencies. It requires two API keys:
   - `LANGCHAIN_API_KEY` — used to create the LangSmith client
   - `GEMINI_API_KEY` — used to call Google Generative AI embeddings & chat (named `gem_key` in the code)

2. The code builds a FAISS vector store by reading PDF files (via `PyPDFLoader`) and splitting them into chunks using a `RecursiveCharacterTextSplitter`.
   - Function: `encode_pdf(path, chunk_size=1000, chunk_overlap=200)`
   - Merging: `create_merged_vector_store(paths)` loops the list `pdf_paths` and merges per-file FAISS stores.

3. A LangGraph (`StateGraph`) pipeline is defined with four traceable nodes:
   - `plan_node(state)` — decides whether retrieval is needed (returns `"retrieve"` or `"direct"`). Uses `ChatGoogleGenerativeAI`.
   - `retrieve_node(state)` — when needed, performs `vectorstore.similarity_search(query, k=5)` to fetch relevant chunks and concatenates them into `state['context']`.
   - `answer_node(state)` — generates the final answer using the LLM with the retrieved `context`.
   - `reflect_node(state)` — generates a reference answer and computes automatic evaluation metrics: BLEU, ROUGE-L, and BERTScore. Results are logged to LangSmith via `client.create_run`.

4. `run_query(query: str)` executes the compiled graph and returns a formatted string with the decision, answer, reference, and metrics. `run_query` is the function exposed to the Gradio UI.

5. The Gradio app (`gr.Interface`) launches a simple web UI for entering legal questions and viewing results.

## Important files

- `app.py` — main application and pipeline. Read this file to see exact model names, LLM calls, and where metrics/logging are performed.
- `requirements.txt` — pinned runtime dependencies used by the project.

## Dependencies

Contents of `requirements.txt` (used by this project):

- faiss-cpu
- gradio
- nltk
- rouge-score
- bert-score
- langchain
- langgraph
- langsmith
- langchain-google-genai
- langchain-community
- pypdf

## Setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create a `.env` file in the project root or set environment variables in PowerShell:

- Using `.env` (the code uses `python-dotenv` via `dotenv.load_dotenv()`):

```
LANGCHAIN_API_KEY=sk-...
GEMINI_API_KEY=ya29...  # or the appropriate Google key for Gemini usage
```

- Or set them in PowerShell before running:

```powershell
$env:LANGCHAIN_API_KEY = "<your_langchain_api_key>"
$env:GEMINI_API_KEY = "<your_gemini_api_key>"
```

4. Add your PDFs to the project root. By default `app.py` looks for:

```
BNSS_2023.pdf
BNS_2023.pdf
BSA.pdf
COI.pdf
```

Either add files with those exact names or edit the `pdf_paths` list in `app.py` to match your filenames.

5. Run the app:

```powershell
python app.py
```

The code will:
- download NLTK `punkt` tokenizer if missing,
- build a FAISS vector store (this may take time depending on PDF sizes),
- launch a Gradio UI and open a local web page.

## Configuration notes & quick tips

- If `GEMINI_API_KEY` is not present, `app.py` raises `ValueError("❌ Missing gem_key environment variable.")` as an early guard.
- If none of the PDFs are found or the merged vector store is empty, `app.py` raises `RuntimeError("❌ Failed to build vector store. Check PDF paths.")`.
- Chunk size and overlap can be tuned in `encode_pdf` (defaults: 1000 tokens, 200 overlap). Reduce chunk_size to create smaller context windows.
- The LLM model names used are `gemini-2.5-flash-lite` (for planning and answering) and `gemini-2.5-flash` (for reference generation in reflect). Edit these model strings in `app.py` if you want to change models.
- The reflect/evaluation step generates and logs BLEU/ROUGE/BERTScore to LangSmith; this requires a valid `LANGCHAIN_API_KEY` for the `Client`.

## Data flow & state shape

The pipeline uses a `QAState` TypedDict inside `app.py`. Key fields:

- `query` (str)
- `decision` (str) — `"retrieve"` or `"direct"`
- `context` (str) — concatenated retrieved chunks
- `answer` (str) — LLM-generated answer
- `reference` (str) — LLM-generated reference used for evaluation
- `bleu`, `rougeL`, `bert_score` (float) — evaluation metrics

Each LangGraph node expects and returns this `state` dict.

## Troubleshooting

- Missing API keys: verify `.env` or environment variables. `GEMINI_API_KEY` is mandatory.
- Slow or failing FAISS build: ensure `pypdf` can read your PDFs, sufficient memory is available, and `faiss-cpu` installed correctly.
- Tokenizer download: the first run downloads NLTK `punkt`. If blocked, run `python -c "import nltk; nltk.download('punkt')"` separately.
- If Gradio does not open the browser, look at the console output for the local URL and open it manually.

## Extending or customizing

- Add more PDF sources: modify `pdf_paths` in `app.py` or add a file-picker UI in the Gradio app.
- Change retrieval size `k` in `retrieve_node` to fetch more or fewer chunks.
- Replace FAISS with another vector store supported by LangChain if you need remote/managed vector stores.
- Add caching: once the FAISS store is created it could be saved/loaded to disk to avoid rebuilding on every run.

## Where to look in code

- `encode_pdf` — PDF loading, splitting, and embeddings
- `create_merged_vector_store` — merges per-file FAISS stores
- LangGraph node functions: `plan_node`, `retrieve_node`, `answer_node`, `reflect_node`
- `run_query` — orchestrates the graph and is the Gradio-exposed function


---

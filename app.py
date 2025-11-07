import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict

import gradio as gr
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langgraph.graph import StateGraph, END
from langsmith import Client, traceable

# =========================================================
# 🏁 Initialization
# =========================================================
print("🔧 Initializing environment...")

nltk.download("punkt", quiet=True)

# =========================================================
# 🔧 Environment setup
# =========================================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Legal RAG Agent (Gemini)"

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
gem_key = os.getenv("GEMINI_API_KEY")
if not gem_key:
    raise ValueError("❌ Missing gem_key environment variable.")

print("✅ Environment variables loaded successfully.")

client = Client(api_key=LANGCHAIN_API_KEY)

# =========================================================
# 🧠 Build Vector Store
# =========================================================
def replace_t_with_space(texts):
    for text in texts:
        if hasattr(text, "page_content"):
            text.page_content = text.page_content.replace("\t", " ")
    return texts

@traceable(run_type="embedding")
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    print(f"📘 Encoding PDF: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    cleaned = replace_t_with_space(chunks)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=gem_key
    )
    print(f"✅ PDF encoded successfully: {path}")
    return FAISS.from_documents(cleaned, embeddings)

pdf_paths = ["BNSS_2023.pdf", "BNS_2023.pdf", "BSA.pdf", "COI.pdf"]

def create_merged_vector_store(paths):
    print("🧩 Building merged vector store...")
    merged = None
    for p in paths:
        if os.path.exists(p):
            print(f"🔹 Processing file: {p}")
            store = encode_pdf(p)
            if merged is None:
                merged = store
            else:
                merged.merge_from(store)
        else:
            print(f"⚠️ Warning: file not found - {p}")
    print("✅ Vector store created successfully.")
    return merged

vectorstore = create_merged_vector_store(pdf_paths)
if not vectorstore:
    raise RuntimeError("❌ Failed to build vector store. Check PDF paths.")

# =========================================================
# 🧩 LangGraph Nodes
# =========================================================

@traceable(run_type="prompt")
def plan_node(state):
    print("🧭 [PLAN NODE] Running planning stage...")
    query = state["query"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=gem_key)
    plan_prompt = f"""
    You are a planner. Given this query:
    {query}
    Decide if retrieval from law PDFs is needed.
    Reply only with 'retrieve' or 'direct'.
    """
    decision = llm.invoke(plan_prompt).content.strip().lower()
    print(f"🧩 Plan decision: {decision}")
    state["decision"] = decision
    return state


@traceable(run_type="retriever")
def retrieve_node(state):
    print("📖 [RETRIEVE NODE] Running retrieval stage...")
    if state["decision"] == "direct":
        print("ℹ️ Retrieval skipped (direct answer chosen).")
        state["context"] = "No retrieval needed."
        return state
    query = state["query"]
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    print(f"✅ Retrieved {len(docs)} relevant document chunks.")
    state["context"] = context
    return state


@traceable(run_type="llm")
def answer_node(state):
    print("💬 [ANSWER NODE] Generating answer...")
    query = state["query"]
    context = state.get("context", "")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=gem_key)
    answer_prompt = f"""
    You are a legal assistant.
    Context:
    {context}

    Question:
    {query}

    Provide a precise, legally accurate answer.
    """
    response = llm.invoke(answer_prompt).content
    print("✅ Answer generated successfully.")
    state["answer"] = response
    return state


@traceable(run_type="tool")
def reflect_node(state):
    print("🔍 [REFLECT NODE] Evaluating answer quality (no LLM)...")

    query = state["query"]
    answer = state["answer"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gem_key) 
    reference_prompt = f"Provide an ideal concise answer for evaluation.\nQuestion: {query}" 
    reference_answer = llm.invoke(reference_prompt).content.strip() 
    print("📏 Reference answer generated for comparison.")
    state["reference"] = reference_answer

    # --- BLEU ---
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([reference_answer.split()], answer.split(), smoothing_function=smoothing)

    # --- ROUGE-L ---
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_val = rouge.score(reference_answer, answer)["rougeL"].fmeasure

    # --- BERTScore ---
    P, R, F1 = bert_score([answer], [reference_answer], lang="en", verbose=False)
    bert_f1 = float(F1[0])

    state.update({
        "bleu": round(bleu, 4),
        "rougeL": round(rouge_val, 4),
        "bert_score": round(bert_f1, 4),
    })

    print(f"📊 Metrics → BLEU: {state['bleu']}, ROUGE-L: {state['rougeL']}, BERTScore: {state['bert_score']}")

    client.create_run(
        name="Reflect-Metrics",
        run_type="tool",
        inputs={"query": query, "answer": answer, "reference": reference_answer},
        outputs={
            "BLEU": state["bleu"],
            "ROUGE-L": state["rougeL"],
            "BERTScore": state["bert_score"]
        },
        project_name="Legal RAG Agent (Gemini)"
    )
    print("✅ Evaluation logged.")
    return state


# =========================================================
# 🔗 LangGraph Flow
# =========================================================
print("⚙️ Building LangGraph pipeline...")

class QAState(TypedDict):
    query: str
    decision: str
    context: str
    answer: str
    reference: str
    bleu: float
    rougeL: float
    bert_score: float

graph = StateGraph(QAState)

graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("reflect", reflect_node)
graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")
graph.add_edge("reflect", END)
compiled_graph = graph.compile()

print("✅ LangGraph pipeline ready.")

# =========================================================
# ⚙️ Execution Function
# =========================================================

@traceable(run_type="chain", name="Legal-RAG-Query")
def run_query(query: str):
    print(f"\n🚀 Running query: {query}")
    initial_state = {"query": query}

    # Execute your LangGraph flow
    final = compiled_graph.invoke(initial_state)

    # Log results to LangSmith manually
    client.create_run(
        name="Legal-RAG-Query",
        run_type="chain",
        inputs={"query": query},
        outputs={
            #"decision": final.get("decision"),
            "answer": final.get("answer")
            #"reference": final.get("reference"),
            #"BLEU": final.get("bleu"),
            #"ROUGE-L": final.get("rougeL"),
            #"BERTScore": final.get("bert_score"),
        },
        project_name="Legal RAG Agent (Gemini)"
    )

    print("✅ Query execution complete.\n")

    return (
        f"**Query:** {query}\n\n"
        f"**Decision:** {final['decision']}\n\n"
        f"**Answer:** {final['answer']}\n\n"
        f"**Reference:** {final['reference']}\n\n"
        f"**Metrics:**\n"
        f"• BLEU: {final['bleu']}\n"
        f"• ROUGE-L: {final['rougeL']}\n"
        f"• BERTScore: {final['bert_score']}"
    )

# =========================================================
# 🧭 Gradio UI
# =========================================================
print("🎨 Launching Gradio interface...")

demo = gr.Interface(
    fn=run_query,
    inputs=[gr.Textbox(label="Enter your legal question", lines=2)],
    outputs=[gr.Textbox(label="Result", lines=16)],
    title="⚖️ Legal RAG Agent",
    description=())

if __name__ == "__main__":
    print("✅ Ready! Opening app...")
    demo.queue().launch(share=True, show_error=True)

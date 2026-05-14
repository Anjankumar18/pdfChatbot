# main.py — terminal chatbot
from pathlib import Path
import sys
import json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chatbot_core import build_chat_graph, ask, load_or_create_vectorstore

# ── Load Documents ────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = BASE_DIR / "pdfs"

def load_documents(path):
    docs = []
    docs.extend(DirectoryLoader(str(path), glob="**/*.pdf",
                loader_cls=PyPDFLoader).load())
    docs.extend(DirectoryLoader(str(path), glob="**/*.docx",
                loader_cls=Docx2txtLoader).load())
    return docs

input_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None

if input_path is not None:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pages          = PyPDFLoader(str(input_path)).load()
        source_display = str(input_path)
    elif input_path.is_file() and input_path.suffix.lower() == ".docx":
        pages          = Docx2txtLoader(str(input_path)).load()
        source_display = str(input_path)
    elif input_path.is_dir():
        pages          = load_documents(input_path)
        source_display = str(input_path)
    else:
        raise FileNotFoundError(f"Not found: {input_path}")
else:
    if DEFAULT_PDF_DIR.exists():
        pages          = load_documents(DEFAULT_PDF_DIR)
        source_display = str(DEFAULT_PDF_DIR)
    else:
        root_files = list(BASE_DIR.glob("*.pdf")) + list(BASE_DIR.glob("*.docx"))
        if not root_files:
            raise FileNotFoundError(
                "No files found. Create a 'pdfs' folder or pass a path:\n"
                "python main.py \"C:/path/to/folder\""
            )
        pages = []
        for pdf in BASE_DIR.glob("*.pdf"):
            pages.extend(PyPDFLoader(str(pdf)).load())
        for docx in BASE_DIR.glob("*.docx"):
            pages.extend(Docx2txtLoader(str(docx)).load())
        source_display = str(BASE_DIR)

if not pages:
    raise ValueError("No files found.")

# ── Build ─────────────────────────────────────────────────────
splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks     = splitter.split_documents(pages)
chat_graph = build_chat_graph(pages, chunks)
history    = ChatMessageHistory()

# ── Evaluation ────────────────────────────────────────────────
qa_log = []

def run_evaluation():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_recall, context_precision
        )
    except ImportError:
        print("Run: pip install ragas datasets")
        return

    print("\n" + "="*50)
    print("RAG EVALUATION")
    print("="*50)

    # Evaluate logged Q&A
    if qa_log:
        valid = [qa for qa in qa_log if qa["contexts"]]
        if valid:
            print(f"Evaluating {len(valid)} logged Q&A pairs...")
            try:
                ds      = Dataset.from_dict({
                    "question": [q["question"] for q in valid],
                    "answer":   [q["answer"]   for q in valid],
                    "contexts": [q["contexts"] for q in valid],
                })
                results = evaluate(
                    dataset=ds,
                    metrics=[faithfulness, answer_relevancy]
                )
                print(f"Faithfulness:     {results['faithfulness']:.3f}")
                print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
            except Exception as e:
                print(f"Evaluation error: {e}")

    # Predefined test cases
    print("\n--- Running predefined test questions ---")

    # !! Update these with your actual document questions !!
    test_cases = [
        {
            "question":     "What is the main topic of the document?",
            "ground_truth": "Update this with your expected answer"
        },
        {
            "question":     "Summarize the key points",
            "ground_truth": "Update this with your expected answer"
        },
    ]

    vectorstore = load_or_create_vectorstore(chunks, pages)
    retriever   = vectorstore.as_retriever()
    eval_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_prompt  = ChatPromptTemplate.from_template("""
    Answer based only on context below.
    If not found say "Not found in documents."
    Context: {context}
    Question: {input}
    """)

    eval_chain  = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(eval_llm, rag_prompt)
    )

    questions, answers, contexts, ground_truths = [], [], [], []

    for case in test_cases:
        print(f"  Testing: {case['question'][:50]}...")
        result = eval_chain.invoke({"input": case["question"]})
        questions.append(case["question"])
        answers.append(result["answer"])
        ground_truths.append(case["ground_truth"])
        contexts.append([d.page_content for d in result["context"]])

    try:
        ds      = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })
        results = evaluate(
            dataset=ds,
            metrics=[
                faithfulness, answer_relevancy,
                context_recall, context_precision
            ]
        )
        overall = sum([
            results['faithfulness'],
            results['answer_relevancy'],
            results['context_recall'],
            results['context_precision']
        ]) / 4

        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Faithfulness:      {results['faithfulness']:.3f}")
        print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
        print(f"Context Recall:    {results['context_recall']:.3f}")
        print(f"Context Precision: {results['context_precision']:.3f}")
        print(f"Overall Score:     {overall:.3f}")
        print("="*50)

        with open("eval_results.json", "w") as f:
            json.dump({
                "timestamp":         datetime.now().isoformat(),
                "faithfulness":      results['faithfulness'],
                "answer_relevancy":  results['answer_relevancy'],
                "context_recall":    results['context_recall'],
                "context_precision": results['context_precision'],
                "overall":           overall
            }, f, indent=2)
        print("Saved to eval_results.json ✅")

    except Exception as e:
        print(f"Evaluation error: {e}")

# ── Chat Loop ─────────────────────────────────────────────────
print(f"\nLoaded {len(pages)} pages from: {source_display}")
print("Agent + LangGraph: ON")
print("Commands: 'exit' | 'clear' | 'eval'\n")

while True:
    question = input("\n> ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    if question.lower() == "clear":
        history.clear()
        print("Memory cleared!")
        continue

    if question.lower() == "eval":
        run_evaluation()
        continue

    try:
        answer = ask(chat_graph, question, history)

        # Log for evaluation
        qa_log.append({
            "timestamp": datetime.now().isoformat(),
            "question":  question,
            "answer":    answer,
            "contexts":  []
        })

        print(f"\n{answer}")

    except Exception as e:
        print(f"\nSomething went wrong: {e}")
        print("Please try again.")
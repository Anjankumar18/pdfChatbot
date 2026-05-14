# chatbot_core.py
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")

# ── Vector Store ──────────────────────────────────────────────
def load_or_create_vectorstore(chunks, pages):
    hash_file    = Path(CHROMA_DIR) / "doc_hash.json"
    current_hash = hashlib.md5(
        "".join([p.page_content for p in pages]).encode()
    ).hexdigest()

    if Path(CHROMA_DIR).exists() and hash_file.exists():
        saved = json.loads(hash_file.read_text())
        if saved["hash"] == current_hash:
            print("✓ Loading existing embeddings...")
            return Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=OpenAIEmbeddings()
            )

    print("◌ Embedding documents...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DIR
    )
    Path(CHROMA_DIR).mkdir(exist_ok=True)
    hash_file.write_text(json.dumps({"hash": current_hash}))
    return vs

# ── LangGraph State ───────────────────────────────────────────
class GraphState(TypedDict):
    input:        str
    chat_history: List[BaseMessage]
    output:       str
    tools_used:   List[str]
    error:        str

# ── Build Chat Graph ──────────────────────────────────────────
def build_chat_graph(pages, chunks):
    vectorstore = load_or_create_vectorstore(chunks, pages)
    retriever   = vectorstore.as_retriever()
    llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── RAG Chain ─────────────────────────────────────────────
    rag_prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    If the answer is not in the context, say "NOT_FOUND_IN_DOCUMENTS".

    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain      = create_retrieval_chain(retriever, document_chain)

    # ── Tools ─────────────────────────────────────────────────
    @tool
    def rag_search(query: str) -> str:
        """
        ALWAYS USE THIS TOOL FIRST.
        Searches loaded PDF and Word documents using RAG.
        Use this for ANY question — document related or not.
        Only if this returns NOT_FOUND_IN_DOCUMENTS should
        you consider using web search.
        Input: a clear question or search query.
        """
        result = rag_chain.invoke({"input": query})
        return result["answer"]

    @tool
    def web_search(query: str) -> str:
        """
        Search the internet.
        ONLY use this if rag_search returned NOT_FOUND_IN_DOCUMENTS.
        Use for current events or info not in the documents.
        Input: a clear search query.
        """
        return DuckDuckGoSearchRun().run(query)

    tools = [rag_search, web_search]

    # ── Agent Prompt — strict ordering ────────────────────────
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful document assistant.

STRICT TOOL ORDER — follow this every single time:

STEP 1: ALWAYS call rag_search first, no exceptions.
STEP 2: Read the result carefully.
        - If result contains useful information → use it to answer. STOP.
        - If result says "NOT_FOUND_IN_DOCUMENTS" → proceed to Step 3.
STEP 3: Only if Step 2 returned NOT_FOUND_IN_DOCUMENTS,
        call web_search for current or external information.
STEP 4: Answer using whatever information you found.

NEVER skip rag_search.
NEVER call web_search before rag_search.
NEVER make up information not returned by tools."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # ── Agent ─────────────────────────────────────────────────
    agent          = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # ── LangGraph Nodes ───────────────────────────────────────
    def run_agent(state: GraphState) -> dict:
        try:
            result     = agent_executor.invoke({
                "input":        state["input"],
                "chat_history": state["chat_history"]
            })
            tools_used = []
            for step in result.get("intermediate_steps", []):
                tool_name = step[0].tool
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
            return {
                "output":     result["output"],
                "tools_used": tools_used,
                "error":      ""
            }
        except Exception as e:
            return {
                "output":     "",
                "tools_used": [],
                "error":      str(e)
            }

    def handle_error(state: GraphState) -> dict:
        print(f"--- Error: {state['error']} ---")
        return {
            "output": "Sorry, something went wrong. Please try again.",
            "error":  ""
        }

    def format_output(state: GraphState) -> dict:
        tools = state["tools_used"]
        if "rag_search" in tools and "web_search" in tools:
            source_tag = "[Source: Documents + Web]"
        elif "rag_search" in tools:
            source_tag = "[Source: Documents]"
        elif "web_search" in tools:
            source_tag = "[Source: Web]"
        else:
            source_tag = "[Source: Agent]"
        return {"output": f"{source_tag}\n{state['output']}"}

    def route_after_agent(state: GraphState) -> str:
        return "handle_error" if state["error"] else "format_output"

    # ── Build Graph ───────────────────────────────────────────
    graph = StateGraph(GraphState)
    graph.add_node("run_agent",     run_agent)
    graph.add_node("handle_error",  handle_error)
    graph.add_node("format_output", format_output)
    graph.set_entry_point("run_agent")
    graph.add_conditional_edges(
        "run_agent",
        route_after_agent,
        {
            "handle_error":  "handle_error",
            "format_output": "format_output"
        }
    )
    graph.add_edge("handle_error",  END)
    graph.add_edge("format_output", END)

    return graph.compile()


# ── Ask Function ──────────────────────────────────────────────
def ask(chat_graph, question: str, history: ChatMessageHistory) -> str:
    result = chat_graph.invoke({
        "input":        question,
        "chat_history": history.messages,
        "output":       "",
        "tools_used":   [],
        "error":        ""
    })
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=result["output"]))
    return result["output"]
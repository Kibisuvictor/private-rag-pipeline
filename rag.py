# rag.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(model_name: str = "llama3"):
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Vector DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # LLM (local via Ollama)
    llm = ChatOllama(model=model_name, temperature=0.1)

    # Prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Step 1: retrieve docs + keep question
    retrieval = RunnableParallel({
        "docs": retriever,
        "question": RunnablePassthrough()
    })

    # Step 2: RAG pipeline (answer + sources)
    rag_chain = (
        retrieval
        | RunnableParallel({
            "answer": (
                {
                    "context": lambda x: format_docs(x["docs"]),
                    "question": lambda x: x["question"],
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "sources": lambda x: x["docs"]
        })
    )

    return rag_chain


def query(chain, question: str):
    response = chain.invoke(question)
    return response["answer"], response["sources"]
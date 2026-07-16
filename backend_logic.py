import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Fixed, persistent paths -> no random UUID folders, no re-indexing every click
DATA_DIR = "./data"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

# Groq model to use. Options: "qwen/qwen3-32b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def init_settings():
    """Configure the global LLM + embedding model once (Groq = fast cloud inference)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to your .env file or hosting platform secrets."
        )

    Settings.llm = Groq(
        model=GROQ_MODEL,
        api_key=api_key,
        request_timeout=60.0,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _get_chroma_collection():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return db.get_or_create_collection(COLLECTION_NAME)


def has_existing_index() -> bool:
    """Check if a persisted index already exists on disk."""
    if not os.path.exists(CHROMA_DB_PATH):
        return False
    collection = _get_chroma_collection()
    return collection.count() > 0


def clear_index():
    """Wipe out any existing embeddings before a new document comes in."""
    collection = _get_chroma_collection()
    if collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)


def load_existing_index():
    """Load the index from the existing persistent Chroma collection (fast, no rebuild)."""
    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def process_documents(data_path: str = DATA_DIR, force_rebuild: bool = True):
    """
    Build (or reuse) the vector index.

    force_rebuild=True (default): always re-embed the current contents of the
    data folder. Since /upload clears old data + old embeddings before saving
    the new file, this keeps only ONE document's data active at a time.
    """
    init_settings()

    if not force_rebuild and has_existing_index():
        return load_existing_index()

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    documents = SimpleDirectoryReader(data_path).load_data()
    if not documents:
        raise ValueError("No documents found in the data folder!")

    collection = _get_chroma_collection()
    if force_rebuild and collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def get_query_engine(index, top_k: int = 5):
    """similarity_top_k=5 gives enough context to retrieve full code blocks, not just fragments."""
    from llama_index.core import PromptTemplate

    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Answer ONLY using the provided context. If the answer is NOT in the context, "
        "strictly say: 'I am sorry, but this information is not in the uploaded document.'\n"
        "Choose the clearest format for the content automatically, without being asked:\n"
        "- Comparisons (X vs Y) -> a Markdown table\n"
        "- Steps or multiple items -> bullet or numbered lists\n"
        "- Code (e.g. Python) -> a fenced code block with the language tag, e.g. ```python\n"
        "- Key terms -> **bold**\n"
        "- A single simple fact -> plain sentence, no need to force structure\n"
        "Preserve code from the context exactly as written, inside a fenced code block.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_template = PromptTemplate(qa_prompt_str)

    return index.as_query_engine(
        streaming=True,
        similarity_top_k=top_k,
        text_qa_template=qa_template,
    )
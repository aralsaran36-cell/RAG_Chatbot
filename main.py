import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend_logic import process_documents, has_existing_index, clear_index, DATA_DIR

load_dotenv()

app = FastAPI(title="Qwen RAG API")

# In-memory holder for the loaded index (kept alive across requests in this process)
STATE = {"index": None}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
    else:
        os.makedirs(DATA_DIR)

    clear_index()
    STATE["index"] = None

    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "ok", "filename": file.filename}


@app.post("/build-index")
def build_index():
    try:
        index = process_documents(force_rebuild=True)
        STATE["index"] = index
        return {"status": "ready"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Build failed: {str(e)}")


@app.get("/chat")
def chat(query: str):
    from llama_index.core import PromptTemplate, Settings

    index = STATE.get("index")
    if index is None:
        raise HTTPException(status_code=400, detail="Knowledge base not built yet.")

    try:
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

        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=5,
            text_qa_template=qa_template,
            llm=Settings.llm,
        )

        if query_engine is None:
            raise HTTPException(
                status_code=500,
                detail="index.as_query_engine() itself returned None — this should never happen.",
            )

        response = query_engine.query(query)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    def stream():
        for chunk in response.response_gen:
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")


# Serve the simple chat frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index_page():
    return FileResponse("static/index.html")

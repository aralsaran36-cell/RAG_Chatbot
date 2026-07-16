import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend_logic import process_documents, get_query_engine, has_existing_index, clear_index, DATA_DIR

load_dotenv()

app = FastAPI(title="Qwen RAG API")

# In-memory holder for the loaded index (kept alive across requests in this process)
STATE = {"index": None}


@app.on_event("startup")
def startup():
    # If an index was already built in a previous run, load it immediately so
    # the app is ready to answer questions without waiting for a manual rebuild.
    if has_existing_index():
        STATE["index"] = process_documents()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
    else:
        os.makedirs(DATA_DIR)

    # Wipe old embeddings — new document replaces the old one entirely
    clear_index()
    STATE["index"] = None

    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "ok", "filename": file.filename}


@app.post("/build-index")
def build_index(force_rebuild: bool = True):
    try:
        STATE["index"] = process_documents(force_rebuild=force_rebuild)
        return {"status": "ready"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chat")
def chat(query: str):
    if STATE["index"] is None:
        raise HTTPException(status_code=400, detail="Knowledge base not built yet.")

    query_engine = get_query_engine(STATE["index"])
    response = query_engine.query(query)

    def stream():
        for chunk in response.response_gen:
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")


# Serve the simple chat frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index_page():
    return FileResponse("static/index.html")
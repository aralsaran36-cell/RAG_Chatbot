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

STATE = {"index": None}


@app.on_event("startup")
def startup():
    if has_existing_index():
        try:
            STATE["index"] = process_documents(force_rebuild=False)
        except Exception as e:
            print(f"STARTUP LOAD FAILED: {e}")
            STATE["index"] = None


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
def build_index(force_rebuild: bool = True):
    try:
        index = process_documents(force_rebuild=force_rebuild)
        print(f"BUILD RESULT: index type = {type(index)}, is None = {index is None}")
        if index is None:
            raise HTTPException(status_code=500, detail="Index build returned nothing.")
        STATE["index"] = index
        return {"status": "ready"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print("BUILD EXCEPTION:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Build failed: {str(e)}")


@app.get("/chat")
def chat(query: str):
    print(f"CHAT CALLED with query={query!r}")
    index = STATE.get("index")
    print(f"STATE index type = {type(index)}, is None = {index is None}")

    if index is None:
        raise HTTPException(status_code=400, detail="Knowledge base not built yet.")

    try:
        print("Calling get_query_engine...")
        query_engine = get_query_engine(index)
        print(f"query_engine type = {type(query_engine)}, is None = {query_engine is None}")

        if query_engine is None:
            raise HTTPException(status_code=500, detail="Failed to create query engine.")

        print("Calling query_engine.query...")
        response = query_engine.query(query)
        print(f"response type = {type(response)}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("CHAT EXCEPTION:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    def stream():
        for chunk in response.response_gen:
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index_page():
    return FileResponse("static/index.html")

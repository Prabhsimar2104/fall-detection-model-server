# server.py
import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from inference_module import infer_frame_np, infer_video_path, load_model_if_needed
from dotenv import load_dotenv
import uvicorn
import tempfile

load_dotenv()

API_KEY = os.getenv("MODEL_API_KEY", "changeme")
# Ensure model loaded on start
load_model_if_needed()

app = FastAPI(title="Fall Detection Model Server")

def check_api_key(key: str):
    return key == API_KEY

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer/frame")
async def infer_frame(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    if not check_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    content = await file.read()
    # write to temp file and call inference
    import numpy as np
    import cv2
    import io
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    result = infer_frame_np(img)
    return JSONResponse(content=result)

@app.post("/infer/video")
async def infer_video(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    if not check_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = infer_video_path(tmp_path)
        return JSONResponse(content=result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

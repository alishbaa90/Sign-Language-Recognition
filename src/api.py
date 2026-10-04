from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
import shutil
import os
import uuid

from pipeline_core import process_video
from logger_config import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Sign Language Recognition API",
    description="Upload a video of hand gestures and get back a generated sentence with bilingual audio.",
    version="1.0.0"
)

UPLOAD_DIR = "../uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recognize-signs")
async def recognize_signs(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Please upload a .mp4, .mov, or .avi video file.")

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Received video upload: {temp_filename}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded video.")

    try:
        result = await run_in_threadpool(process_video, temp_path)
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if result["error"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return JSONResponse(content={
    "collected_signs": [str(s) for s in result["collected_signs"]],
    "english_sentence": result["english_sentence"],
    "german_sentence": result["german_sentence"],
    "low_confidence": bool(result["low_confidence"]),
    "audio_english_url": "/audio/english",
    "audio_german_url": "/audio/german",
})


@app.get("/audio/english")
def get_english_audio():
    path = "../outputs/sentence_en.mp3"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="English audio not found. Run /recognize-signs first.")
    return FileResponse(path, media_type="audio/mpeg", filename="sentence_en.mp3")


@app.get("/audio/german")
def get_german_audio():
    path = "../outputs/sentence_de.mp3"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="German audio not found. Run /recognize-signs first.")
    return FileResponse(path, media_type="audio/mpeg", filename="sentence_de.mp3")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
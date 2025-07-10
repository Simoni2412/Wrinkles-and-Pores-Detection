from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from PIL import Image
import numpy as np
import cv2
import io
import datetime
import logging
import traceback
import os

# Your analyzer functions
from analyzer import (
    analyze_skin_type_patches,
    detect_skin_tone,
    analyze_wrinkles,
    detect_dark_circles_otsu,
    analyze_pores
)

app = FastAPI()

# Logging configuration
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# File handler (debug logs saved here)
file_handler = logging.FileHandler("app_debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logging.getLogger("").addHandler(file_handler)

# Set log level from environment or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger(__name__)

# CORS origins
origins = [
    "https://f215b48d-daff-427d-89ef-7ee6c22b514d.lovableproject.com",  # update with your frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Middleware to log incoming requests and outgoing responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        try:
            body = await request.body()
            logger.debug(f"Request Body (truncated): {body[:1000]}")  # truncate to first 1000 bytes
        except Exception as e:
            logger.warning(f"Could not read request body: {e}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Exception during request processing: {e}")
            logger.error(traceback.format_exc())
            raise
        logger.info(f"Response status: {response.status_code}")
        return response

app.add_middleware(LoggingMiddleware)


def analyze_image_np(image: np.ndarray) -> dict:
    timestamp = datetime.datetime.now().isoformat()
    logger.info("Starting image analysis")

    try:
        logger.debug("Calling analyze_skin_type_patches()")
        skin_type = analyze_skin_type_patches(image)

        logger.debug("Calling detect_skin_tone()")
        skin_tone = detect_skin_tone(image)

        logger.debug("Calling analyze_wrinkles()")
        wrinkle_score, skin_age = analyze_wrinkles(image)

        logger.debug("Calling detect_dark_circles_otsu()")
        dark_circle_score = detect_dark_circles_otsu(image)

        logger.debug("Calling analyze_pores()")
        pores_score = analyze_pores(image)

        result = {
            "skin_type": skin_type,
            "skin_tone": skin_tone,
            "wrinkle_score": wrinkle_score,
            "skin_age": skin_age,
            "dark_circle_score": dark_circle_score[2],
            "pores_score": pores_score[3],  # Assuming third item is score
            "timestamp": timestamp
        }
        logger.debug(f"Analysis result: {result}")
        logger.info("Completed image analysis")
        return result

    except Exception as e:
        logger.error("Error during image analysis")
        logger.error(traceback.format_exc())
        raise


@app.post("/analyze-photo/")
async def analyze_photo(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: filename={file.filename}, content_type={file.content_type}")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.debug("Image loaded and converted to RGB")

        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.debug("Image converted to OpenCV BGR format")

        result = analyze_image_np(cv_img)
        logger.info("Image analysis completed successfully")

        return {"status": "success", "analysis": result}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing image: {e}")
        logger.error(tb)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": tb}
        )

@app.get("/job-status/{job_id}")

async def get_job_status(job_id: str):
    # Assuming supabase_client is initialized and available in your real code
    job = await supabase_client.from_('analysis_jobs').select('*').eq('id', job_id).single()
    if not job:
        logger.warning(f"Job with id {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    logger.info(f"Returning status for job id {job_id}: {job.status}")
    return {
        "jobId": job.id,
        "status": job.status,
        "result": job.result_data,
        "error": job.error_message
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)

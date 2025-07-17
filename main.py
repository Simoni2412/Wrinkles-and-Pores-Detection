from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import numpy as np
import cv2
import io
import datetime
import logging
import traceback
import os

# === Output Directory Setup ===
base_output_dir = "Output"
wrinkle_output_dir = os.path.join(base_output_dir, "predicted_wrinkle_masks")
pore_output_dir = os.path.join(base_output_dir, "predicted_pore_masks")
dark_circle_dir = os.path.join(base_output_dir, "predicted_dark_circle_masks")

# Create directories if they don't exist
os.makedirs(wrinkle_output_dir, exist_ok=True)
os.makedirs(pore_output_dir, exist_ok=True)
os.makedirs(dark_circle_dir, exist_ok=True)

# Your analyzer functions
from analyzer import (
    analyze_skin_type_patches,
    detect_skin_tone,
    analyze_wrinkles,
    detect_dark_circles_otsu,
    analyze_pores,
    overlay_mask
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


def analyze_image_np(image) -> dict:
    timestamp = datetime.datetime.now().isoformat()
    logger.info("Starting image analysis")
    logger.info(f"Image: {image}")

    try:
        logger.debug("Calling analyze_skin_type_patches()")
        skin_type = analyze_skin_type_patches(image)
        # logger.info(f"Skin type: {skin_type}")

        logger.debug("Calling detect_skin_tone()")
        skin_tone = detect_skin_tone(image)
        # logger.info(f"Skin tone: {skin_tone}")

        logger.debug("Calling analyze_wrinkles()")
        wrinkle_score, skin_age, binary_mask = analyze_wrinkles(image)
        # logger.info(f"Wrinkle score: {wrinkle_score}")
        # logger.info(f"Skin age: {skin_age}")

        logger.debug("Calling detect_dark_circles_otsu()")
        dark_circle_score = detect_dark_circles_otsu(image)
        # logger.info(f"Dark circle score: {dark_circle_score}")

        logger.debug("Calling analyze_pores()")
        pores_score = analyze_pores(image)
        # logger.info(f"Pores score: {pores_score}")

        result = {
            "skin_type": skin_type,
            "skin_tone": skin_tone,
            "wrinkle_score": wrinkle_score,
            "skin_age": skin_age,
            "dark_circle_score": dark_circle_score[2],
            "pores_score": pores_score[2],  # Assuming third item is score
            "timestamp": timestamp
        }
        logger.info(f"Analysis result: {result}")
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
        cv_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        # image = Image.open(io.BytesIO(contents)).convert("RGB")
        # logger.debug("Image loaded and converted to RGB")

        # cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # logger.debug("Image converted to OpenCV BGR format")

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

@app.post("/overlay-photo/")
async def overlay_photo(file: UploadFile = File(...), overlay_type: str = "wrinkle"):
    contents = await file.read()
    cv_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    filename = file.filename
    filename_wo_ext = os.path.splitext(os.path.basename(filename))[0]

    # Choose overlay mask path
    if overlay_type == "wrinkle":
        overlay_path = os.path.join(wrinkle_output_dir, f"wrinkle_mask_{filename_wo_ext}.png")
        overlay_color = (0, 0, 255)  # Red
    elif overlay_type == "pores":
        overlay_path = os.path.join(pore_output_dir, f"pores_mask_{filename_wo_ext}.png")
        overlay_color = (255, 0, 255)  # Magenta
    elif overlay_type == "dark_circles":
        overlay_path = os.path.join(dark_circle_dir, f"darkcircles_mask_{filename_wo_ext}.png")
        overlay_color = (0, 255, 255)  # Yellowish
    else:
        raise HTTPException(status_code=400, detail="Invalid overlay type")

    # Check if mask exists
    if not os.path.exists(overlay_path):
        raise HTTPException(status_code=404, detail="Overlay not found. Analyze first.")

    # Load the binary mask
    mask = cv2.imread(overlay_path, cv2.IMREAD_GRAYSCALE)  # Binary mask

    # Apply overlay
    overlayed_img = overlay_mask(cv_img, mask, color=overlay_color, alpha=0.5)

    # Encode overlayed image to PNG
    success, buffer = cv2.imencode(".png", overlayed_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode overlay image.")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


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

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io
import datetime
import logging
import traceback

# Your analyzer functions
from analyzer import (
    analyze_skin_type_patches,
    detect_skin_tone,
    analyze_wrinkles,
    detect_dark_circles_otsu,
    analyze_pores
)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

origins = [
    "https://f215b48d-daff-427d-89ef-7ee6c22b514d.lovableproject.com",  # update to your frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

def analyze_image_np(image: np.ndarray) -> dict:
    timestamp = datetime.datetime.now().isoformat()
    logging.info("Starting image analysis")

    try:
        skin_type = analyze_skin_type_patches(image)
        skin_tone = detect_skin_tone(image)
        wrinkle_score, skin_age = analyze_wrinkles(image)
        dark_circle_score = detect_dark_circles_otsu(image)
        pores_score = analyze_pores(image)

        result = {
            "skin_type": skin_type,
            "skin_tone": skin_tone,
            "wrinkle_score": wrinkle_score,
            "skin_age": skin_age,
            "dark_circle_score": dark_circle_score[2],
            "pores_score": pores_score[3],
            "timestamp": timestamp
        }
        logging.info("Completed image analysis")
        return result

    except Exception as e:
        logging.error("Error during image analysis")
        logging.error(traceback.format_exc())
        raise e

@app.post("/analyze-photo/")
async def analyze_photo(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = analyze_image_np(cv_img)
        return {"status": "success", "analysis": result}

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error processing image: {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": tb}
        )

# For quick local testing with Uvicorn:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)



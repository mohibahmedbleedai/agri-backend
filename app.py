import io
import pathlib
import sys

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI()

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    img_np = np.array(image)

    results = model(img_np)

    results.render()

    img_with_boxes = Image.fromarray(results.ims[0])

    img_byte_arr = io.BytesIO()
    img_with_boxes.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

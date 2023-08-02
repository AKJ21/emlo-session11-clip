import torch
import io

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/predict")
async def predict(file: Annotated[bytes, File()], text: str):
    img = Image.open(io.BytesIO(file))
    img = img.convert("RGB")
    img_np = np.array(img)

    # print(f"shape = {img_np.shape}")

    inputs = processor(text=text, images=img_np, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)[0].numpy() # we can take the softmax to get the label probabilities
        dic={}
        for i in range(len(text)):
            dic[text[i]] = round(probs[i],2)

    return dic

@app.get("/health")
async def health():
    return {"message": "ok"}
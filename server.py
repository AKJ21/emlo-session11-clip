import torch
import io

import numpy as np
from typing import Annotated
from fastapi import FastAPI, File
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
    list_of_texts = text.split(", ")

    input_text = processor(list_of_texts, return_tensors="pt", padding=True)
    input_image = processor(images=img, return_tensors="pt", padding=True)

    text_features = model.get_text_features(**input_text)
    img_features = model.get_image_features(**input_image)

    image_embeds = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        probs = logits_per_image.softmax(dim=1).squeeze(0).tolist() # we can take the softmax to get the label probabilities

    return {list_of_texts[i]: round(probs[i],2) for i in range(len(list_of_texts))}

@app.get("/health")
async def health():
    return {"message": "ok"}
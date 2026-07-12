import os
import io
import re
import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
async def answer_image(req: ImageRequest):
    try:
        # Decode image
        image = Image.open(
            io.BytesIO(base64.b64decode(req.image_base64))
        ).convert("RGB")

        prompt = f"""
You are an expert OCR and visual question answering system.

Carefully inspect the image.

Answer the user's question accurately.

Question:
{req.question}

Instructions:
- Read all text, tables, charts, receipts, invoices, and diagrams carefully.
- Perform calculations if the question requires them.
- Return ONLY the final answer.
- No explanation.
- No markdown.
- No labels.
- If the answer is numeric, return only the number.
- Do not include currency symbols.
- Do not include commas in numbers unless required.
- Remove units unless explicitly requested.
"""

        response = model.generate_content(
            [prompt, image],
            generation_config=GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
            ),
        )

        answer = response.text.strip()

        # Normalize common formatting
        answer = answer.replace("$", "")
        answer = answer.replace("₹", "")
        answer = answer.replace("€", "")
        answer = answer.replace(",", "")
        answer = answer.replace("\n", " ").strip()

        # Collapse multiple spaces
        answer = re.sub(r"\s+", " ", answer)

        return {"answer": str(answer)}

    except Exception as e:
        return {"answer": str(e)}

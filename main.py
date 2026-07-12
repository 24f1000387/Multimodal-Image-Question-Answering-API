import os
import io
import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

# Enable CORS
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
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        prompt = f"""
You are an OCR and document question-answering assistant.

Question:
{req.question}

Rules:
- Return ONLY the answer.
- If the answer is numeric, return only the number.
- No currency symbols.
- No units.
- No explanation.
"""

        response = model.generate_content([prompt, image])

        return {
            "answer": response.text.strip()
        }

    except Exception as e:
        return {
            "answer": "",
            "error": str(e)
        }

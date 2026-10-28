"""
Parser for uploaded Job Descriptions: supports PDF (text & scanned fallback), DOCX (paragraphs & tables), and Images (PNG/JPG/WEBP).
Uses Groq Vision (model from env GROQ_VISION_MODEL, default: qwen/qwen3.8-27b) for image OCR.
"""
import base64
import io
import logging
import os
import re
from typing import Optional
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger("jd_parser")


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if l and not l.isdigit()]  # drop blanks and bare page numbers
    return "\n".join(lines)


def _extract_text_from_image(data: bytes, mime_type: str = "image/png") -> str:
    """Extracts text from image data using Groq vision model (GROQ_VISION_MODEL)."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured for image OCR extraction.")

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        # Ensure dimensions >= 32x32 required by Groq vision models
        if img.width < 32 or img.height < 32:
            img = img.resize((max(32, img.width), max(32, img.height)))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()
        mime_type = "image/jpeg"
    except Exception as img_err:
        logger.debug(f"Pillow image processing skipped/fallback: {img_err}")
        img_bytes = data

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()

    import groq

    client = groq.Groq(api_key=api_key, timeout=30.0)

    prompt = (
        "Extract all readable text from this job description image. "
        "Preserve role title, company name, requirements, responsibilities, and qualifications. "
        "Return ONLY the extracted text with no introductory or concluding conversational remarks."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        extracted = (resp.choices[0].message.content or "").strip()
        if not extracted:
            raise ValueError("No text could be extracted from the uploaded image.")
        return extracted
    except Exception as exc:
        logger.error(f"Groq vision OCR failed on model '{model}': {exc}")
        raise ValueError(f"Could not read text from image ({type(exc).__name__}). Please paste the job description text instead.")


def extract(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            cleaned = _clean(text)

            # Check for scanned PDF with no text layer
            if len(cleaned) < 50:
                logger.info("PDF has insufficient text layer; attempting embedded image extraction via vision...")
                image_parts = []
                for p_idx, page in enumerate(reader.pages[:3]):  # check up to first 3 pages
                    try:
                        for img_obj in page.images:
                            img_text = _extract_text_from_image(img_obj.data, mime_type="image/png")
                            if img_text:
                                image_parts.append(img_text)
                    except Exception as e:
                        logger.warning(f"Failed extracting images from PDF page {p_idx}: {e}")

                if image_parts:
                    return _clean("\n".join(image_parts))

                raise ValueError(
                    "Scanned PDF detected with no readable text layer. Please paste the job description text or upload as an image."
                )

            return cleaned
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError("Could not parse PDF file. If scanned, paste the text directly.")

    elif name.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for t in doc.tables:
                for row in t.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
            return _clean("\n".join(parts))
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ValueError("Could not parse DOCX file.")

    elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
        mime = "image/png" if name.endswith(".png") else "image/webp" if name.endswith(".webp") else "image/jpeg"
        raw_text = _extract_text_from_image(data, mime_type=mime)
        return _clean(raw_text)

    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, PNG, JPG, or WEBP file.")

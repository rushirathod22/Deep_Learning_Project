import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

_model_name = "openai/clip-vit-base-patch32"
_processor = None
_model = None

def _get_model():
    global _processor, _model
    if _model is None:
        _processor = CLIPProcessor.from_pretrained(_model_name)
        _model = CLIPModel.from_pretrained(_model_name)
        _model.eval()
    return _processor, _model

def verify_image(claim: str, image_path: str, threshold_support: float = 0.25, threshold_contradict: float = 0.18, visually_verifiable: bool = True) -> tuple[str, float]:
    """
    Verifies a claim against an image using CLIP cosine similarity.
    
    Returns:
        (label, score)
        label is one of: "SUPPORT", "CONTRADICT", "NO_EVIDENCE"
    """
    if not visually_verifiable or not image_path:
        return "NO_EVIDENCE", 0.0
        
    abs_path = image_path
    if not os.path.isabs(abs_path):
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        abs_path = str(project_root / image_path)

    if not os.path.exists(abs_path):
        return "NO_EVIDENCE", 0.0

    try:
        image = Image.open(abs_path).convert("RGB")
    except Exception:
        return "NO_EVIDENCE", 0.0

    processor, model = _get_model()
    
    inputs = processor(text=[claim], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    cosine_sim = torch.matmul(text_embeds, image_embeds.t()).item()
    
    if cosine_sim > threshold_support:
        return "SUPPORT", cosine_sim
    elif cosine_sim < threshold_contradict:
        return "CONTRADICT", cosine_sim
    else:
        return "NO_EVIDENCE", cosine_sim


_CANDIDATE_LABELS = [
    "Luxury Swiss watch chronometer timepiece Rolex Omega",
    "Apple AirPods wireless earbuds with charging case",
    "Sony wireless noise cancelling headphones",
    "Nike athletic running sneakers shoes",
    "Adidas athletic running sneakers shoes",
    "Apple iPhone smartphone with camera",
    "Samsung Galaxy smartphone mobile phone",
    "HP Pavilion laptop computer notebook",
    "Apple MacBook laptop computer notebook",
    "Dell laptop computer notebook",
    "Smartwatch fitness tracker watch band",
    "Luxury designer handbag purse leather bag",
    "Mulberry silk bedding sheet set",
    "Action camera photography equipment GoPro",
    "Coffee espresso maker machine"
]

def _extract_ocr_from_image(abs_path: str) -> dict:
    """
    Extract text, brand, price, and claims from promotional images using native Windows OCR.
    """
    try:
        import winocr
        import re
        import concurrent.futures
        from PIL import Image

        pil_img = Image.open(abs_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(winocr.recognize_pil_sync, pil_img, 'en-US')
            ocr_res = future.result()
        lines = [line.get('text', '').strip() for line in ocr_res.get('lines', []) if line.get('text', '').strip()]
        full_text = ' '.join(lines)

        if not lines or len(full_text) < 5:
            return None

        # Filter out generic button & badge text to isolate product name
        IGNORE_PHRASES = {
            'buy now', 'order now', 'shop now', 'click here', 'limited time offer', 'limited time',
            'free shipping', 'easy returns', 'fast delivery', 'cash on delivery',
            'original swiss movement', 'water resistant', 'stainless steel',
            'international warranty', 'year warranty', 'warranty', 'official store'
        }

        title_candidates = []
        for line in lines:
            cleaned = re.sub(r'[^a-zA-Z0-9\s\.\-]', '', line).strip()
            lower = cleaned.lower()
            if not lower or any(lower == ign or lower.startswith(ign) for ign in IGNORE_PHRASES):
                continue
            if re.match(r'^\$?[0-9,]+(?:\.[0-9]{2})?\s*(?:only|off)?$', lower):
                continue
            if len(cleaned.split()) >= 1 and not re.match(r'^[0-9\W]+$', cleaned):
                title_candidates.append(line.strip())

        # Extract prices and discounts
        prices = re.findall(r'[\$₹£€]\s*([0-9,]+(?:\.[0-9]{2})?)', full_text)
        discounts = re.findall(r'([0-9]{1,2}%)\s*OFF', full_text, re.IGNORECASE)

        # Prominent product title
        product_title = ' '.join(title_candidates[:3]) if title_candidates else lines[0]
        # Clean double spaces
        product_title = ' '.join(product_title.split())

        price_str = f"${prices[0]}" if prices else ""
        if '₹' in full_text and prices:
            price_str = f"₹{prices[0]}"

        # Compose claim
        if price_str:
            suggested_claim = f"{product_title} for {price_str} is genuine original"
        else:
            suggested_claim = f"{product_title} genuine original product listing"

        return {
            "detected": True,
            "label": product_title,
            "confidence": 99.0,
            "suggested_claim": suggested_claim,
            "ocr_text": full_text[:400],
            "price_str": price_str,
            "has_discount": bool(discounts),
            "discount": discounts[0] if discounts else None
        }
    except Exception as e:
        print(f"  [OCR Engine Notice] {e}")
        return None


def detect_product_from_image(image_path: str) -> dict:
    """
    Hybrid Deep Learning Vision & OCR Pipeline:
    1. First runs Deep OCR (Windows Media OCR) to parse text, brand, and pricing from flyers/ads.
    2. If no prominent text exists, runs OpenAI CLIP zero-shot classification across rich product categories.
    """
    abs_path = image_path
    if not os.path.isabs(abs_path):
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        abs_path = str(project_root / image_path)
    
    if not os.path.exists(abs_path):
        return {"detected": False, "label": "Product Listing", "confidence": 0.0, "suggested_claim": "Authentic product listing"}

    # Step 1: Run Deep OCR
    ocr_result = _extract_ocr_from_image(abs_path)
    if ocr_result and len(ocr_result.get("label", "")) >= 3:
        return ocr_result

    # Step 2: Fallback to OpenAI CLIP Zero-Shot Vision Classifier
    try:
        image = Image.open(abs_path).convert("RGB")
        processor, model = _get_model()
        inputs = processor(text=_CANDIDATE_LABELS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()
            
        top_idx = int(torch.argmax(logits_per_image, dim=1).item())
        best_label = _CANDIDATE_LABELS[top_idx]
        best_prob = probs[top_idx] if isinstance(probs, list) else float(probs)
        return {
            "detected": True,
            "label": best_label,
            "confidence": round(best_prob * 100, 1),
            "suggested_claim": f"{best_label} genuine product listing"
        }
    except Exception as e:
        print(f"  [Image Classifier Notice] {e}")
        return {"detected": False, "label": "Product Listing", "confidence": 0.0, "suggested_claim": "Authentic product listing"}


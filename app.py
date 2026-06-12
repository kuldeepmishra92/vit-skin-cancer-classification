import gradio as gr
import torch
import torch.nn.functional as F
from transformers import pipeline, CLIPModel, CLIPProcessor
from PIL import Image

# ── Device ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load ViT skin cancer classifier ──────────────────────────────
print("Loading ViT classifier...")
classifier = pipeline(
    "image-classification",
    model="Kuldeepmishra3/vit-large-skin-cancer-ham10000",
    device=0 if device == "cuda" else -1,
)
print("ViT classifier ready.")

# ── Load CLIP gate ────────────────────────────────────────────────
print("Loading CLIP gate...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP gate ready.")

# ── Skin validation prompts ───────────────────────────────────────
SKIN_PROMPTS = [
    "a dermoscopy image of human skin",
    "a close-up photo of a skin lesion",
    "a medical photo of a mole or skin growth",
    "a dermatological image of human body skin",
    "a macro photo of human skin texture",
]
ALL_PROMPTS = SKIN_PROMPTS + ["a random non-medical image"]
SKIN_THRESHOLD = 0.40

# ── Class info ────────────────────────────────────────────────────
CLASS_INFO = {
    "akiec": {
        "full": "Actinic Keratoses",
        "info": "A rough, scaly patch caused by years of sun exposure. Can develop into skin cancer if untreated."
    },
    "bcc": {
        "full": "Basal Cell Carcinoma",
        "info": "The most common type of skin cancer. Rarely spreads but needs treatment."
    },
    "bkl": {
        "full": "Benign Keratosis",
        "info": "A non-cancerous skin growth. Very common, especially in older adults."
    },
    "df": {
        "full": "Dermatofibroma",
        "info": "A harmless skin growth that often appears on the legs. Usually no treatment needed."
    },
    "mel": {
        "full": "Melanoma",
        "info": "The most serious type of skin cancer. Early detection is critical."
    },
    "nv": {
        "full": "Melanocytic Nevi",
        "info": "Common moles. Usually harmless but should be monitored for changes."
    },
    "vasc": {
        "full": "Vascular Lesions",
        "info": "Abnormalities of blood vessels in the skin. Usually benign."
    },
}


# ── Skin gate function ────────────────────────────────────────────
@torch.no_grad()
def is_skin_image(pil_image):
    """Returns (passed: bool, skin_score: float)"""
    inputs = clip_processor(
        text=ALL_PROMPTS,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    outputs = clip_model(**inputs)
    probs = F.softmax(outputs.logits_per_image, dim=-1).squeeze()
    skin_score = probs[:len(SKIN_PROMPTS)].sum().item()
    return skin_score >= SKIN_THRESHOLD, skin_score


# ── Main predict function ─────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image."

    # Step 1 — Validate it's a skin image
    passed, skin_score = is_skin_image(image)

    if not passed:
        return (
            "## ❌ Invalid Image\n\n"
            f"This does not appear to be a skin or dermoscopy image.\n\n"
            f"**Skin confidence score:** `{skin_score:.2f}` (minimum required: `{SKIN_THRESHOLD}`)\n\n"
            "---\n\n"
            "Please upload a **dermoscopy or skin lesion image** for classification.\n\n"
            "> This model is trained only on skin lesion images (HAM10000 dataset). "
            "Uploading unrelated images (e.g. cars, animals, food) will be rejected."
        )

    # Step 2 — Run ViT skin cancer classifier
    results = classifier(image, top_k=7)

    output = f"*Skin confidence: `{skin_score:.2f}` — image validated*\n\n---\n\n"

    for i, r in enumerate(results):
        label = r["label"]
        score = r["score"]
        info = CLASS_INFO.get(label, {})
        full_name = info.get("full", label)
        description = info.get("info", "")

        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)

        if i == 0:
            output += f"### Top prediction: {full_name} ({score:.1%})\n"
            output += f"> {description}\n\n"
            output += "---\n\n"
            output += "**All predictions:**\n\n"

        output += f"`{label}` — {full_name}\n"
        output += f"{bar} {score:.1%}\n\n"

    output += "\n---\n"
    output += "⚠️ **This is not a medical diagnostic tool. Always consult a dermatologist.**"

    return output


# ── Gradio UI ─────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a dermoscopy image"),
    outputs=gr.Markdown(label="Prediction"),
    title="Skin Cancer Classifier — ViT-Large",
    description="""
Fine-tuned **google/vit-large-patch16-224** on the HAM10000 dataset (10,000+ dermoscopy images, 7 classes).
Achieved **92.74% accuracy** and **92.60% weighted F1** on the validation set.

Upload a dermoscopy skin image to classify it into one of 7 categories.

> ⚠️ This is a portfolio/learning project — not a medical tool. Do not use for real diagnosis.

> 🛡️ Non-skin images are automatically rejected by a CLIP-based validation gate.
""",
    examples=[],
    theme=gr.themes.Soft(),
)

demo.launch()

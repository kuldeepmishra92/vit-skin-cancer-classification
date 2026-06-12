"""
skin_validation_gate.py
=======================
CLIP-based skin-image validation gate for the ViT skin cancer classifier.

PROBLEM
-------
Our ViT model was fine-tuned only on dermoscopy skin lesion images.
Because it has no concept of "non-skin" inputs, feeding it a random image
(a car, a dog, a landscape) still produces a confident-looking skin cancer
prediction — which is misleading and potentially dangerous.

SOLUTION
--------
Before calling the ViT classifier we run a CLIP (Contrastive Language-Image
Pre-Training) model as a zero-shot gating step:

  1. CLIP compares the input image against a set of text prompts.
  2. If the skin-related prompts score below a threshold → reject with warning.
  3. Only images that pass the gate are forwarded to the ViT classifier.

No additional training required — CLIP's rich visual-language embeddings
generalise to medical images out of the box.

HOW TO USE
----------
Copy-paste the cells below into your Colab notebook (after Step 9),
OR import this module:

    from skin_validation_gate import is_skin_image, predict_skin_image

DEPENDENCIES (already installed in the notebook environment)
------------------------------------------------------------
    pip install transformers torch pillow matplotlib requests
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# ─── Configuration ────────────────────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"   # ~340 MB, fast, zero-shot capable

# Positive prompts — what a dermoscopy / skin lesion image looks like
SKIN_PROMPTS = [
    "a dermoscopy image of human skin",
    "a close-up photo of a skin lesion",
    "a medical photo of a mole or skin growth",
    "a dermatological image of human body skin",
    "a macro photo of human skin texture",
]

# Negative prompts — common non-skin image categories to guard against
NON_SKIN_PROMPTS = [
    "a photo of a car or vehicle",
    "a photo of an animal or pet",
    "a landscape or nature photo",
    "a photo of food or drink",
    "a photo of a building or architecture",
    "a photo of a person's face or portrait",
    "a photo of an object or product",
    "a digital artwork or illustration",
]

ALL_PROMPTS = SKIN_PROMPTS + NON_SKIN_PROMPTS

# Tune between 0.3 (loose) and 0.6 (strict). 0.4 works well in practice.
# Increase if you get false positives (random images slipping through).
# Decrease if valid skin images are being rejected.
SKIN_CONFIDENCE_THRESHOLD = 0.40


# ─── Load CLIP model (call once at startup) ───────────────────────────────────
def load_clip_gate(device=None):
    """Load the CLIP gating model. Returns (clip_model, clip_processor)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP gate on {device}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model.eval()
    print("CLIP gate ready.")
    return clip_model, clip_processor, device


# ─── Gate function ────────────────────────────────────────────────────────────
@torch.no_grad()
def is_skin_image(
    pil_image,
    clip_model,
    clip_processor,
    device,
    threshold=SKIN_CONFIDENCE_THRESHOLD,
    verbose=True,
):
    """
    Use CLIP to verify that an image is a skin / dermoscopy image.

    Parameters
    ----------
    pil_image      : PIL.Image
    clip_model     : CLIPModel loaded via load_clip_gate()
    clip_processor : CLIPProcessor loaded via load_clip_gate()
    device         : torch.device
    threshold      : float  — min aggregate skin-prompt probability to pass
    verbose        : bool   — print per-prompt scores

    Returns
    -------
    is_skin    : bool
    skin_score : float  — aggregate probability for all skin prompts
    top_label  : str    — the single most-matching text prompt
    """
    inputs = clip_processor(
        text=ALL_PROMPTS,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = clip_model(**inputs)
    # logits_per_image: shape [1, num_prompts]
    probs = F.softmax(outputs.logits_per_image, dim=-1).squeeze()  # [num_prompts]

    # Sum probabilities over skin prompts only
    n_skin = len(SKIN_PROMPTS)
    skin_score = probs[:n_skin].sum().item()
    top_idx = probs.argmax().item()
    top_label = ALL_PROMPTS[top_idx]

    passed = skin_score >= threshold

    if verbose:
        status = "✅ PASSED" if passed else "❌ REJECTED"
        print(f"[Skin Gate] {status}")
        print(f"  Skin score : {skin_score:.3f}  (threshold = {threshold})")
        print(f"  Best match : \"{top_label}\"")
        print()
        for i, (p, s) in enumerate(zip(ALL_PROMPTS, probs.tolist())):
            tag = "(skin)" if i < n_skin else "      "
            bar = "█" * int(s * 40)
            print(f"  {tag}  {s:.3f}  {bar}  {p}")

    return passed, skin_score, top_label


# ─── Safe inference wrapper ───────────────────────────────────────────────────
def predict_skin_image(
    image,
    classifier,           # HuggingFace pipeline("image-classification", model=model)
    clip_model,
    clip_processor,
    device,
    top_k=3,
    skin_threshold=SKIN_CONFIDENCE_THRESHOLD,
    class_full_names=None,
):
    """
    Safe skin cancer inference with a CLIP validation gate.

    Flow
    ----
    1. Run CLIP zero-shot gate to verify the image is a skin/dermoscopy image.
    2. If REJECTED  → display a clear warning panel and return None.
    3. If ACCEPTED  → run the fine-tuned ViT classifier and visualise results.

    Parameters
    ----------
    image           : PIL.Image
    classifier      : HuggingFace pipeline for image-classification
    clip_model      : CLIPModel (from load_clip_gate)
    clip_processor  : CLIPProcessor (from load_clip_gate)
    device          : torch.device
    top_k           : int   — number of top predictions to show
    skin_threshold  : float — gate threshold (default 0.40)
    class_full_names: list  — optional list mapping index → full class name

    Returns
    -------
    results : list of {label, score} dicts, or None if rejected
    """
    print("=" * 55)
    print("STEP 1: Skin image validation (CLIP gate)")
    print("=" * 55)
    passed, skin_score, top_label = is_skin_image(
        image, clip_model, clip_processor, device,
        threshold=skin_threshold,
    )

    # ── Rejection path ────────────────────────────────────────────────────────
    if not passed:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Input image", fontsize=12)

        # Rejection panel
        axes[1].set_facecolor("#FFF3F3")
        axes[1].text(
            0.5, 0.65,
            "❌  Not a skin image",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#C0392B", transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5, 0.46,
            f"Skin confidence: {skin_score:.2f}  /  threshold: {skin_threshold:.2f}",
            ha="center", va="center", fontsize=11,
            color="#7F8C8D", transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5, 0.29,
            f'Detected as:\n"{top_label}"',
            ha="center", va="center", fontsize=9,
            color="#555555", transform=axes[1].transAxes, style="italic",
        )
        axes[1].text(
            0.5, 0.10,
            "Please upload a dermoscopy or\nskin lesion image for classification.",
            ha="center", va="center", fontsize=9,
            color="#2C3E50", transform=axes[1].transAxes,
        )
        axes[1].axis("off")
        plt.suptitle(
            "Classification Rejected — Not a Skin Image",
            fontsize=13, fontweight="bold", color="#C0392B",
        )
        plt.tight_layout()
        plt.show()

        print()
        print("[ERROR] Classification aborted.")
        print(f"        Image does not appear to be a skin/dermoscopy image.")
        print(f"        Skin confidence = {skin_score:.3f}  (need >= {skin_threshold})")
        return None

    # ── Acceptance path — run ViT classifier ──────────────────────────────────
    print()
    print("=" * 55)
    print("STEP 2: Skin cancer classification (ViT-Large)")
    print("=" * 55)
    results = classifier(image, top_k=top_k)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title(
        f"Input image\n[Gate ✅  skin score: {skin_score:.2f}]", fontsize=10
    )

    labels_plot = [r["label"] for r in results]
    scores_plot = [r["score"] for r in results]
    colors = ["#378ADD" if i == 0 else "#B5D4F4" for i in range(len(results))]
    bars = axes[1].barh(labels_plot, scores_plot, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Confidence")
    axes[1].set_title(f"Top-{top_k} Skin Cancer Predictions")
    for bar, score in zip(bars, scores_plot):
        axes[1].text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}", va="center", fontsize=11,
        )

    plt.tight_layout()
    plt.show()

    print("\nPredictions:")
    for r in results:
        name = ""
        if class_full_names:
            # Try to look up the full name if label is a short code
            pass  # label is already returned by the pipeline
        print(f"  {r['label']:8s}: {r['score']:.4f}")

    return results


# ─── Quick demo (run as script) ───────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone demo — loads the saved model and tests the gate.
    Run in your Colab or local env:

        python skin_validation_gate.py
    """
    from transformers import (
        ViTForImageClassification,
        ViTImageProcessor,
        pipeline as hf_pipeline,
    )
    from datasets import load_dataset
    import requests
    from io import BytesIO
    from PIL import Image

    SAVE_PATH = "./vit-large-ham10000-final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ViT skin cancer classifier
    print("Loading ViT classifier...")
    loaded_model = ViTForImageClassification.from_pretrained(SAVE_PATH)
    loaded_processor = ViTImageProcessor.from_pretrained(SAVE_PATH)
    classifier = hf_pipeline(
        "image-classification",
        model=loaded_model,
        feature_extractor=loaded_processor,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Load CLIP gate
    clip_model, clip_processor, device = load_clip_gate(device)

    CLASS_FULL_NAMES = [
        "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
        "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions",
    ]

    # ── Demo 1: skin image (should PASS) ──────────────────────────────────────
    print("\n" + "─" * 55)
    print("Demo 1: Dermoscopy skin image (should PASS gate)")
    print("─" * 55)
    ds = load_dataset("marmal88/skin_cancer", split="test[:1]")
    skin_img = ds[0]["image"]
    predict_skin_image(skin_img, classifier, clip_model, clip_processor, device,
                       class_full_names=CLASS_FULL_NAMES)

    # ── Demo 2: car image (should be REJECTED) ────────────────────────────────
    print("\n" + "─" * 55)
    print("Demo 2: Car image (should be REJECTED by gate)")
    print("─" * 55)
    try:
        car_url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/"
            "2019_Toyota_Corolla_sedan_%28facelift%2C_red%29%2C_front_8.15.19.jpg"
            "/320px-2019_Toyota_Corolla_sedan_%28facelift%2C_red%29%2C_front_8.15.19.jpg"
        )
        resp = requests.get(car_url, timeout=10)
        car_img = Image.open(BytesIO(resp.content)).convert("RGB")
        result = predict_skin_image(car_img, classifier, clip_model, clip_processor, device)
        if result is None:
            print("✅ Gate correctly rejected the non-skin image.")
    except Exception as e:
        print(f"Could not fetch demo image: {e}")

"""
Out-of-domain / invalid-image gate for the Plant Disease Detection app.

This module is intentionally standalone:
- It does NOT import or touch the Baseline CNN, its weights, or its
  [0, 255] float32 preprocessing pipeline.
- It does NOT import Streamlit — it can be unit-tested or reused on its
  own.
- It uses a pretrained CLIP model (zero-shot, no training required) to
  answer a single question: "does this image look like a plant leaf,
  more than it looks like anything else?" That is a domain-membership
  check, not a confidence/entropy measure on the 38-class output.

Design notes (see accompanying explanation for the full rationale):
- Confidence / entropy / top1-vs-top2 gap on the Baseline CNN's own
  output were deliberately NOT used, because a closed-set softmax
  classifier can be confidently wrong on inputs outside its training
  distribution. This gate never looks at the Baseline CNN's output at
  all — it runs *before* the CNN is ever called.
- The decision uses a margin between the best "leaf" prompt score and
  the best "non-leaf" prompt score, not a single absolute threshold.
  This is more robust than thresholding raw confidence because it's
  comparing the image against explicit alternative hypotheses, rather
  than asking "how sure is the model" in isolation.

Requirements:
    pip install transformers torch --break-system-packages
(First call downloads the "openai/clip-vit-base-patch32" weights from
Hugging Face and caches them locally; subsequent runs are offline.)
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from PIL import Image

# Prompts describing the in-domain subject (plant leaves, healthy or
# diseased). Multiple phrasings reduce sensitivity to CLIP's prompt
# wording quirks.
POSITIVE_PROMPTS: List[str] = [
    "a photo of a plant leaf",
    "a close-up photo of a leaf",
    "a photo of a diseased plant leaf with spots or discoloration",
    "a photo of a healthy green leaf",
]

# Prompts describing plausible out-of-domain uploads. Kept broad on
# purpose — the goal is to cover common "wrong photo" categories, not
# to enumerate every possible non-leaf object.
NEGATIVE_PROMPTS: List[str] = [
    "a photo of a car",
    "a photo of a person",
    "a photo of a building",
    "a photo of an animal",
    "a photo of food on a plate",
    "a photo of a random household object",
    "a blank, blurry, or empty image",
    "a screenshot of text or a document",
]

# Default decision margin: how much higher the best leaf-prompt score
# must be than the best non-leaf-prompt score for the image to pass.
# This is a plain config value — tune it against a small labeled sample
# of real leaf / non-leaf uploads, no retraining involved.
DEFAULT_MARGIN = 0.03

_MODEL_NAME = "openai/clip-vit-base-patch32"


@dataclass
class ValidityResult:
    is_valid: bool
    leaf_score: float
    best_negative_label: str
    best_negative_score: float
    margin_used: float

    @property
    def margin_achieved(self) -> float:
        return self.leaf_score - self.best_negative_score


@lru_cache(maxsize=1)
def _load_clip():
    """
    Lazily loads and caches the CLIP model + processor once per process.
    Kept separate from the Baseline CNN's tf.keras.models.load_model
    call entirely — different library (transformers/torch), different
    weights, different preprocessing.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    model.eval()
    return model, processor, torch


def check_leaf_validity(
    image: Image.Image,
    margin: float = DEFAULT_MARGIN,
) -> ValidityResult:
    """
    Zero-shot domain check: is `image` more similar to the leaf prompts
    than to any of the non-leaf prompts, by at least `margin`?

    This never touches the Baseline CNN — it's meant to run *before* it,
    as an independent gate. Raises no exception on a "not a leaf"
    verdict; callers decide how to present that (see app.py).
    """
    model, processor, torch = _load_clip()

    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

    inputs = processor(
        text=all_prompts,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # CLIP's own image-text similarity, softmax-normalized over the
        # provided prompt set (a closed comparison across our chosen
        # hypotheses — not the Baseline CNN's 38-class output).
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

    leaf_scores = probs[: len(POSITIVE_PROMPTS)]
    negative_scores = probs[len(POSITIVE_PROMPTS):]

    leaf_score = max(leaf_scores)
    best_negative_idx = max(range(len(negative_scores)), key=lambda i: negative_scores[i])
    best_negative_score = negative_scores[best_negative_idx]
    best_negative_label = NEGATIVE_PROMPTS[best_negative_idx]

    is_valid = (leaf_score - best_negative_score) >= margin

    return ValidityResult(
        is_valid=is_valid,
        leaf_score=leaf_score,
        best_negative_label=best_negative_label,
        best_negative_score=best_negative_score,
        margin_used=margin,
    )
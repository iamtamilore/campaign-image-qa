"""
compare_sam2_vs_grounded_sam.py - SAM2 alone (manual point) vs Grounding DINO
+ SAM2 ("Grounded-SAM", auto text-prompted box) on the same test image.

Test image: the real Mullay jacket photo (already has known-good coordinates
from the manual color-threshold masking done earlier this session, useful as
a sanity baseline to compare against).

Approach A - SAM2 alone: give it ONE manual point (hand-picked, same kind of
  manual coordinate work done all session via connected-components).
Approach B - Grounded-SAM: text prompt ("red logo text") -> Grounding DINO
  box -> SAM2 mask. Zero manual coordinates.

Saves both masks + overlays for visual comparison.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2

SAM2_ROOT = "/media/newtazer/38266955266914E2/ai_models/sam2"
GDINO_ROOT = "/media/newtazer/38266955266914E2/ai_models/groundingdino"
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, GDINO_ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

IMAGE_PATH = "/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png"
OUT_DIR = Path("/home/newtazer/Desktop/NCI/2ND/Machine Learning/CA2_mullay_qa/outputs/mullay_experiment/sam_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# from this session's earlier manual connected-components work: the wordmark
# letters center roughly at (490-597, 823-1614) in the 3000x3000 original.
MANUAL_POINT = np.array([[540, 1200]])  # one hand-picked point on the "MULLAY" text


def overlay_mask(image_rgb, mask, color=(0, 255, 255), alpha=0.6):
    # cyan on a black/red jacket stays visible; the old red-on-red overlay
    # was indistinguishable from the wordmark itself.
    out = image_rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    # also draw a solid outline so a thin/small mask isn't lost at low zoom
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 4)
    return out


def save_zoomed_crop(overlay_rgb, mask, out_path, pad=150):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, overlay_rgb.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, overlay_rgb.shape[0])
    Image.fromarray(overlay_rgb[y0:y1, x0:x1]).save(out_path)


def main():
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)

    # ---------- Approach A: SAM2 alone, manual point ----------
    print("[A] SAM2 alone (manual point)...")
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                             f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt", device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)
    masks_a, scores_a, _ = predictor.predict(
        point_coords=MANUAL_POINT, point_labels=np.array([1]), multimask_output=True)
    best_a = masks_a[np.argmax(scores_a)].astype(bool)
    print(f"  best score: {scores_a.max():.4f}, mask area: {best_a.sum()} px")
    overlay_a = overlay_mask(image_np, best_a)
    Image.fromarray(overlay_a).save(OUT_DIR / "A_sam2_manual_point.png")
    save_zoomed_crop(overlay_a, best_a, OUT_DIR / "A_sam2_manual_point_ZOOM.png")

    # ---------- Approach B: Grounding DINO -> box -> SAM2 ----------
    print("\n[B] Grounded-SAM (text prompt -> auto box -> SAM2)...")
    gdino_model = load_model(f"{GDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                              f"{GDINO_ROOT}/weights/groundingdino_swint_ogc.pth")
    image_source, image_transformed = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=gdino_model, image=image_transformed,
        caption="red logo text", box_threshold=0.3, text_threshold=0.25)
    print(f"  detections: {list(zip(phrases, [round(l.item(),3) for l in logits]))}")

    if len(boxes) == 0:
        print("  [FAIL] Grounding DINO found nothing for 'red logo text'")
        return

    # boxes are cxcywh normalized -> convert to xyxy pixels
    h, w = image_np.shape[:2]
    box = boxes[logits.argmax()]
    cx, cy, bw, bh = box.tolist()
    x0, y0 = (cx - bw/2) * w, (cy - bh/2) * h
    x1, y1 = (cx + bw/2) * w, (cy + bh/2) * h
    box_xyxy = np.array([x0, y0, x1, y1])
    print(f"  auto box (pixels): {box_xyxy.round(0)}")

    masks_b, scores_b, _ = predictor.predict(box=box_xyxy, multimask_output=True)
    best_b = masks_b[np.argmax(scores_b)].astype(bool)
    print(f"  best score: {scores_b.max():.4f}, mask area: {best_b.sum()} px")
    overlay_b = overlay_mask(image_np, best_b)
    Image.fromarray(overlay_b).save(OUT_DIR / "B_grounded_sam_auto.png")
    save_zoomed_crop(overlay_b, best_b, OUT_DIR / "B_grounded_sam_auto_ZOOM.png")
    # also save the raw auto-detected box on its own, so it's visible that
    # Grounding DINO (not a human) produced it
    box_vis = image_np.copy()
    cv2.rectangle(box_vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 6)
    Image.fromarray(box_vis[max(int(y0)-150,0):int(y1)+150, max(int(x0)-150,0):int(x1)+150]).save(OUT_DIR / "B_autobox_ZOOM.png")

    print(f"\nSaved both -> {OUT_DIR}")


if __name__ == "__main__":
    main()

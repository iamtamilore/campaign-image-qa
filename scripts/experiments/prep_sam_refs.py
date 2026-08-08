import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

SAM2_ROOT = "/media/newtazer/38266955266914E2/ai_models/sam2"
GDINO_ROOT = "/media/newtazer/38266955266914E2/ai_models/groundingdino"
sys.path.insert(0, SAM2_ROOT)
sys.path.insert(0, GDINO_ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

IMAGE_PATH = "/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png"
OUT_DIR = Path("/home/newtazer/Desktop/NCI/2ND/Machine Learning/CA2_mullay_qa/outputs/mullay_experiment/sam_refs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GroundingDINO's own auto-detected wordmark box from the M16.18 run
WORDMARK_BOX = (370, 810, 616, 1698)


def main():
    image_np = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    h, w = image_np.shape[:2]

    # ---------- test #2 prep: green box annotation (no models needed) ----------
    annotated = image_np.copy()
    x0, y0, x1, y1 = WORDMARK_BOX
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 8)
    Image.fromarray(annotated).save(OUT_DIR / "ref_annotated_greenbox.png")
    print(f"[#2] annotated ref saved ({x0},{y0})-({x1},{y1})")

    # ---------- test #3 prep: Grounded-SAM isolate jacket onto white ----------
    print("[#3] Grounding DINO: caption 'jacket'...")
    gdino = load_model(f"{GDINO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       f"{GDINO_ROOT}/weights/groundingdino_swint_ogc.pth")
    _, image_transformed = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(model=gdino, image=image_transformed,
                                     caption="jacket", box_threshold=0.3,
                                     text_threshold=0.25)
    print(f"  detections: {list(zip(phrases, [round(l.item(),3) for l in logits]))}")
    if len(boxes) == 0:
        print("  [FAIL] no jacket detected"); return

    box = boxes[logits.argmax()]
    cx, cy, bw, bh = box.tolist()
    bx0, by0 = (cx - bw/2) * w, (cy - bh/2) * h
    bx1, by1 = (cx + bw/2) * w, (cy + bh/2) * h
    box_xyxy = np.array([bx0, by0, bx1, by1])
    print(f"  best jacket box: {box_xyxy.round(0)}")

    print("[#3] SAM2: masking the jacket from the auto box...")
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                            f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt",
                            device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=True)
    mask = masks[np.argmax(scores)].astype(bool)
    print(f"  mask score {scores.max():.4f}, area {mask.sum()} px")

    white = np.full_like(image_np, 255)
    white[mask] = image_np[mask]
    # crop to the jacket with padding so the ref is mostly garment, not canvas
    ys, xs = np.where(mask)
    pad = 100
    cy0, cy1 = max(ys.min()-pad, 0), min(ys.max()+pad, h)
    cx0, cx1 = max(xs.min()-pad, 0), min(xs.max()+pad, w)
    Image.fromarray(white[cy0:cy1, cx0:cx1]).save(OUT_DIR / "ref_isolated_white.png")
    print(f"[#3] isolated ref saved, crop {cx1-cx0}x{cy1-cy0}")


if __name__ == "__main__":
    main()

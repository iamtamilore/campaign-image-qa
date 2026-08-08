import json, numpy as np
from pathlib import Path

try:
    import matplotlib
    #draw to files without needing a screen (same trick as evaluate.py)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    PLOT_AVAILABLE = True
except ImportError:
    print("[warn] Install: pip install matplotlib pillow")
    PLOT_AVAILABLE = False


#walk all 90 graded answers, pull out the mistakes WITH their case files
def find_errors(y_true: np.ndarray, y_pred: np.ndarray,
                pairs: list[dict]) -> dict:
    "returns two piles: false_positives (said same, was different) and false_negatives (said different, was same), each with image paths + metadata"
    false_positives = []   #predicted same (1), actually different (0)
    false_negatives = []   #predicted different (0), actually same (1)

    #compare every prediction against the truth, one by one
    for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
        #false alarm: said same, was different
        if pred == 1 and gt == 0:
            false_positives.append({
                "pair_index": i,
                #keep the actual image paths, the case file, not just the index
                "img_a":      pairs[i].get("img_a", ""),
                "img_b":      pairs[i].get("img_b", ""),
                "grade":      pairs[i].get("grade", "unknown"),
                "error_type": "false_positive",
            })
        #miss: said different, was same
        elif pred == 0 and gt == 1:
            false_negatives.append({
                "pair_index": i,
                "img_a":      pairs[i].get("img_a", ""),
                "img_b":      pairs[i].get("img_b", ""),
                "grade":      pairs[i].get("grade", "unknown"),
                "error_type": "false_negative",
            })

    return {"false_positives": false_positives, "false_negatives": false_negatives}


#count errors per colour grade: does it fail more on one particular grade?
def grade_error_breakdown(errors: dict) -> dict:
    "tally each pile by grade. note: my manifest has no grade key, so this shows 'unknown', the real angle pattern comes from reading the image paths in the json"
    breakdown = {}
    for err_type, err_list in errors.items():
        grade_counts = {}
        for e in err_list:
            g = e.get("grade", "unknown")
            grade_counts[g] = grade_counts.get(g, 0) + 1
        breakdown[err_type] = grade_counts
    return breakdown


#lay the failed pairs side by side, the figure where a human can SEE the miss
def plot_error_pairs(error_list: list[dict], title: str,
                     n_samples: int = 4,
                     save_path: str = "outputs/plots/error_samples.png"):
    "draw the top-N error pairs, one row each: img_a | img_b with grade label"
    #nothing to draw? do nothing (e.g. my false positives pile is empty)
    if not PLOT_AVAILABLE or not error_list:
        return

    samples = error_list[:n_samples]
    fig, axes = plt.subplots(n_samples, 2, figsize=(6, 3 * n_samples))
    #with one row, matplotlib gives a flat pair instead of a grid, wrap it
    if n_samples == 1:
        axes = [axes]

    for row, err in enumerate(samples):
        for col, key in enumerate(["img_a", "img_b"]):
            ax = axes[row][col]
            try:
                img = Image.open(err[key]).convert("RGB").resize((128, 128))
                ax.imshow(np.array(img))
            except Exception:
                #a missing file prints a note in that cell instead of killing the whole analysis
                ax.text(0.5, 0.5, "image\nnot found", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{'A' if col == 0 else 'B'} | grade: {err.get('grade','?')}",
                         fontsize=8)
            ax.axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[error_analysis] plot saved → {save_path}")


#the whole post-mortem: find, count, draw, save the receipt
def run_error_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                        pairs: list[dict],
                        model_name: str = "siamese") -> dict:
    "full error analysis, call after evaluation. returns the dict the report's Error Analysis subsection quotes"
    errors = find_errors(y_true, y_pred, pairs)
    breakdown = grade_error_breakdown(errors)

    n_fp = len(errors["false_positives"])
    n_fn = len(errors["false_negatives"])

    print(f"\n[error_analysis] === {model_name} ===")
    print(f"  False Positives (predicted same, different subject): {n_fp}")
    print(f"  False Negatives (predicted different, same subject): {n_fn}")
    print(f"  Grade breakdown: {breakdown}")

    #the money figures: fp_siamese.png (empty for me) and fn_siamese.png (the 8 misses)
    plot_error_pairs(errors["false_positives"], f"{model_name}: False Positives",
                     save_path=f"outputs/plots/fp_{model_name}.png")
    plot_error_pairs(errors["false_negatives"], f"{model_name}: False Negatives",
                     save_path=f"outputs/plots/fn_{model_name}.png")

    #the receipt, keep only the first 4 examples of each pile to stay readable
    report = {
        "model":            model_name,
        "n_false_positive": n_fp,
        "n_false_negative": n_fn,
        "grade_breakdown":  breakdown,
        "fp_examples":      errors["false_positives"][:4],
        "fn_examples":      errors["false_negatives"][:4],
    }

    out_path = Path(f"outputs/results/error_analysis_{model_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[error_analysis] report saved → {out_path}")
    return report

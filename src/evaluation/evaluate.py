"""
evaluate.py - exam day: one marker, three students (Evaluation rubric = 20% of marks).

the 90 test flashcards have been sealed since preprocessing.py dealt them - 
nobody trained on them, nobody tuned on them. now three students sit the exam:
  student A: siamese network (my proposed method)
  student B: pixel-difference detector (the tracing-paper kid)
  student C: kNN on embeddings (the ruler)

the fairness rule: the SAME marker grades all three with the SAME rubric.
if each model measured itself differently, the comparison table would be meaningless.

covers all metrics the rubric asks for:
 - precision, recall, F1
 - ROC-AUC (siamese only - baselines output hard yes/no, no confidence to rank,
    which is why the report table shows N/A for them: structurally impossible, not missing)
 - confusion matrix (TP/TN/FP/FN - my 37/45/0/8 comes from here)

every number the report quotes lands in outputs/results/all_metrics.json - 
the single source of truth. open that file, not my memory.
"""

import json, numpy as np
from pathlib import Path

try:
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  roc_auc_score, confusion_matrix,
                                  classification_report, roc_curve)
    import matplotlib
    #sneaky but vital: draw to files without needing a screen - 
    #without this the pipeline crashes on a server or plain terminal
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[warn] Install: pip install scikit-learn matplotlib")
    SKLEARN_AVAILABLE = False


OUT_DIR = Path("outputs")


#the marking machine - grades ONE student's answer sheet
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray | None = None,
                                   name: str = "model") -> dict:
    "precision, recall, F1, ROC-AUC + confusion matrix for one binary classifier. y_prob=None for baselines (no confidence to rank)"
    metrics = {
        "name":      name,
        #zero_division=0: if a student never says 'same', precision would be 0/0 and crash.
        #this guard scores it 0 instead - a wall against a lazy student breaking the marker
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    #ROC-AUC only when the student gave confidence scores (siamese) - baselines cant be ranked
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    #the 2x2 scorecard, unpacked into the four boxes
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["TP"] = int(cm[1, 1])   #caught it: said same, was same
    metrics["TN"] = int(cm[0, 0])   #all clear: said different, was different
    metrics["FP"] = int(cm[0, 1])   #false alarm: said same, was different
    metrics["FN"] = int(cm[1, 0])   #miss: said different, was same

    print(f"\n[eval] === {name} ===")
    print(f"  Precision: {metrics['precision']}  Recall: {metrics['recall']}  "
          f"F1: {metrics['f1']}", end="")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC: {metrics['roc_auc']}", end="")
    print()
    print(f"  CM: TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")

    return metrics


#all rankings on one graph, plus the coin-flipper diagonal
def plot_roc_curves(roc_results: list[dict], save_path: str = "outputs/plots/roc_curves.png"):
    "ROC curves for every model that has y_prob. the closer to the top-left corner, the better the ranking"
    plt.figure(figsize=(8, 6))
    for r in roc_results:
        #baselines have no confidence scores → skip them
        if r.get("y_prob") is None:
            continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        auc = roc_auc_score(r["y_true"], r["y_prob"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={auc:.3f})")

    #the dashed diagonal = a student flipping a coin (AUC 0.5)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Siamese vs Baselines")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] ROC plot saved → {save_path}")


#one student's 2x2 scorecard as a blue heatmap
def plot_confusion_matrix(metrics: dict, save_path: str | None = None):
    "draw the confusion matrix for a single model → outputs/plots/cm_<name>.png"
    cm   = np.array(metrics["confusion_matrix"])
    name = metrics["name"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Different", "Same"]); ax.set_yticklabels(["Different", "Same"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    #write the count inside each of the four boxes
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    if save_path is None:
        save_path = f"outputs/plots/cm_{name.replace(' ', '_')}.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] CM plot saved → {save_path}")


#the receipt: every number the report quotes, in one json
def save_results(all_metrics: list[dict], path: str = "outputs/results/all_metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[eval] results saved → {path}")


#exam day itself: same marker, same rubric, all three students
def run_full_evaluation(siamese_preds: dict, pixel_preds: dict,
                         knn_preds: dict) -> list[dict]:
    """
    master evaluation - call with predictions from all three models.

    each input dict must have:
      labels:      ground truth (0/1)
      predictions: predicted labels (0/1)
      y_prob:      probabilities (None for baselines)
      name:        str

    returns list of metrics dicts (one per model).
    """
    all_metrics = []
    roc_data    = []

    #the fairness loop: the SAME marking machine grades every student
    for preds in [siamese_preds, pixel_preds, knn_preds]:
        m = compute_classification_metrics(
            y_true=preds["labels"],
            y_pred=preds["predictions"],
            y_prob=preds.get("y_prob"),
            name=preds["name"]
        )
        all_metrics.append(m)
        roc_data.append({
            "name":   preds["name"],
            "y_true": preds["labels"],
            "y_prob": preds.get("y_prob"),
        })
        plot_confusion_matrix(m)

    plot_roc_curves(roc_data)
    save_results(all_metrics)
    return all_metrics

"""
main.py - START HERE.
CA2: Automated QA for AI-Generated Fashion Campaign Images
H9MLAI Machine Learning | Student: x24335592

This is the single entry point for the whole project. It forwards to the
master pipeline (scripts/pipeline/run_qa_pipeline.py), which runs the
KDD/CRISP-DM stages end to end.

Quickstart (reproduce the reported results from the included trained models):
    pip install -r requirements.txt
    python main.py --phase eval

Other modes:
    python main.py --phase all                      # retrain from scratch, then evaluate
    python main.py --phase data                     # build the 600 image pairs only
    python main.py --phase train                    # train Siamese + Autoencoder only
    python main.py --phase qa --test-image img.png  # QA one new image (pass/fail)

Full project map (which script produced which number): see README.md
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Hand over to the master pipeline as if it were run directly.
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "pipeline" / "run_qa_pipeline.py"),
        run_name="__main__",
    )

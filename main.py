import runpy
from pathlib import Path

if __name__ == "__main__":
    # Hand over to the master pipeline as if it were run directly.
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "pipeline" / "run_qa_pipeline.py"),
        run_name="__main__",
    )

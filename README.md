> **[Read the full write-up](https://iamtamilore.github.io/p/image-qa/)** with diagrams, plots and results.
> Part of [Taiwo Alabi's work](https://iamtamilore.github.io).

# CA2: Automated QA for AI-Generated Fashion Campaign Images

**H9MLAI Machine Learning | x24335592**

A two-part QA gate for AI-generated campaign images: a Siamese network on a frozen
ResNet50 checks **subject identity**, an autoencoder checks **colour grade**. Both are
compared against pixel-difference and kNN baselines.

# Run it

```bash
pip install -r requirements.txt
python main.py --phase eval        # reproduces every reported metric (~1 min)
```

If your GPU errors out with a cuDNN autotuner failure, force CPU:

```bash
CUDA_VISIBLE_DEVICES="" python main.py --phase eval
```

Other phases: `--phase all` (retrain then evaluate), `data`, `train`,
`qa --test-image img.png`.

# Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Siamese (proposed)** | 1.0000 | 0.8222 | **0.9024** | 1.0000 |
| Pixel-difference baseline | 1.0000 | 0.4222 | 0.5938 | n/a |
| kNN baseline | 0.3913 | 0.2000 | 0.2647 | n/a |

Siamese confusion matrix: TP=37, TN=45, FP=0, FN=8. Seeded (seed=42), so the numbers
reproduce exactly on every run.

# Layout

```
main.py            single entry point -> scripts/pipeline/run_qa_pipeline.py
main.tex           the report (+ references.bib)
src/               library: api/ data/ models/ training/ evaluation/
scripts/
  pipeline/        QA pipeline + parametrisation + advanced experiments (free)
  metrics/         flux-2 vs gpt-image-2 comparison (free)
  generation/      image generation - COSTS KIE.AI CREDITS, do not re-run
data/              subject_A/, subject_B/, pairs/pairs_manifest.json (600 pairs)
outputs/           models/ results/ plots/ (plots/ holds the four report figures)
```

Working notes (rubric checklist, scoring) live outside the project in
`../CA2_working_notes/` and are not part of the submission.

# Which script produced which number

| Claim | Run | Output |
|---|---|---|
| Siamese vs baselines (P/R/F1/AUC, confusion matrices) | `python main.py --phase eval` | `outputs/results/all_metrics.json` |
| Error analysis (0 FP, 8 FN) | same run | `outputs/results/error_analysis_siamese.json` |
| Threshold / kNN distance / AE percentile / embedding dim | `scripts/pipeline/run_parametrisation_experiments.py` | `outputs/results/parametrisation_experiments.json` |
| Monte-Carlo CV, L1 vs Euclidean vs cosine, ablation ladder, McNemar | `scripts/pipeline/run_advanced_experiments.py` | `outputs/results/advanced_experiments.json` |
| flux-2 vs gpt-image-2 per angle | `scripts/metrics/compute_angle_metrics.py` | `reports/comparison_metrics.json` |
| AE threshold (95th percentile MSE) | set during training | `outputs/results/ae_threshold.json` |



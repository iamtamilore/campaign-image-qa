> **[Read the full write-up](https://iamtamilore.github.io/p/image-qa/)** with diagrams, plots and results.
> Part of [Taiwo Alabi's work](https://iamtamilore.github.io).

# Automated QA Gate for AI-Generated Campaign Images

**Catches the moment an AI-generated model stops being the same person.**

---

## Why this matters

When a fashion brand shoots a campaign, every photo has to be the **same model**. That
sounds obvious because with a real photoshoot it is automatic - you hire a person, you
photograph them from different angles, and it is obviously still them.

With AI-generated imagery, it is not automatic at all.

Ask an image generator for "the same model, from the side" and it does not remember the
person from the last picture. It reads your reference photo and **reinvents someone
similar, from scratch, every single time**. The jaw shifts. The nose changes. The face
gets slightly younger.

You end up with a set of photos of *similar-looking but different people*, and you are
supposed to sell it as one campaign.

The same thing happens to the lighting and colour. Ask for a new angle and the generator
re-decides the studio, the lamps, the film look. Now your photos look like they were taken
on four different days by four different photographers.

### A real problem it solves

I was generating campaign images for a fashion brand. For every shot, I had to open it,
open the original reference photo, and **stare at both to decide whether it was still the
same woman**.

Fifty images per model. Two models. Doing it by eye, one at a time.

Worse, human eyes are bad at this. After twenty images you stop noticing small drift. You
approve something, and the mistake only becomes obvious later when all the photos sit next
to each other in a lookbook.

So I built something that checks every image automatically and gives a straight pass or
fail. No staring.

---

## Two different checks, because there are two different failure modes

**Check 1 - is this still the right person?**

The system compares the new image to the reference photo and answers: same person, or not.

**Check 2 - does this still look like our campaign?**

An image can pass check 1 and still be wrong. Right woman, but washed-out colour and the
wrong lighting mood. So a second, separate check looks at the *style*.

An image only passes if **both** checks clear it.

### The clever bit about check 2

The style checker is never shown a single bad image.

It only ever studies good, on-brand photos, and it learns to redraw them from memory. Then
when a new image arrives, it tries to redraw that one too. A normal, on-brand image gets
redrawn almost perfectly. An off-style image comes out badly wrong - because the system has
never seen anything like it.

**The size of that redraw error is the alarm.** It detects bad images without ever having
been shown one, which matters because nobody has a labelled library of "campaign photos
that went wrong".

---

## Does it work

Tested on 90 image pairs the system had never seen:

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **This system** | **1.00** | 0.82 | **0.90** | **1.00** |
| Simple pixel comparison | 1.00 | 0.42 | 0.59 | - |
| Nearest-neighbour lookup | 0.39 | 0.20 | 0.26 | - |

**Precision 1.00 means zero false alarms.** It never once rejected a good image. For a
working pipeline that matters enormously - a checker that cries wolf gets switched off
within a week.

It missed 8 genuine matches out of 45. Every single one was the same subject at angles
where the face is barely visible - shot from behind, or in sharp profile. That is a
sensible thing to be uncertain about, and it fails in the safe direction: it asks a human
rather than waving something through.

**Why compare against those two baselines?** Because they are what you would try first.
Pixel comparison collapses the moment someone turns their head - every pixel moves.
Nearest-neighbour uses the same underlying image fingerprints as the main system but does
not *learn* which parts of a face actually identify a person, so background and lighting
noise count just as much as bone structure.

The gap between 0.90 and 0.26 is what the learned comparison is worth.

---

## The trap I had to design around

Here is a mistake that would have made the results meaningless.

If model A only appeared in images from generator 1, and model B only in images from
generator 2, the system could get a perfect score **without ever looking at a face** - just
by spotting which generator made the picture.

So every model appears in images from **both** generators, in matched styles and matched
angles. Everything is held identical except identity. The only thing left to learn is the
face.

That is not a detail. Shortcut learning is the most common reason a machine learning result
looks excellent and means nothing.

---

## Everything is reproducible

One command reproduces every number above:

```bash
pip install -r requirements.txt
python main.py --phase eval
```

Every random choice is seeded (`seed=42`). The dataset - which images pair with which - is
saved to a manifest file, so it is auditable rather than regenerated differently each run.

To check one new image:

```bash
python main.py --phase qa --test-image your_image.png
```

It returns a pass/fail verdict as JSON.

---

## How it was validated

Beyond the headline table:

- **Monte-Carlo cross-validation** - re-run over many random splits, so the result is not a
  lucky split
- **McNemar's test** - a statistical check that the improvement over the baselines is real
  and not chance
- **Distance-measure comparison** - three different ways of measuring "how far apart are
  these two faces", compared rather than assumed
- **Hard vs random negatives** - does it still work when the wrong answers are deliberately
  difficult
- **Embedding-size sweep** - 64, 128 and 256, to check the choice was justified

---

## Honest limitations

The dataset is **two subjects and 106 images**. That is small. The numbers are real and
reproducible, but two people is not proof it generalises to fifty.

The 8 misses are all one subject at low-face-visibility angles, which suggests the system
leans on facial features more than body shape - fine for portraits, less so for a
back-turned shot.

The style checker was tested against **synthetically** degraded images (hue shifts,
over-exposure) rather than real-world failures, because real off-grade campaign images were
not available in quantity.

---

## Under the hood

| Piece | What it does |
|---|---|
| **ResNet50, frozen** | turns each photo into 2,048 numbers describing what it sees. Pre-trained on a million photos, never retrained here - 600 pairs cannot improve on a million images |
| **Siamese network** | compares two of those fingerprints and learns *which* differences actually indicate a different person |
| **Autoencoder** | learns to redraw on-brand images; its failure to redraw an off-brand one is the style alarm |
| **TensorFlow / Keras** | the training framework |

Transfer learning showed its value plainly: the identity model, standing on a pre-trained
backbone, converged in **11 epochs**. The style model, learning from scratch, needed **67**.

---

Built for **H9MLAI - Machine Learning**, MSc Artificial Intelligence, National College of
Ireland.

**Note:** trained model files and the generated image dataset are excluded from this
repository for size. `python main.py --phase all` retrains everything from scratch.
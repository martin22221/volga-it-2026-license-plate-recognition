# Real-image splits

`real_splits.csv` is the frozen record of which real image belongs to
`train`, `val` or `holdout`. It holds the **13 real photographs** promoted on
2026-09-22: train 10, val 1, holdout 2, over 11 groups.

`synthetic_splits.csv` is the companion for the 12,000 synthetic images
(train 10,200 / val 1,200 / test 600), and `dataset_v1.json` is the Dataset V1
freeze that pins both. The two manifests are split by different rules on
purpose — see "Two populations" below — and
[`../../docs/baseline_v1.md`](../../docs/baseline_v1.md) says how each is used.

| Column | Meaning |
| --- | --- |
| `image` | Path relative to `dataset/`, exactly as in `meta.csv`. |
| `group` | The leakage unit (see below). Everything sharing a group shares a split. |
| `split` | `train`, `val` or `holdout`. |
| `source` | `source_id` the image came from, copied from `meta.csv`. |
| `assigned_on` | ISO date the row was first written. |
| `note` | Free text: why a row was forced, if it ever is. |

## Groups

A group is **whatever would leak if it were split**. Assign the same group to:

- every frame of one burst, video or continuous capture session;
- every photograph of the same vehicle, even minutes apart or from another angle;
- every photograph of the same physical plate;
- crops or re-edits derived from one original photograph.

When in doubt, group more coarsely. A group that is too large costs a little
split balance; a group that is too small leaks a plate the model has already
seen into the holdout, and every number measured on it becomes optimistic.

Groups are recorded per source in `groups.csv` during intake
(`image;group`), and frozen into this manifest by
`scripts/plan_real_splits.py`.

## How the assignment works

`src/real_splits.py` hashes `"<seed>:<group>"` with SHA-256 and maps the
result into the weight ranges (default train 0.70 / val 0.15 / holdout 0.15).
So the assignment is:

- **deterministic** — the same groups, weights and seed give the same answer on
  any machine, whatever order the images arrive in;
- **group-aware** — the split is a property of the group, never of the file name;
- **frozen** — a group already in this file keeps its split. New images are
  assigned around the existing ones, and `plan_real_splits.py` refuses to move
  an image to a different group.

```bash
python scripts/plan_real_splits.py --check                       # audit for leakage
python scripts/plan_real_splits.py --add <accepted dir> --write  # freeze new rows
```

## The holdout rule

**The holdout is never trained on, never tuned on, and never used to pick a
checkpoint.** It exists to approximate the jury's hidden test set with real
photographs the model has never seen. Look at it when a decision is final, and
report what it says whether it flatters us or not. Anything else silently turns
it into a validation set.


## Two populations, two rules

**Real** images are split by **group**, and the assignment is **frozen**: once a
group is written here it keeps its split forever. A group is whatever would leak
if it were divided — one vehicle, one plate, one capture session, one
photographer on one day. This is the rule that makes the holdout mean something,
and it is not relaxed when the numbers would look nicer with a different draw.

**Synthetic** images are split per image, stratified by
`(plate_type, difficulty)`, seeded by `SPLIT_SEED` in `src/dataset_v1.py`. Per
image is safe here for a reason worth stating: the 12,000 synthetic images carry
**12,000 distinct plate numbers**, so no plate identity can appear on both sides
of a split however the images are divided.

```bash
python scripts/freeze_dataset_v1.py           # re-derive and freeze
python scripts/freeze_dataset_v1.py --check   # prove it still reproduces
```

`--check` recomputes the split from the seed, re-hashes every image and
`meta.csv`, and compares against the frozen digests. That is what "Dataset V1"
means: not a folder, a digest.

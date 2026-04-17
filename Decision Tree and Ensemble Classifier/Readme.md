## Mushroom Classification — Decision Tree & Ensemble

Predict whether a mushroom is **edible** or **poisonous** from discrete attributes using an ID3 decision tree and an ensemble classifier. Built from scratch with zero external libraries.

## Features

| Column | Attribute   | Possible Values |
|--------|-------------|-----------------|
| 1      | class       | edible (e), poisonous (p) |
| 2      | cap_shape   | bell, conical, convex, flat, knobbed, sunken |
| 3      | cap_surface | fibrous, grooves, scaly, smooth |
| 4      | cap_color   | brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow, tan, fawn |
| 5      | bruises     | yes, no |
| 6      | odor        | almond, anise, creosote, fishy, foul, musty, none, pungent, spicy |

## Data

Training and test data are in `resources/`:

- `mushroom_train.csv` — 100 labeled samples for training
- `mushroom_test.csv` — 50 labeled samples for testing
- `ensemble_fold_1.csv`, `ensemble_fold_2.csv` — random splits for cross-validation

## Usage

```bash
python3 main.py
```

```bash
python3 -m tests.run_all
```

## Project Structure

```
src/
  engine/
    math_core.py       — log2 (pure Python, double precision)
    information.py     — entropy, information gain, feature selection
  utils/
    data.py            — CSV loading, counting, partitioning
    tree_ops.py        — tree node, building, prediction, pruning, display
  metrics/
    evaluation.py      — accuracy, confusion matrix, precision/recall/F1
  classifier.py        — DecisionTree (public API)
  ensemble.py          — EnsembleClassifier (majority-vote, k-fold CV)

tests/                 — 37 unit tests across 7 test files
resources/             — CSV data files
main.py                — entry point, runs both classifiers
```

## Algorithms

**Decision Tree (ID3):** Splits on the feature with highest information gain (lowest weighted entropy) at each node. Supports configurable max depth, minimum samples per split, and reduced-error post-pruning.

**Ensemble Classifier:** Trains one decision tree per fold using k-fold cross-validation. Classifies new samples via majority vote across all trees.

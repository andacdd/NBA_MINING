# NBA Shot Success Prediction

Predicting whether an NBA field-goal attempt goes in, from the game context and the physical
matchup between shooter and defender.

Built on the 2014–15 NBA shot logs, enriched with player anthropometrics (height, weight,
wingspan). The interesting part is not the raw accuracy — shot outcomes are genuinely noisy — but
what the engineered features reveal about *which* factors actually move the needle.

---

## Dataset

`shot_logs_with_physics_safe.csv` — 127,974 shot attempts, 27 columns.

| Field group | Columns |
|---|---|
| Game context | `game_id`, `matchup`, `location`, `w`, `final_margin`, `period`, `game_clock`, `shot_clock`, `shot_number` |
| Shot mechanics | `dribbles`, `touch_time`, `shot_dist`, `pts_type` |
| Defense | `closest_defender`, `closest_defender_player_id`, `close_def_dist` |
| Physicals | `shooter_height/weight/wingspan`, `defender_height/weight/wingspan` |
| Outcome | `shot_result`, `fgm`, `pts` |

**Class balance:** ~45.7% made / 54.3% missed — mildly imbalanced, handled with
`class_weight='balanced'` (Random Forest) and `scale_pos_weight` (XGBoost).

### Missing data strategy

| Column | Missing | Handling |
|---|---|---|
| `shot_clock` | 4.3% (5,559) | Dropped — the rows are too few to justify imputing a pressure signal |
| `shooter_wingspan` | 30.0% | Ratio imputation (see below) |
| `defender_wingspan` | 32.5% | Ratio imputation |
| `shooter_height` | 0.15% | Mean fill |
| `defender_height` | 0.43% | Mean fill |

**Wingspan imputation.** A third of wingspan values are missing, which is too many to drop and too
many to mean-fill without flattening real variation. Instead the average wingspan-to-height ratio
is computed from the players who *do* have both, then missing wingspans are estimated as
`height × ratio`. NBA wingspan scales tightly with height, so this preserves the tall-player /
short-player spread that a flat mean would erase.

## Feature Engineering

This is the core of the project. Heights are parsed from `6-1` strings to centimetres and
distances converted feet → metres, then:

| Feature | Idea |
|---|---|
| `shot_clock_isSlope` | Non-linear clock pressure: 0 above 5 seconds, then rising quadratically as the clock expires — models the "panic" curve instead of treating 24s and 2s as linearly different |
| `ratio_wingspan_def_of` | Shooter wingspan ÷ defender wingspan. Above 1 means the shooter has the physical advantage over the contest |
| `shot_difficulty` | `shot_dist − 1.5 × close_def_dist`. Composite index; the defender term is weighted 1.5× because a hand in the face costs more than a few extra feet |
| `shot_angle` | `arctan2(loc_x, loc_y)` in degrees — corner threes and straight-on shots are geometrically different problems |
| `pressure_level` | `close_def_dist` bucketed into Very Tight / Tight / Open / Wide Open |
| `is_catch_and_shoot` | `dribbles == 0` — rhythm shots off the pass |
| `is_high_dribble` | `dribbles > 3` — self-created, usually harder shots |
| `is_home` | Home-court advantage flag |
| `is_critical` | `abs(final_margin) ≤ 10` — close-game proxy |

### Encoding

- **`action_type` → target encoding.** Shot types (Jump Shot, Dunk, Fadeaway, Driving Layup, …)
  are high-cardinality; one-hot would explode the feature space. Each category is replaced with
  its historical success rate. **Crucially, the rates are computed on the training split only**
  and then mapped onto test — computing them on the full dataset would leak the answer. Unseen
  test categories fall back to the global mean.
- **`shot_zone_basic` / `shot_zone_area` → label encoded**, with nulls kept as an explicit
  `"Unknown"` category rather than dropped.

### Leakage control

`fgm`, `pts`, `api_points` and `shot_result` are dropped before training — they encode the answer
directly. IDs (`game_id`, `player_id`, defender id) are dropped because they carry no
generalisable pattern. Raw columns superseded by engineered versions (`shot_clock`,
`shooter_height`, `shot_type`, …) are dropped to avoid duplicating signal.

Split: 80/20, `stratify=y`, `random_state=42`.

## Models

Three classifiers, each chosen to test a different hypothesis about the data:

| Model | Setup |
|---|---|
| **Random Forest** | `RandomizedSearchCV`, 20 candidates × 3-fold, `class_weight='balanced'` |
| **XGBoost** | `RandomizedSearchCV`, 20 candidates × 3-fold, scored on F1, `scale_pos_weight` tuned |
| **KNN** | k=21, `weights='distance'`, on `StandardScaler`-ed features; Elbow method sweep over k = 1…29 |

KNN is included as a deliberate contrast: it needs scaling (Euclidean distance would otherwise be
dominated by `shot_dist`'s 0–90 range over `shot_clock`'s 0–24) and it makes the
lazy-learning cost concrete — training is instant, prediction is expensive.

### Threshold optimisation

Rather than accepting the default 0.50 cutoff, both tree models sweep thresholds from 0.30 to 0.70
and report accuracy, recall on made shots, and F1 at each. In shot prediction the default
threshold under-predicts makes; the sweep exposes the precision/recall trade-off explicitly
instead of hiding it. Evaluation output includes confusion matrices and ROC/AUC curves with the
chosen threshold marked.

## Results

An earlier, smaller-feature version of the pipeline is preserved in
`.ipynb_checkpoints/main-checkpoint.ipynb` with saved outputs:

| Stage | Test accuracy |
|---|---|
| Random Forest, default params | 0.6034 |
| Random Forest, tuned (`n_estimators=100, max_depth=10, min_samples_leaf=4, max_features='sqrt'`) | **0.6206** |

Classification report at the untuned baseline showed the expected asymmetry — recall 0.77 on
missed shots vs 0.40 on made shots — which is exactly what motivated the threshold-tuning step in
the current script.

**On interpreting ~62%:** against a 54.3% majority-class baseline this is a real but modest gain,
and that is the honest ceiling of the problem. Whether an NBA jump shot drops depends on release
mechanics and luck that no row of tabular game-context data captures. The value here is in the
feature analysis — which measurable conditions shift the probability — not in the headline number.

> The current `main.py` (with `shot_angle`, `shot_difficulty`, `action_type` target encoding and
> XGBoost/KNN added) has no committed run output, so its metrics are not quoted above rather than
> estimated.

## Running It

> **Note:** `main.py` reads `nba_final_clean.csv`, which is **not** in this repository. It expects
> a version of the shot logs already merged with NBA API shot-chart fields — `loc_x`, `loc_y`,
> `action_type`, `shot_zone_basic`, `shot_zone_area`, `shot_type`, `api_points`, `api_calc_dist` —
> none of which exist in the committed `shot_logs_with_physics_safe.csv`. The script will not run
> against the shipped CSV as-is. The merge/enrichment step needs to be added to this repo (or the
> `nba_final_clean.csv` file committed) before the pipeline is reproducible end to end.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

```bash
python main.py
```

Expect a long runtime: two `RandomizedSearchCV` passes (60 fits each) plus a KNN elbow sweep that
refits 15 models on ~98k rows.

## Repository Contents

| File | Description |
|---|---|
| `main.py` | Full pipeline: cleaning, feature engineering, encoding, three tuned models, evaluation |
| `shot_logs_with_physics_safe.csv` | Shot logs joined with player physical attributes (127,974 rows) |
| `.ipynb_checkpoints/main-checkpoint.ipynb` | Earlier notebook version, with saved Random Forest results |

## Notes & Limitations

- **`is_critical` uses `final_margin`**, the game's *final* score gap — information not available
  at the moment the shot is taken. It is a useful descriptive feature but a look-ahead one; a
  strictly causal version would need the live score differential instead.
- **The threshold is selected on the test set.** Best-F1 is chosen by sweeping against `y_test`,
  which makes the reported figure at that threshold mildly optimistic. A validation split for
  threshold selection would separate the two cleanly.
- The KNN elbow sweep refits on the full training set; the script contains a commented-out
  subsample option for faster iteration.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · seaborn

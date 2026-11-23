# Spaceship Titanic: Kaggle Entry

My entry for the Kaggle competition **Spaceship Titanic**. The task is to predict which passengers were transported to another dimension (`Transported` = True/False) from their records.

| | Score |
|---|---|
| **My leaderboard score** | **0.80056** |
| Average of the top 49 scores | 0.83340 |
| Difference | 0.03284 |

---

## Data

- **Train:** 8,693 passengers, 14 columns
- **Test:** 4,277 passengers, 13 columns
- Target is nearly balanced, so accuracy is a fair metric.
- About 2 to 3% of values are missing in almost every column.

---

## What I Did

### 1. Exploration
I compared the transport rate across the main columns:

| Feature | Finding |
|---|---|
| **CryoSleep** | Strongest signal: 82% transported if asleep vs 33% if not |
| **HomePlanet** | Europa 66%, Mars 52%, Earth 42% |
| **Spending** | Transported passengers spent far less on RoomService, Spa and VRDeck |
| **Deck / Side** | Decks B and C are high (73% and 68%), deck T is very low (20%). Side S beats side P (56% vs 45%) |
| **VIP** | VIPs were transported less often (38% vs 51%) |

### 2. Feature engineering
- `TotalSpend`: sum of the five spending columns
- `Deck`, `CabinNumber`, `Side`: split from the `Cabin` string (`deck/number/side`)

### 3. Preprocessing
- Numeric columns: median imputation
- Categorical columns: most-frequent imputation, then one-hot encoding
- Everything is in a single scikit-learn `Pipeline`, so the same steps apply to train and test.

### 4. Models (80/20 stratified validation split)

| Model | Validation accuracy |
|---|---|
| Random Forest (300 trees) | 0.8108 |
| **HistGradientBoosting** (300 iterations, learning rate 0.05) | **0.8171** |
| HistGradientBoosting + `GroupId` / `GroupSize` | 0.8137 (worse, so I dropped it) |

### 5. Final model
HistGradientBoosting on the 14 features above, **retrained on all the training data**, then used to predict the test set and write `submission.csv`.

---

## Limitations

- **Validation vs leaderboard gap.** The model scored 0.817 on my validation split but 0.80056 on the leaderboard. A single split is a noisy estimate, so I probably over-trusted it.
- **No cross-validation and no hyperparameter tuning.** I used reasonable defaults and compared only two models.
- **Simple imputation.** Missing values were filled with the median or mode. Better options exist, for example setting spending to 0 when `CryoSleep` is True (sleeping passengers cannot spend).
- **Unused columns.** I did not use `Name`, so I missed surname-based family groups. I also did not use group information properly. `GroupSize` did not help in my test, but features like how many group members share a cabin or home planet were never tried.
- **Single model.** I did not ensemble (for example with XGBoost, LightGBM or CatBoost).
- I am about 0.033 below the average of the top 49 scores, so this is a solid baseline but not a top solution.

---

## Ideas for Next Time
1. Use 5-fold cross-validation to compare models more reliably.
2. Fix the spending logic: fill spending with 0 for CryoSleep passengers, and infer CryoSleep from zero spending.
3. Fill missing values using the passenger's group (same `GroupId` often shares HomePlanet and Cabin).
4. Add surname and family-size features from `Name`.
5. Tune hyperparameters and try an ensemble of gradient boosting models.

---

## Reproducing
1. Download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition page.
2. Run the notebook top to bottom. It writes `submission.csv`.
3. Needs Python 3, `pandas`, `scikit-learn` and `matplotlib`.

*Built as a learning project in machine learning.*

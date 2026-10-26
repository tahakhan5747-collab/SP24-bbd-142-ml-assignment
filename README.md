# Simple Kaggle Machine Learning App

Two easy tasks with real recorded datasets available on Kaggle:

| Task | Input | Output | Model |
| --- | --- | --- | --- |
| Restaurant tip prediction | Total bill in US dollars | Tip in US dollars | Simple Linear Regression |
| Iris flower identification | Petal length and width in cm | Flower species | Decision Tree (maximum depth 2) |

Both models are trained directly from the included CSVs, not generated data or scikit-learn's bundled Iris loader.

## Start here: Windows and VS Code

Use Python 3.11 or 3.12. Extract the ZIP, open the `fullstack-ml` folder using VS Code's File > Open Folder, then Terminal > New Terminal. Select Command Prompt as the terminal type.

Run one command at a time:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python train_models.py
python -m unittest -v
python -m uvicorn main:app --reload
```

On macOS/Linux replace the activation command with `source .venv/bin/activate`.

Open http://127.0.0.1:8000 in your browser. Leave the terminal running. Press Ctrl+C to stop. Do not double-click index.html. No API keys, paid services or dataset downloads are needed to run this package; installation requires internet.

## Example inputs

- Bill: **20** dollars. Expected predicted tip: **3.06** dollars.
- Petal length: **1.4** cm; petal width: **0.2** cm. Expected: **Iris-setosa**.
- Petal length: **4.5** cm; petal width: **1.3** cm. Expected: **Iris-versicolor**.
- Petal length: **5.8** cm; petal width: **2.2** cm. Expected: **Iris-virginica**.

## Measured evaluation results

| Model | Training/test records | Test metric |
| --- | --- | --- |
| Linear Regression | 195 / 49 | R² = **0.5449381659** |
| Decision Tree | 120 / 30 | Accuracy = **0.9333333333** (28/30 = 93.33%) |

Both use an 80/20 split and random_state=42. Iris uses stratification to retain class proportions. Scores are measured on records excluded from fitting. They are not training scores.

R² is not classification accuracy. The tip model explains about 54.49% of the variation in held-out tips; customers with similar bills can leave different tips. A modest score is reasonable for a deliberately simple one-input model.

## Dataset sources and preparation

### 1. Waiter's Tips

- Kaggle: https://www.kaggle.com/datasets/aminizahra/tips-dataset
- Downloaded archive: https://www.kaggle.com/api/v1/datasets/download/aminizahra/tips-dataset
- 244 recorded restaurant observations. Original core dataset attribution: Bryant and Smith (1995), Practical Data Analysis: Case Studies in Business Statistics.
- Background on the recorded observations: https://iowabiostat.github.io/data-sets/tips/tips.html
- The Kaggle CSV includes additional columns. This package retains **only `total_bill` and `tip`**, preserving every row's values and order. No observations are generated. The included `data/tips.csv` is this two-column subset, not the full original Kaggle file.
- Both selected columns are numeric and have no missing values. No encoding or scaling is needed.
- Bill input is restricted to $3–$51, close to the observed range. This is a historical dataset from one restaurant, not a universal tipping rule. The public dataset card should be consulted for its current redistribution terms; no new ownership or license is claimed here.

### 2. Iris Species

- Kaggle: https://www.kaggle.com/datasets/uciml/iris
- Downloaded archive: https://www.kaggle.com/api/v1/datasets/download/uciml/iris
- Publisher: UCI Machine Learning on Kaggle; source is Fisher's classic Iris measurements. Kaggle lists CC0/Public Domain.
- Included `data/Iris.csv` is the original Kaggle CSV. 150 flowers, three species, 50 records per species.
- We select only **PetalLengthCm** and **PetalWidthCm** as inputs and **Species** as the target. ID and sepal measurements are unused. No rows are removed or generated.

## API

POST `/predict/regression`:

```json
{"total_bill":20}
```

Returns `{"prediction":3.06,"unit":"USD"}`.

POST `/predict/classification`:

```json
{"petal_length":1.4,"petal_width":0.2}
```

Returns `{"prediction":"Iris-setosa"}`. Missing, invalid or unexpected fields return HTTP 422.

GET `/` serves the frontend. GET `/metrics` returns measured evaluation results. http://127.0.0.1:8000/docs provides interactive endpoint documentation. Frontend and backend share one origin.

## Files to understand

| File | Purpose |
| --- | --- |
| train_models.py | Read CSVs, split data, train, evaluate and save both models |
| main.py | Load saved models, validate requests and return predictions |
| static/index.html | Forms, minimal styling and JavaScript requests |
| data/tips.csv | Two-column subset of Kaggle tips data |
| data/Iris.csv | Original Kaggle Iris data |
| models/*.joblib | Saved fitted models |
| models/metrics.json | Scores, split sizes and fitted regression coefficients |
| models/tree_rules.txt | Readable decision rules learned from training data |
| EXPLANATION.md | Beginner walkthrough and viva questions |
| requirements.txt | Tested package versions |
| test_app.py | API prediction and validation tests |

Run train_models.py before starting the backend, especially if you change data or package versions. Only load trusted joblib files.

## Submit

Upload the project contents to your GitHub repository, including data, models and README. Preserve the folder structure. Do not upload `.venv` or `__pycache__`. The `.gitignore` excludes them from normal Git tracking. Submit the repository link; this package has not created a repository for you.

The assignment permits ChatGPT help but requires you to write and explain your own code. Use this AI-assisted implementation as a study reference, develop your own understanding and describe assistance honestly.

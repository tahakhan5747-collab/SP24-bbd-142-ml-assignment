# Understand the entire assignment

## Your 30-second project explanation

“I built one webpage with two machine learning forms. The first uses Linear Regression to predict a restaurant tip from the total bill. The second uses a small Decision Tree to classify an Iris flower from petal length and width. I downloaded both datasets from Kaggle, divided them into training and testing sets, evaluated the models, and saved them with joblib. FastAPI loads the saved models and returns predictions to the webpage as JSON.”

## 1. Two very simple models

### Regression: bill amount predicts tip

Each row has a bill amount and an observed tip. X is the bill; y is the tip. Linear Regression learns a straight line:

**Predicted tip = intercept + slope × total bill**

Our fitted equation is approximately:

**Predicted tip = 0.9252 + 0.1070 × bill**

For a $20 bill: 0.9252356 + 0.1069637 × 20 = 3.0645, displayed as **$3.06**. The slope means an extra dollar of bill is associated with about $0.107 more predicted tip in this model. This describes an association, not proof of causation. The intercept is a mathematical part of the line, not a recommendation for a zero-dollar bill.

### Classification: petals predict flower species

The Decision Tree learns numerical questions from flower examples. With this training split, its rules are:

1. If petal length is at most 2.45 cm, predict **Iris-setosa**.
2. Otherwise, if petal width is at most 1.65 cm, predict **Iris-versicolor**.
3. Otherwise, predict **Iris-virginica**.

We did not manually code these thresholds into the prediction function; the algorithm learned them during `fit`. `max_depth=2` limits the tree to two levels of questions. It makes the model easy to explain and restricts complexity.

## 2. What train_models.py does

Read this list alongside the code, from top to bottom:

| Code | Meaning |
| --- | --- |
| import csv | Read comma-separated data files |
| import json | Write evaluation results as JSON |
| Path | Build paths relative to this script |
| import joblib | Save and load fitted model objects |
| sklearn imports | Bring in algorithms, metrics, splitting and tree-rule display |
| BASE = Path(__file__).resolve().parent | Find the folder containing this Python file |
| def main() | Group the training instructions in one function |
| with ...open() as file | Open a file and close it automatically afterward |
| list(csv.DictReader(file)) | Read CSV rows as dictionaries keyed by column names |
| X = [[float(...)]] | Build numeric feature rows; each row is a list |
| y = [float(...)] | Build the numeric target list |
| train_test_split | Reserve some records for testing and use the rest for training |
| test_size=0.2 | Reserve 20% for testing |
| random_state=42 | Make the random split reproducible |
| LinearRegression() | Create an untrained straight-line model |
| fit(X_train, y_train) | Learn the line from training examples |
| predict(X_test) | Estimate tips for unseen test bills |
| r2_score(y_test, predictions) | Evaluate numeric predictions against actual test tips |
| mkdir(exist_ok=True) | Create the models folder if needed |
| joblib.dump | Save a trained model to a file |
| len(...) | Count the records |
| Iris feature list | Select petal length and width; ignore ID and sepal columns |
| Species target list | Keep species names as target labels |
| stratify=y | Preserve species proportions in train and test splits |
| DecisionTreeClassifier(max_depth=2) | Create a small classification tree |
| accuracy_score | Calculate the fraction of correct test classifications |
| metrics dictionary | Collect scores, sample counts and line coefficients |
| float(...) | Convert model numbers to ordinary Python numbers for JSON |
| write_text(json.dumps(...)) | Save the dictionary in readable JSON format |
| export_text | Convert the fitted decision tree into readable rules |
| if __name__ == '__main__' | Run training when this script is executed directly |

The Iris target names are strings. DecisionTreeClassifier can handle string class labels, so we do not need label encoding. We also do not need a scaler for this small tree or a one-feature linear regression.

## 3. What main.py does

This file handles requests, not training.

- Imports provide paths, JSON reading, joblib loading, FastAPI, HTML file responses and input validation.
- `joblib.load` reads each fitted model once when the backend module is loaded. `metrics` loads the saved test results. Train first so these files exist.
- `app = FastAPI(...)` creates the web application.
- `BillInput` describes the request body: one numeric `total_bill`. `FlowerInput` describes two numeric petal measurements. They inherit from Pydantic's `BaseModel` so FastAPI can validate JSON.
- `Field(ge=3, le=51)` means the number must be at least 3 and at most 51. The petal bounds similarly limit input to reasonable dataset ranges. These checks do not guarantee a combination occurs in the dataset.
- `ConfigDict(extra='forbid', allow_inf_nan=False)` rejects unexpected fields and infinite/not-a-number values.
- `@app.get('/')` connects the homepage URL to `home`. `FileResponse` returns the HTML file to the browser.
- `@app.get('/metrics')` makes the saved results available to the page.
- `@app.post(...)` connects each required prediction endpoint to its function. The type annotation tells FastAPI which input schema to use.
- `[[data.total_bill]]` is one row with one feature. The flower input is one row with two features, in training order. sklearn expects a two-dimensional input.
- `.predict(...)[0]` runs the model and selects the only result. `float`, `str` and `round` create JSON-compatible, readable outputs.
- Returning a dictionary causes FastAPI to send JSON to the client.

## 4. What index.html does

HTML creates the text, two forms, labels, input fields and result areas. CSS sets colors, spacing and a mobile layout. Neither HTML nor CSS trains the models.

JavaScript connects the forms to Python:

1. Loop over `regression` and `classification` and find the matching form.
2. Listen for a submit event. `preventDefault()` prevents a page reload.
3. `FormData(form)` collects the inputs. `Number(value)` changes input text into numbers.
4. Disable the button while waiting and show “Predicting”.
5. `fetch` sends a POST request to the relevant route. The header says the body is JSON; `JSON.stringify` converts the object to JSON text.
6. `await` waits for the server response without blocking the browser interface.
7. If the status is unsuccessful, throw an error. Otherwise parse the response using `response.json()`.
8. Use `textContent` to display the predicted tip or species safely as plain text.
9. `catch` displays failures; `finally` restores the button.
10. `loadScores()` separately gets the metrics and displays R² and percentage accuracy.

The form's field names match the Python request schema. Both frontend and backend run at the same address, so no separate CORS setup is needed.

## 5. Metrics and viva questions

**Why train and test separately?** Training teaches the model. Testing estimates its performance on records it did not learn from.

**What is R²?** R² = 1 − SSE/SST, where SSE sums squared prediction errors and SST sums squared differences from the test target mean. Our result is 0.5449, meaning the model explains about 54.49% of the variation in held-out tips. R² is not accuracy; it can be negative.

**Why isn't R² almost 1?** Real customers tip differently even for similar bills. Our beginner model deliberately uses only the bill amount and a straight line.

**What is accuracy?** Correct predictions / total test records. Here, 28/30 = 93.33%. This is a small test set, not a promise about every future flower.

**What is regression?** Predicting a continuous number, such as a tip in dollars.

**What is classification?** Predicting a category, such as a flower species.

**Why choose these models?** The straight line has one slope and one intercept. The shallow tree has three easy-to-read rules. Both are easy to demonstrate and explain.

**What are features and a target?** Features are the inputs. The target is the answer the model learns to predict.

**Why save models?** So the backend can reuse learned parameters without training on every request.

**What is an endpoint?** A server URL that handles a particular action. This assignment has separate POST URLs for numeric and category predictions.

**What is JSON?** A text format that lets JavaScript and Python exchange named values.

**Are both datasets from Kaggle?** Yes. We downloaded both CSVs through Kaggle's public dataset download endpoints. Tips was reduced to its two relevant columns without changing observations; Iris is included unchanged. Source links are in README.md.

**What are the limitations?** Tips data come from one historical restaurant, and bill size is only one factor. Iris includes three species only. The tree can misclassify flowers. This is an educational app without accounts or stored user submissions.

**What did AI help with?** Explain the assistance honestly, then demonstrate that you understand your implementation and any modifications you made.

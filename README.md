# Golf Tournament Prediction

This project aims to predict whether a golfer will **make the cut** in a golf tournament based on their **strokes gained** metrics. We compare two machine learning models, **Logistic Regression** and **XGBoost**, to determine the best-performing model for predicting the outcome of golf tournaments.

## Table of Contents
- [Project Description](#project-description)
- [Dependencies](#dependencies)
- [Data](#data)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Visualization](#visualization)
- [License](#license)

## Project Description

This project uses historical golf tournament data to build predictive models that can determine whether a golfer will **make the cut** (i.e., advance to the next round of a tournament) based on **strokes gained** metrics. The dataset includes various features such as **strokes gained putting (sg_putt)**, **strokes gained approach (sg_app)**, and **strokes gained off the tee (sg_ott)**. The models used for this project are:
- **Logistic Regression**: A simple yet powerful classification algorithm to model the probability of making the cut.
- **XGBoost (eXtreme Gradient Boosting)**: A more advanced gradient boosting method known for its speed and performance.

The models are evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **AUC** to determine which model performs better.

## Dependencies

To run this project, you'll need the following Python libraries:

- `numpy` (for numerical operations)
- `pandas` (for data manipulation)
- `matplotlib` (for visualization)
- `seaborn` (for advanced plotting)
- `scikit-learn` (for machine learning)
- `xgboost` (for XGBoost model)
- `jupyter` (for interactive development and running notebooks)

You can install the required dependencies using the following `pip` command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

## Data

The dataset consists of golf tournament statistics, including several features related to the player's performance (e.g., strokes gained putting, strokes gained approach, etc.). The **target variable** is whether a golfer **made the cut** (binary: `1` for making the cut, `0` for missing the cut).

Ensure that you have the dataset in `.csv` format, and place it in the appropriate directory. Here's an example of loading the dataset into a pandas DataFrame:

```python
import pandas as pd

# Load the golf data
data = pd.read_csv('path/to/your/golf_data.csv')
```

## Modeling

### Logistic Regression

Logistic Regression is used to model the probability that a golfer makes the cut, based on various features. The `LogisticRegression` model is trained using the **strokes gained** features, and it provides the probability of making the cut.

### XGBoost

XGBoost is an advanced machine learning algorithm that uses boosting (combining multiple weak learners) to create a strong model. It's well-suited for this task as it handles large datasets, non-linear relationships, and overfitting effectively.

Both models are trained on the same features, and their performance is evaluated using common classification metrics.

## Evaluation

The models are evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of true positives among the predicted positives.
- **Recall**: The proportion of true positives among the actual positives.
- **F1-score**: The harmonic mean of precision and recall.
- **AUC (Area Under the Curve)**: A measure of the model's ability to distinguish between classes.

Visualizations such as **confusion matrices** and **ROC curves** are used to better understand the performance of the models.

### Example of Model Evaluation:

```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Split data into features (X) and target (y)
X = data[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']]
y = data['made_cut']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation (accuracy, precision, recall, F1-score)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Detailed classification report for both models
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
```

## Usage

1. **Load the data**: Import the dataset into a pandas DataFrame.
2. **Train models**: Use Logistic Regression and XGBoost to train the models on your data.
3. **Evaluate models**: Use classification metrics like accuracy, precision, recall, F1-score, and AUC to evaluate the models.
4. **Visualize results**: Generate bar plots, confusion matrices, and ROC curves to compare the models.

### Example of Visualizing Model Performance:

```python
import matplotlib.pyplot as plt

# Create a bar plot comparing accuracies of both models
plt.bar(['Logistic Regression', 'XGBoost'], [accuracy_logreg, accuracy_xgb], color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()
```

## Visualization

The project includes several visualization techniques to compare the performance of the models:
- **Accuracy Bar Chart**: Compare the accuracy of the Logistic Regression and XGBoost models.

  
**Dataset used: https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022**

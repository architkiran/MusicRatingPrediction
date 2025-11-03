# Music Review Rating Prediction
## Project Overview

This project predicts **reviewer ratings (1–5)** of music releases using metadata and text features.  
It’s a **multi-class classification** problem, evaluated with **Macro F1 Score** (as per Kaggle rules).

Dataset fields include:
- `summary`, `reviewText`, `genres` (text)
- `VotedHelpful`, `TotalVotes`, `unixReviewTime`, `album_mbid`, etc.

---

## 1. Data Exploration

### Code:
```
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print(train.shape, test.shape)
display(train.head())
```

- This step loads the training and testing datasets and performs an initial inspection.  
- We analyze dimensions, columns, and data samples to understand the type and quality of features available.  
- It helps confirm that the dataset loads correctly and identifies any immediate issues such as null values or inconsistent formatting.

### Key Insights:


#### Explored Score target distribution:
```
train['Score'].value_counts(normalize=True).sort_index()
```

#### Checked missing data:
```
train.isna().mean().sort_values(ascending=False).head(10)
```

## 2. Feature Extraction / Engineering

- Feature engineering transforms raw columns into meaningful representations for the model.  
- We include text preprocessing, numerical transformations, log scaling, and time-based features.  
- These derived features help capture reviewer behavior, review length, and temporal patterns that might correlate with score.


#### Text Cleaning:
```
for col in ['summary', 'reviewText', 'genres']:
    train[col] = train[col].fillna('')
```

#### Helpful Ratio:
```
def helpful_ratio(df):
    denom = df['TotalVotes'].replace({0: np.nan})
    return df['VotedHelpful'] / denom

train['helpful_ratio'] = helpful_ratio(train).fillna(0)
```

#### Text Length and Logs:

```
train['review_len'] = train['reviewText'].str.split().apply(len)
train['summary_len'] = train['summary'].str.split().apply(len)
train['review_len_log'] = np.log1p(train['review_len'])
train['summary_len_log'] = np.log1p(train['summary_len'])
``` 

#### Genre Count:

```
train['genre_count'] = train['genres'].str.split(',').apply(
    lambda x: len([g.strip() for g in x if g.strip()])
)
```

#### Time-based Features:

```
train['review_dt'] = pd.to_datetime(train['unixReviewTime'], unit='s')
train['review_year'] = train['review_dt'].dt.year
train['review_month'] = train['review_dt'].dt.month
train['review_dayofweek'] = train['review_dt'].dt.dayofweek
```

## 3. Model Creation and Assumptions

- The model uses **Multinomial Logistic Regression**, a classic linear algorithm ideal for multi-class text classification.  
- All preprocessing and transformations are wrapped in a single pipeline to ensure reproducibility and cleaner code.

### Model: Multinomial Logistic Regression.

#### Pipeline Setup:

```
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

numeric_features = ['review_year', 'review_month', 'TotalVotes', 'VotedHelpful',
                    'helpful_ratio', 'review_len_log', 'summary_len_log', 'genre_count']

text_features = ['summary', 'reviewText', 'genres']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MaxAbsScaler())
])

preprocessor = ColumnTransformer([
    ('summary_tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1, 2)), 'summary'),
    ('review_tfidf', TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=4), 'reviewText'),
    ('genre_tfidf', TfidfVectorizer(max_features=400, ngram_range=(1, 1)), 'genres'),
    ('numeric', numeric_transformer, numeric_features)
], remainder='drop', sparse_threshold=0.3)

logreg_model = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=300,
        random_state=42
    ))
])
```

### Why Logistic Regression:

- Interpretable and efficient for text data.
- Regularized to prevent overfitting.
- Works natively with sparse TF-IDF matrices.

## 4. Model Tuning

- To improve performance, a **Grid Search with Stratified K-Folds** is used to tune hyperparameters.  
- The parameters tested include regularization strength (`C`) and class balancing (`class_weight`).
- Performed a Grid Search CV on regularization strength (C) and class weights.

```
from sklearn.model_selection import StratifiedKFold, GridSearchCV

param_grid = {
    'classifier__C': [0.5, 1.0, 2.0],
    'classifier__class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    logreg_model,
    param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print('Best parameters:', grid_search.best_params_)
print('Best CV Macro F1:', grid_search.best_score_)
```

## 5. Model Evaluation / Performance

- This step validates the model on held-out data and measures predictive performance using Accuracy, Macro F1, and Weighted F1.  
- The classification report further reveals class-wise recall and precision.


#### Validation Metrics:

```
from sklearn.metrics import accuracy_score, f1_score, classification_report

y_pred = grid_search.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Validation Macro F1:", f1_score(y_val, y_pred, average='macro'))
print(classification_report(y_val, y_pred))
```

#### Typical Results:

- Validation Accuracy ≈ 0.65–0.70
- Macro F1 ≈ 0.45–0.50
- Weighted F1 ≈ 0.69

## 6. Final Model and Kaggle Submission

- After selecting the best hyperparameters, the model is retrained on the **entire training dataset** and used to predict ratings for the unlabeled test set.  
- Predictions are formatted to match Kaggle’s submission requirements.

#### Retraining on Full Data:

```
best_model = grid_search.best_estimator_
best_model.fit(X, y)
```

#### Predict and Export Submission:

```
unlabeled = train['Score'].isna()
X_test_final = train.loc[unlabeled, text_features + numeric_features]

test_preds = best_model.predict(X_test_final)

submission = test[['id']].copy()
submission['Score'] = test_preds.astype(int)
submission.to_csv('submission_final.csv', index=False)
```
## 7. Challenges faced

1. **Imbalanced Ratings:**  
   The dataset was heavily skewed toward 4- and 5-star reviews, causing poor Macro F1 scores.  
   Using `class_weight='balanced'` improved fairness across classes.

2. **Sparse Text Features:**  
   TF-IDF on multiple text fields created large, sparse matrices.  
   Limiting `max_features` and using Logistic Regression helped maintain efficiency.

3. **Missing and Noisy Data:**  
   Several columns had missing values or zeros (especially `TotalVotes`, `VotedHelpful`).  
   Missing numeric data was filled using median imputation, and zeros were log-transformed (`np.log1p`) to stabilize distributions.

4. **Overfitting Risk:**  
   With many derived numeric features (like length logs and helpful ratios), the model could overfit to training data.  
   Cross-validation and regularization (`C` parameter tuning) mitigated this problem.





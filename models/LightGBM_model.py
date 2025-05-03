import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set file path
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "../ds-bu-awards-analysis/data/combine_dataset/cleaned_combined_awards.csv")

# Load data
df = pd.read_csv(file_path, delimiter="\t", encoding="utf-8")

# Preprocessing: drop rows with missing values
df = df.dropna(subset=["awardname", "personname", "stats_Discipline", "award_category", "award_age"])

# Map prestige levels to numeric scores
df["prestige_score"] = df["prestige"].map({
    "Not Designated": 0,
    "Prestigious": 1,
    "Highly Prestigious": 2
})

# Filter out people with less than 5 awards
person_counts = df["personname"].value_counts()
df = df[df["personname"].isin(person_counts[person_counts >= 5].index)]

# Compute award count per person
award_counts = df["personname"].value_counts().to_dict()
df["person_award_count"] = df["personname"].map(award_counts)

# Create year and decade features
df["award_year"] = df["awardreceivedawardyear"]
df["award_decade"] = (df["award_year"] // 10) * 10

# Encode categorical variables
person_encoder = LabelEncoder()
df["person_id"] = person_encoder.fit_transform(df["personname"])

award_encoder = LabelEncoder()
df["award_id"] = award_encoder.fit_transform(df["awardname"])

discipline_encoder = LabelEncoder()
df["discipline_id"] = discipline_encoder.fit_transform(df["stats_Discipline"])

df["category_id"] = LabelEncoder().fit_transform(df["award_category"])
df["decade_id"] = LabelEncoder().fit_transform(df["award_decade"])

# Select features for training
features = [
    "person_award_count",
    "award_age",
    "discipline_id",
    "award_id"
]

X = df[features]
y = df["person_id"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# Train LightGBM model
# ----------------------
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

params_lgb = {
    "objective": "multiclass",
    "num_class": len(y.unique()),
    "metric": "multi_logloss",
    "learning_rate": 0.01,
    "max_depth": 12,
    "num_leaves": 160,
    "min_data_in_leaf": 15,
    "min_gain_to_split": 0.001,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "seed": 42
}

lgb_model = lgb.train(
    params_lgb,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(10)
    ]
)

# ----------------------
# Train XGBoost model
# ----------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    'objective': 'multi:softprob',
    'num_class': len(y.unique()),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.01,
    'max_depth': 12,
    'seed': 42
}

xgb_model = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=30
)

# Evaluate individual models
lgb_preds = lgb_model.predict(X_test)
lgb_pred_class = lgb_preds.argmax(axis=1)
lgb_acc = accuracy_score(y_test, lgb_pred_class)

xgb_preds = xgb_model.predict(dtest)
xgb_pred_class = xgb_preds.argmax(axis=1)
xgb_acc = accuracy_score(y_test, xgb_pred_class)

print(f"LightGBM Accuracy: {lgb_acc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# ----------------------
# Ensemble predictions: Average the probabilities
# ----------------------
ensemble_probs = (lgb_preds + xgb_preds) / 2
ensemble_pred_class = ensemble_probs.argmax(axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred_class)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# ----------------------
# Predict top-k awardees using ensemble
# ----------------------
def predict_top_k_awardees_ensemble(award_name, top_k=10):
    # Normalize award name to match case and whitespace
    normalized_input = award_name.strip().lower()
    award_name_matched = None
    for name in award_encoder.classes_:
        if normalized_input == name.strip().lower():
            award_name_matched = name
            break
    if award_name_matched is None:
        print("Award name not found")
        return []

    # Use default discipline and average age from the dataset
    discipline = df["stats_Discipline"].mode()[0]
    age = int(df["award_age"].mean())
    count = 5

    # Prepare input for prediction
    input_df = pd.DataFrame([{
        "award_id": award_encoder.transform([award_name_matched])[0],
        "discipline_id": discipline_encoder.transform([discipline])[0],
        "award_age": age,
        "person_award_count": count
    }])
    
    # Create DMatrix for XGBoost prediction
    input_df = input_df[X_train.columns]
    dmatrix_input = xgb.DMatrix(input_df)
    
    # Get prediction probabilities from both models
    lgb_prob = lgb_model.predict(input_df)
    xgb_prob = xgb_model.predict(dmatrix_input)
    
    # Average the probabilities for ensemble prediction
    ensemble_prob = (lgb_prob + xgb_prob) / 2
    
    top_k_indices = np.argsort(ensemble_prob[0])[::-1][:top_k]
    top_k_names = person_encoder.inverse_transform(top_k_indices)

    print(f"\nTop {top_k} possible winner(s) of ({award_name_matched}):")
    for i, name in enumerate(top_k_names, 1):
        print(f"{i}. {name}")
    return top_k_names

# User input for ensemble prediction
if __name__ == "__main__":
    user_award = input("Type an award name to predict possible winners: ").strip()
    predict_top_k_awardees_ensemble(user_award)
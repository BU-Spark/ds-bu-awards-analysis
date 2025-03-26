import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "../combine_dataset/cleaned_combined_awards.csv")

df = pd.read_csv(file_path, delimiter="\t", encoding="utf-8")

# preprocessing
df = df.dropna(subset=["awardname", "personname", "stats_Discipline", "award_category", "award_age"])
df["prestige_score"] = df["prestige"].map({
    "Not Designated": 0,
    "Prestigious": 1,
    "Highly Prestigious": 2
})

# filter award count < 10
person_counts = df["personname"].value_counts()
df = df[df["personname"].isin(person_counts[person_counts >= 10].index)]

# selecting award count as feature
award_counts = df["personname"].value_counts().to_dict()
df["person_award_count"] = df["personname"].map(award_counts)

# year related feature
df["award_year"] = df["awardreceivedawardyear"]
df["award_decade"] = (df["award_year"] // 10) * 10

# encoding features
person_encoder = LabelEncoder()
df["person_id"] = person_encoder.fit_transform(df["personname"])
award_encoder = LabelEncoder()
df["award_id"] = award_encoder.fit_transform(df["awardname"])
discipline_encoder = LabelEncoder()
df["discipline_id"] = discipline_encoder.fit_transform(df["stats_Discipline"])
df["category_id"] = LabelEncoder().fit_transform(df["award_category"])
df["decade_id"] = LabelEncoder().fit_transform(df["award_decade"])

# selecting 4 features (award counts, award age, displince_id, awad_id)
features = [
    "person_award_count",
    "award_age",
    "discipline_id",
    "award_id"
]

X = df[features]
y = df["person_id"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# tuning parameters
params = {
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

# training model
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(10)
    ]
)

# F1 score
"""
y_pred = model.predict(X_test)
y_pred_class = y_pred.argmax(axis=1)
f1 = f1_score(y_test, y_pred_class, average="macro")
print(f"\n LightGBM Macro F1 Score: {f1:.4f}")
"""

# predictive function to top k
def predict_top_k_awardees_simple(award_name, top_k=3):
    if award_name not in award_encoder.classes_:
        print("Award name not found")
        return []

    discipline = df["stats_Discipline"].mode()[0]
    age = int(df["award_age"].mean())
    count = 5

    input_df = pd.DataFrame([{
        "award_id": award_encoder.transform([award_name])[0],
        "discipline_id": discipline_encoder.transform([discipline])[0],
        "award_age": age,
        "person_award_count": count
    }])

    probs = model.predict(input_df)
    top_k_indices = np.argsort(probs[0])[::-1][:top_k]
    top_k_names = person_encoder.inverse_transform(top_k_indices)

    print(f"\n Top {top_k} possible winner of ({award_name}):")
    for i, name in enumerate(top_k_names, 1):
        print(f"{i}. {name}")
    return top_k_names

# user input
if __name__ == "__main__":
    user_award = input("Type an award name to predict possible winners: ").strip()
    predict_top_k_awardees_simple(user_award)
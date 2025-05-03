import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from difflib import get_close_matches

# File path to dataset (update if needed)
file_path = "../ds-bu-awards-analysis/data/combine_dataset/cleaned_combined_awards.csv"

def load_and_prepare_data():
    # Load and prepare the awards data
    df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')

    faculty_profiles = {}
    faculty_groups = df.groupby('personname')

    for faculty_name, faculty_data in faculty_groups:
        faculty_data = faculty_data.sort_values('awardreceivedawardyear')

        faculty_profiles[faculty_name] = {
            'awards': faculty_data.to_dict('records'),
            'award_count': len(faculty_data),
            'prestige_counts': faculty_data['prestige'].value_counts().to_dict(),
            'category_counts': faculty_data['award_category'].value_counts().to_dict(),
            'earliest_award_year': faculty_data['awardreceivedawardyear'].min(),
            'latest_award_year': faculty_data['awardreceivedawardyear'].max(),
            'career_span': faculty_data['awardreceivedawardyear'].max() - faculty_data['awardreceivedawardyear'].min(),
            'award_names': list(faculty_data['awardname']),
            'award_societies': list(faculty_data['awardgoverningsocietyname'])
        }

    return df, faculty_profiles

def create_feature_matrix(faculty_profiles, target_award):
    # Create feature matrix for predicting top candidates for an award
    features = []
    labels = []
    faculty_names = []

    for faculty_name, profile in faculty_profiles.items():
        has_target = any(target_award in award['awardname'] or 
                         target_award in award['awardgoverningsocietyname']
                         for award in profile['awards'])

        feature_dict = {
            'total_awards': profile['award_count'],
            'not_designated_count': profile['prestige_counts'].get('Not Designated', 0),
            'prestigious_count': profile['prestige_counts'].get('Prestigious', 0),
            'highly_prestigious_count': profile['prestige_counts'].get('Highly Prestigious', 0),
            'standard_award_count': profile['category_counts'].get('Standard Award', 0),
            'notable_award_count': profile['category_counts'].get('Notable Award', 0),
            'major_award_count': profile['category_counts'].get('Major Award', 0),
            'career_span': profile['career_span'] if profile['career_span'] > 0 else 0
        }

        features.append(feature_dict)
        labels.append(1 if has_target else 0)
        faculty_names.append(faculty_name)

    X = pd.DataFrame(features)
    y = np.array(labels)

    return X, y, faculty_names

def train_svm_model(X, y):
    # Train SVM model with error handling for small datasets
    if len(set(y)) < 2:
        print("\n⚠ Warning: Not enough data for a proper train-test split.")
        print("Using entire dataset for training.")
        X_train, y_train = X, y
        X_test, y_test = X, y  # Dummy test set, not used for evaluation
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Calculate class weight for imbalanced data
    class_weight = {
        0: 1,
        1: len(y[y == 0]) / max(len(y[y == 1]), 1)
    }

    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    if len(set(y)) >= 2:
        y_pred = model.predict(X_test)
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("\n⚠ Small dataset: Model trained but no test evaluation available.")

    return model

def predict_award_likelihood(model, X, names):
    # Predict likelihood of winning an award for all faculty
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        'name': names,
        'likelihood': probabilities
    })

    results = results.sort_values('likelihood', ascending=False)
    return results

def predict_for_award(award_name, faculty_profiles):
    # Predict the top faculty candidates for a specific award
    print(f"\nPREDICTING CANDIDATES FOR: {award_name}")

    X, y, faculty_names = create_feature_matrix(faculty_profiles, target_award=award_name)

    if sum(y) == 0:
        print(f"No faculty members have previously received or are a strong fit for {award_name}.")
        return

    model = train_svm_model(X, y)
    predictions = predict_award_likelihood(model, X, faculty_names)

    # Display top 20 candidates
    print(f"\nTop 20 candidates for {award_name}:")
    for i, (_, candidate) in enumerate(predictions.head(20).iterrows()):
        print(f"{i+1}. {candidate['name']}: {candidate['likelihood']*100:.2f}% likelihood")

    return predictions

def find_similar_award(input_award, all_awards, max_matches=5):
    # Find similar award names if exact match isn't found
    matches = get_close_matches(input_award, all_awards, n=max_matches, cutoff=0.6)
    return matches

def interactive_mode():
    # Run interactive mode to predict candidates for an award
    print("Loading and preparing data...")
    df, faculty_profiles = load_and_prepare_data()
    
    # Get all unique award names for matching
    all_awards = set()
    for profile in faculty_profiles.values():
        all_awards.update(profile['award_names'])
    all_awards = list(all_awards)

    while True:
        award_name = input("\nEnter award name (or type 'exit' to quit): ").strip()
        if award_name.lower() == 'exit':
            print("Exiting...")
            break

        # Check if award exists or find similar ones
        if award_name not in all_awards:
            similar_awards = find_similar_award(award_name, all_awards)
            if similar_awards:
                print(f"\nAward '{award_name}' not found. Did you mean one of these?")
                for i, award in enumerate(similar_awards):
                    print(f"{i+1}. {award}")
                choice = input("\nEnter the number of your choice (or press Enter to use your original input): ")
                if choice.isdigit() and 1 <= int(choice) <= len(similar_awards):
                    award_name = similar_awards[int(choice)-1]
                    print(f"\nUsing award: {award_name}")

        predict_for_award(award_name, faculty_profiles)

if __name__ == "__main__":
    interactive_mode()
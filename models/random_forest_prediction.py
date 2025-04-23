import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

COMMON_PATHWAY_AWARDS = [
    "Faculty Early Career Development (CAREER) Program",
    "Fellow",
    "Fulbright Scholar",
    "NBER Affiliated Scholars",
    "Guggenheim Fellowship",
    "AAAS Fellow",
    "NIH Pathway to Independence Award (K99/R00)",
    "APS Fellow",
    "Fulbright U.S. Student Program",
    "NIH Outstanding Investigator Award (R35)",
    "Kavli Frontiers of Science Fellows",
    "Scialog Fellows",
    "Templeton Grants",
    "Summer Stipends Awards",
    "IEEE Fellow",
]

def load_and_prepare_data(file_path):
    """
    Load and prepare the awards data for modeling
    """
    df = pd.read_csv(file_path, delimiter='\t')
    
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
    """
    Create feature matrix for prediction
    """
    features = []
    labels = []
    faculty_names = []
    
    for faculty_name, profile in faculty_profiles.items():
        has_target = False
        for award in profile['awards']:
            if ((award['awardname'] and target_award in award['awardname']) or 
                (award['awardgoverningsocietyname'] and target_award in award['awardgoverningsocietyname'])):
                has_target = True
                break
        
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
        #COMMON PATHWAY AWARDS
        for precursor in COMMON_PATHWAY_AWARDS:
            precursor_field = f"has_{precursor.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}"
            feature_dict[precursor_field] = 1 if any(precursor in award_name for award_name in profile['award_names']) else 0

        features.append(feature_dict)
        labels.append(1 if has_target else 0)
        faculty_names.append(faculty_name)
    
    X = pd.DataFrame(features)
    y = np.array(labels)
    
    return X, y, faculty_names

def train_random_forest_model(X, y, high_accuracy=False):
    """
    Train a Random Forest model with parameters optimized based on dataset size
    """
    positive_count = sum(y)
    small_dataset = positive_count < 50  # Define small dataset threshold
    
    # Adjust class weights to handle imbalance
    class_weights = {
        0: 1,
        1: sum(y == 0) / max(sum(y == 1), 1)  # Avoid division by zero
    }

    # Configure model parameters based on dataset size and accuracy preference
    if small_dataset:
        print(f"Small dataset detected ({positive_count} positive examples). Using specialized parameters to prevent overfitting.")
        model = RandomForestClassifier(
            n_estimators=10 if not high_accuracy else 50,  # Fewer trees for small datasets
            max_depth=3,                # Limit depth to prevent overfitting
            min_samples_split=6,        # Require more samples to split
            min_samples_leaf=4,         # Require more samples at leaf nodes
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,                  # Use all CPU cores
            max_features='sqrt'         # Consider subset of features
        )
    else:
        # Standard parameters for larger datasets
        model = RandomForestClassifier(
            n_estimators=75 if not high_accuracy else 100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    
    # For small datasets, show cross-validation metrics for better evaluation
    if small_dataset:
        cv_folds = min(5, positive_count)  # Adjust folds based on dataset size
        if cv_folds >= 3:  # Only perform CV if we have enough data
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1', n_jobs=-1)
            print(f"Cross-validation F1 scores ({cv_folds}-fold): {cv_scores}")
            print(f"Mean F1: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    print("Model performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    return model, feature_importance, metrics

def predict_award_likelihood(model, X, faculty_names):
    """
    Predict likelihood of winning award for all faculty
    """
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1
    
    results = pd.DataFrame({
        'faculty_name': faculty_names,
        'likelihood': probabilities
    })

    results = results.sort_values('likelihood', ascending=False)
    
    return results

def predict_for_award(award_name, faculty_profiles, output_filename=None,high_accuracy=False):
    """
    Parameters:
    award_name (str): Name of the award to predict candidates for
    faculty_profiles (dict): Dictionary of faculty profiles
    output_filename (str, optional): If provided, save results to CSV file
    
    Returns:
    pd.DataFrame: Ranked list of faculty candidates
    """
    print(f"\n\n{'='*80}")
    print(f"PREDICTION MODEL FOR: {award_name}")
    print(f"{'='*80}")
    
    # Feature matrix
    print(f"\nCreating features for {award_name}...")
    X, y, faculty_names = create_feature_matrix(faculty_profiles, award_name)
    
    # Check class balance
    positive_count = sum(y)
    total_count = len(y)
    print(f"Class distribution: {positive_count} positive examples ({positive_count/total_count*100:.2f}%) out of {total_count} total")
    
    # Training model 
    print("\nTraining random forest model...")
    model, feature_importance, metrics = train_random_forest_model(X, y, high_accuracy=high_accuracy)
    
    # Predict for all faculty
    print("\nPredicting award likelihood for all faculty...")
    predictions = predict_award_likelihood(model, X, faculty_names)
    
    # Show top candidates
    print(f"\nTop 20 candidates for {award_name}:")
    top_candidates = predictions.head(20)
    
    for i, (_, candidate) in enumerate(top_candidates.iterrows()):
        print(f"{i+1}. {candidate['faculty_name']}: {candidate['likelihood']*100:.2f}% likelihood")
    
    # Visualize feature importance
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title(f'Top 10 Features for Predicting {award_name}')
    plt.tight_layout()
    plt.savefig(f'outputs/{award_name.replace(" ", "_")}_feature_importance.png')
    """
    #Can save to CSV if desired
    if output_filename:
        predictions.head(100).to_csv(f'outputs/{output_filename}', index=False)
        print(f"Saved predictions to outputs/{output_filename}")
        
    return predictions, metrics

def find_similar_awards(search_term, df):
    """
    Find awards in the dataset that match a search term
    
    Parameters:
    search_term (str): Term to search for in award names or governing societies
    df (pd.DataFrame): Awards dataframe
    
    Returns:
    list: Matching award names
    """
    #Award names
    award_matches = df[df['awardname'].str.contains(search_term, case=False, na=False)]['awardname'].unique()
    
    #Governing societies
    society_matches = df[df['awardgoverningsocietyname'].str.contains(search_term, case=False, na=False)]['awardgoverningsocietyname'].unique()
    
    #Combine results
    all_matches = list(award_matches) + list(society_matches)
    return sorted(list(set(all_matches)))

def interactive_mode():
    """
    Run the model in interactive mode to recommend faculty for specific awards
    """
    file_path = 'combine_dataset/cleaned_combined_awards.csv'   
    print("Loading and preparing data...")
    df, faculty_profiles = load_and_prepare_data(file_path)
    
    while True:
        print("\n" + "="*80)
        print("FACULTY AWARD PREDICTION SYSTEM")
        print("="*80)
        print("\nOptions:")
        print("1. Predict candidates for a specific award")
        print("2. Search for awards by keyword")
        print("3. Run predictions for top prestigious awards")
        print("4. High accuracy prediction for a specific award")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            award_name = input("\nEnter award name: ")
            save_option = input("Save results to CSV? (y/n): ")
            output_file = f"{award_name.replace(' ', '_')}_candidates.csv" if save_option.lower() == 'y' else None
            
            try:
                predict_for_award(award_name, faculty_profiles, output_file)
            except Exception as e:
                print(f"Error: {e}")
                print("Could not run prediction. Make sure the award name exists in the dataset.")
        
        elif choice == '2':
            search_term = input("\nEnter keyword to search for awards: ")
            matching_awards = find_similar_awards(search_term, df)
            
            if matching_awards:
                print(f"\nFound {len(matching_awards)} matching awards:")
                for i, award in enumerate(matching_awards):
                    print(f"{i+1}. {award}")
            else:
                print("No matching awards found.")
        
        elif choice == '3':
            #Target prestigious awards
            target_awards = [
                "American Academy of Arts and Sciences",
                "National Academy of Sciences",
                "National Academy of Engineering",
                "National Academy of Medicine",
                "AAAS Fellow"
            ]
            
            for award in target_awards:
                predict_for_award(award, faculty_profiles, f"{award.replace(' ', '_')}_candidates.csv")
        
        elif choice == '4':
            award_name = input("\nEnter award name: ")
            save_option = input("Save results to CSV? (y/n): ")
            output_file = f"{award_name.replace(' ', '_')}_high_accuracy_candidates.csv" if save_option.lower() == 'y' else None
            
            try:
                predict_for_award(award_name, faculty_profiles, output_file, high_accuracy=True)
            except Exception as e:
                print(f"Error: {e}")
                print("Could not run prediction. Make sure the award name exists in the dataset.")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

def main():
    """
    Main function to run the faculty award prediction model
    """
    if not os.path.exists('data'):
        print("Error: 'data' directory not found. Please create it and add the cleaned_combined_awards.csv file.")
        return
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory for storing results.")
    
    if not os.path.exists('data/cleaned_combined_awards.csv'):
        print("Error: cleaned_combined_awards.csv not found in the data directory.")
        return
    
    # ayyy interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()
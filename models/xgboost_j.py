import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from difflib import get_close_matches

# Set file path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "combine_dataset", "cleaned_combined_awards.csv")

def load_and_preprocess_data():
    """Load and preprocess data"""
    # Load data
    df = pd.read_csv(file_path, delimiter="\t", encoding="utf-8")
    
    # Remove rows with missing values in essential columns
    df = df.dropna(subset=["awardname", "personname", "stats_Discipline", "award_category", "award_age"])
    
    # Quantify prestige
    df["prestige_score"] = df["prestige"].map({
        "Not Designated": 0,
        "Prestigious": 1,
        "Highly Prestigious": 2
    })
    
    # Create faculty profiles
    faculty_profiles = create_faculty_profiles(df)
    
    # Filter for people with at least 5 awards
    person_counts = df["personname"].value_counts()
    df = df[df["personname"].isin(person_counts[person_counts >= 5].index)]
    
    # Calculate award count per person
    award_counts = df["personname"].value_counts().to_dict()
    df["person_award_count"] = df["personname"].map(award_counts)
    
    # Create year and decade features
    df["award_year"] = df["awardreceivedawardyear"]
    df["award_decade"] = (df["award_year"] // 10) * 10
    
    # Encode categorical variables
    df, encoders = encode_categorical_variables(df)
    
    return df, faculty_profiles, encoders

def create_faculty_profiles(df):
    """Create profile for each faculty member"""
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
            'award_societies': list(faculty_data['awardgoverningsocietyname']),
            'disciplines': list(faculty_data['stats_Discipline'].unique()),
            'avg_award_age': faculty_data['award_age'].mean()
        }
        
        # Analyze award pattern (additional features)
        years = faculty_data['awardreceivedawardyear'].values
        if len(years) >= 2:
            faculty_profiles[faculty_name]['award_frequency'] = np.mean(np.diff(np.sort(years)))
        else:
            faculty_profiles[faculty_name]['award_frequency'] = 0
    
    return faculty_profiles

def encode_categorical_variables(df):
    """Encode categorical variables"""
    encoders = {}
    
    # Create encoders for person, award, discipline, category, decade
    for column, new_column in [
        ("personname", "person_id"),
        ("awardname", "award_id"),
        ("stats_Discipline", "discipline_id"),
        ("award_category", "category_id"),
        ("award_decade", "decade_id")
    ]:
        encoder = LabelEncoder()
        df[new_column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    
    return df, encoders

def create_advanced_features(df):
    """Create advanced features"""
    # Time interval between awards
    df_sorted = df.sort_values(['personname', 'award_year'])
    df_sorted['prev_award_year'] = df_sorted.groupby('personname')['award_year'].shift(1)
    df_sorted['time_since_last_award'] = df_sorted['award_year'] - df_sorted['prev_award_year']
    
    # Handle NaN values for first award
    df_sorted['time_since_last_award'] = df_sorted['time_since_last_award'].fillna(0)
    
    # Cumulative prestige score
    df_sorted['cum_prestige_score'] = df_sorted.groupby('personname')['prestige_score'].cumsum()
    
    # Number of previous awards
    df_sorted['previous_awards'] = df_sorted.groupby('personname').cumcount()
    
    # Ratio of awards in the same discipline
    discipline_counts = df.groupby(['personname', 'stats_Discipline']).size()
    person_counts = df.groupby('personname').size()
    discipline_ratio = discipline_counts / person_counts
    
    # Convert Series to Dictionary for mapping
    discipline_ratio_dict = discipline_ratio.to_dict()
    
    def get_discipline_ratio(row):
        key = (row['personname'], row['stats_Discipline'])
        return discipline_ratio_dict.get(key, 0)
    
    df_sorted['discipline_ratio'] = df_sorted.apply(get_discipline_ratio, axis=1)
    
    return df_sorted

def prepare_data_for_training(df, target='person_id'):
    """Prepare data for training"""
    # Select basic features
    basic_features = [
        "person_award_count",
        "award_age",
        "discipline_id",
        "award_id",
        "category_id",
        "prestige_score"
    ]
    
    # Add advanced features if available
    advanced_features = [
        "time_since_last_award",
        "cum_prestige_score",
        "previous_awards",
        "discipline_ratio"
    ]
    
    # Combine all available features
    features = basic_features.copy()
    for feature in advanced_features:
        if feature in df.columns:
            features.append(feature)
    
    X = df[features]
    y = df[target]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test, features

def train_xgboost_model(X_train, y_train, X_test, y_test, num_class):
    """Train XGBoost model"""
    # Create XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set hyperparameters
    params_xgb = {
        'objective': 'multi:softprob',
        'num_class': num_class,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,
        'max_depth': 12,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42
    }
    
    # Train model
    xgb_model = xgb.train(
        params_xgb,
        dtrain,
        num_boost_round=1000,
        evals=[(dtest, 'eval')],
        early_stopping_rounds=30
    )
    
    # Predict and evaluate test set
    preds = xgb_model.predict(dtest)
    pred_class = preds.argmax(axis=1)
    accuracy = accuracy_score(y_test, pred_class)
    
    print(f"XGBoost accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred_class))
    
    return xgb_model

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance"""
    # XGBoost feature importance
    importance = model.get_score(importance_type='gain')
    feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
    importance_dict = {feature_map.get(f, f): v for f, v in importance.items()}
    importance_sorted = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    print("\nXGBoost feature importance:")
    for feature, score in importance_sorted.items():
        print(f"{feature}: {score}")
    
    return importance_sorted

def cross_validate_model(X, y, n_splits=5):
    """Evaluate model performance using cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = []
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params_xgb = {
            'objective': 'multi:softprob',
            'num_class': len(y.unique()),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.01,
            'max_depth': 12,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'verbosity': 0  # No output
        }
        
        xgb_model = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=30,
            verbose_eval=False  # No output
        )
        
        # Predict on validation set
        dval = xgb.DMatrix(X_val)
        preds = xgb_model.predict(dval)
        pred_class = preds.argmax(axis=1)
        accuracy = accuracy_score(y_val, pred_class)
        
        print(f"Fold {i+1} accuracy: {accuracy:.4f}")
        scores.append(accuracy)
    
    print(f"\nCross-validation average accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    
    return scores

def predict_top_candidates_for_award(award_name, xgb_model, df, encoders, faculty_profiles, features, top_k=10):
    """Predict top candidates for a specific award"""
    # Normalize and match award name
    normalized_input = award_name.strip().lower()
    award_name_matched = None
    
    for name in encoders['awardname'].classes_:
        if normalized_input == name.strip().lower():
            award_name_matched = name
            break
    
    if award_name_matched is None:
        close_matches = get_close_matches(normalized_input, 
                                        [name.strip().lower() for name in encoders['awardname'].classes_], 
                                        n=3, cutoff=0.6)
        
        if close_matches:
            print(f"Award name not found. Did you mean one of these?")
            for i, match in enumerate(close_matches, 1):
                # Find original case-sensitive name
                for name in encoders['awardname'].classes_:
                    if match == name.strip().lower():
                        print(f"{i}. {name}")
                        break
        else:
            print("Award name not found.")
        
        return []
    
    # Use mean/mode values from dataset
    discipline = df["stats_Discipline"].mode()[0]
    age = int(df["award_age"].mean())
    count = 5
    category = df[df["awardname"] == award_name_matched]["award_category"].mode()[0]
    prestige = df[df["awardname"] == award_name_matched]["prestige"].mode()[0]
    prestige_score = {"Not Designated": 0, "Prestigious": 1, "Highly Prestigious": 2}[prestige]
    
    # Prepare input for prediction
    input_dict = {
        "award_id": encoders['awardname'].transform([award_name_matched])[0],
        "discipline_id": encoders['stats_Discipline'].transform([discipline])[0],
        "award_age": age,
        "person_award_count": count,
        "category_id": encoders['award_category'].transform([category])[0],
        "prestige_score": prestige_score
    }
    
    # Add advanced features if used in model (using mean values)
    if "time_since_last_award" in features:
        input_dict["time_since_last_award"] = df["time_since_last_award"].mean()
    if "cum_prestige_score" in features:
        input_dict["cum_prestige_score"] = df["cum_prestige_score"].mean()
    if "previous_awards" in features:
        input_dict["previous_awards"] = df["previous_awards"].mean()
    if "discipline_ratio" in features:
        input_dict["discipline_ratio"] = df["discipline_ratio"].mean()
    
    # Maintain same column order as used in training
    input_df = pd.DataFrame([{feature: input_dict.get(feature, 0) for feature in features}])
    
    # Create DMatrix and predict
    dmatrix_input = xgb.DMatrix(input_df)
    probabilities = xgb_model.predict(dmatrix_input)[0]
    
    # Extract top k indices and names
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_names = encoders['personname'].inverse_transform(top_k_indices)
    top_k_probs = probabilities[top_k_indices]
    
    print(f"\nTop {top_k} potential recipients for {award_name_matched}:")
    
    results = []
    for i, (name, prob) in enumerate(zip(top_k_names, top_k_probs), 1):
        print(f"{i}. {name}: {prob*100:.2f}% probability")
        
        # Additional information: awards received, disciplines, etc.
        if name in faculty_profiles:
            profile = faculty_profiles[name]
            disciplines = ', '.join(profile['disciplines'])
            print(f"   Disciplines: {disciplines}")
            print(f"   Total awards: {profile['award_count']}")
            
            # Check if they've already received similar awards
            similar_awards = []
            for award in profile['award_names']:
                if award != award_name_matched and award_name_matched in award:
                    similar_awards.append(award)
            
            if similar_awards:
                print(f"   Similar awards: {', '.join(similar_awards)}")
        
        results.append((name, prob))
    
    return results

def analyze_award_patterns(df, faculty_profiles):
    """Analyze and visualize award patterns"""
    # Distribution by award category
    plt.figure(figsize=(10, 6))
    category_counts = df['award_category'].value_counts()
    sns.barplot(x=category_counts.values, y=category_counts.index)
    plt.title('Distribution by Award Category')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.close()
    
    # Distribution by award year
    plt.figure(figsize=(12, 6))
    sns.histplot(df['award_year'], bins=30)
    plt.title('Award Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Awards')
    plt.tight_layout()
    plt.close()
    
    # Top awardees analysis
    top_awardees = pd.Series({name: profile['award_count'] 
                             for name, profile in faculty_profiles.items()})
    top_awardees = top_awardees.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_awardees.values, y=top_awardees.index)
    plt.title('Top 20 Award Recipients')
    plt.xlabel('Number of Awards')
    plt.tight_layout()
    plt.close()
    
    # Distribution by academic discipline
    discipline_counts = df['stats_Discipline'].value_counts().head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=discipline_counts.values, y=discipline_counts.index)
    plt.title('Award Distribution by Academic Discipline (Top 15)')
    plt.xlabel('Number of Awards')
    plt.tight_layout()
    plt.close()
    
    return {
        'category_counts': category_counts,
        'top_awardees': top_awardees,
        'discipline_counts': discipline_counts
    }

def create_award_target_prediction_model(award_name, faculty_profiles, encoders, feature_list=None):
    """Create target prediction model for a specific award"""
    print(f"\nCreating prediction model for {award_name}...")
    
    # Create feature matrix
    features = []
    labels = []
    faculty_names = []
    
    for faculty_name, profile in faculty_profiles.items():
        # Check if they've received the target award
        has_target = award_name in profile['award_names']
        
        # Basic features
        feature_dict = {
            'total_awards': profile['award_count'],
            'not_designated_count': profile['prestige_counts'].get('Not Designated', 0),
            'prestigious_count': profile['prestige_counts'].get('Prestigious', 0),
            'highly_prestigious_count': profile['prestige_counts'].get('Highly Prestigious', 0),
            'standard_award_count': profile['category_counts'].get('Standard Award', 0),
            'notable_award_count': profile['category_counts'].get('Notable Award', 0),
            'major_award_count': profile['category_counts'].get('Major Award', 0),
            'career_span': profile['career_span'] if profile['career_span'] > 0 else 0,
            'award_frequency': profile.get('award_frequency', 0),
            'avg_award_age': profile.get('avg_award_age', 0)
        }
        
        # Additional features if provided
        if feature_list:
            feature_dict = {k: feature_dict[k] for k in feature_list if k in feature_dict}
        
        features.append(feature_dict)
        labels.append(1 if has_target else 0)
        faculty_names.append(faculty_name)
    
    X = pd.DataFrame(features)
    y = np.array(labels)
    
    # Check if there are enough positive samples
    positive_count = sum(y)
    if positive_count < 5:
        print(f"Warning: Too few positive samples ({positive_count}). Model reliability may be low.")
    
    # Split into training and test sets
    if len(set(y)) < 2 or positive_count < 3:
        print("Warning: Too little data, training on full dataset.")
        X_train, y_train = X, y
        X_test, y_test = X, y  # Dummy test set
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Calculate weight for handling imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    if len(set(y)) >= 2 and positive_count >= 3:
        y_pred = model.predict(X_test)
        print("\nModel performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
    
    # Predict on full dataset
    probabilities = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        'name': faculty_names,
        'likelihood': probabilities,
        'has_award': labels
    })
    
    results = results.sort_values('likelihood', ascending=False)
    
    # Output top candidates
    print(f"\nTop 10 candidates for {award_name}:")
    for i, (_, candidate) in enumerate(results.head(10).iterrows()):
        status = "Current recipient" if candidate['has_award'] else "Predicted candidate"
        print(f"{i+1}. {candidate['name']}: {candidate['likelihood']*100:.2f}% probability ({status})")
    
    return model, results

def interactive_mode(df, faculty_profiles, encoders, xgb_model, features):
    """Interactive award prediction system"""
    print("\n=== Award Recipient Prediction System ===")
    
    while True:
        print("\nEnter award name for target prediction model (or 'exit' to quit): ")
        award_name = input().strip()
        
        if award_name.lower() == 'exit':
            print("Exiting program.")
            break
        
        # Normalize and match award name
        normalized_input = award_name.strip().lower()
        award_name_matched = None
        
        for name in encoders['awardname'].classes_:
            if normalized_input == name.strip().lower():
                award_name_matched = name
                break
        
        if award_name_matched is None:
            close_matches = get_close_matches(normalized_input, 
                                           [name.strip().lower() for name in encoders['awardname'].classes_], 
                                           n=3, cutoff=0.6)
            
            if close_matches:
                print(f"Award name not found. Did you mean one of these?")
                for i, match in enumerate(close_matches, 1):
                    # Find original case-sensitive name
                    for name in encoders['awardname'].classes_:
                        if match == name.strip().lower():
                            print(f"{i}. {name}")
                            break
                
                print("\nPlease try again with the correct award name.")
            else:
                print("Award name not found. Please try again.")
        else:
            # Create target prediction model
            _, results = create_award_target_prediction_model(award_name_matched, faculty_profiles, encoders)

def main():
    """Main execution function"""
    print("Loading and preprocessing data...")
    df, faculty_profiles, encoders = load_and_preprocess_data()
    
    # Create advanced features
    print("Creating advanced features...")
    df = create_advanced_features(df)
    
    # Prepare training data
    print("Preparing training data...")
    X, y, X_train, X_test, y_train, y_test, features = prepare_data_for_training(df)
    
    # Cross validation (optional)
    print("Performing cross-validation...")
    cv_scores = cross_validate_model(X, y)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test, len(y.unique()))
    
    # Feature importance analysis
    print("Analyzing feature importance...")
    feature_importance = feature_importance_analysis(xgb_model, features)
    
    # Award pattern analysis (optional)
    print("Analyzing award patterns...")
    pattern_analysis = analyze_award_patterns(df, faculty_profiles)
    
    # Interactive mode
    interactive_mode(df, faculty_profiles, encoders, xgb_model, features)

if __name__ == "__main__":
    main()
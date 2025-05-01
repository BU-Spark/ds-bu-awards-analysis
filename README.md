# BU Awards Analysis

This repository contains the workflow, analysis, and visualizations for the project "BU Award Analysis" for CDS DS 539 Spring 2025, at Boston University.

## Overview
This project aims to help Boston University’s Strategy & Innovation department in identifying the most strategic awards for faculty to apply for based on historical data. 

The main objective is to build a tool that recommends awards to faculty by analyzing trends in award progression, prior recognitions, academic age, and institutional affiliations. We conducted data cleaning, EDA, and combined the two datasets to perform model predictions. By using data from Academic Analytics, we analyzed patterns from past award winners and compared that to current BU faculty to predict their likelihood of winning the target awards. 

Our goal is to improve and streamline that process through a machine learning-based system that improves predictive accuracy. 

### Project Description
Academic Analytics is a platform that allows universities to benchmark their research accomplishments against other peer institutions. One aspect of this is allowing them to compare awards. The client is interested in finding a more streamlined and accurate way to predict faculty’s likelihood of winning an award based on their academic history, including length of career and pathway awards. 

### Project Checklist
Combine and clean the BU faculty award history datasets, resolving award name mismatches using similarity scoring.
Engineer relevant features such as award prestige, academic age, and prior award count to support predictive modeling.
Develop and evaluate multiple machine learning models (XGBoost, Random Forest, LightGBM) to predict the likelihood of faculty receiving specific awards.
Identify pathways to high-prestige awards (e.g., American Academy of Arts and Sciences) and analyze trends in award progression.
Validate model results by cross-referencing faculty known to have received awards.
Document the data preprocessing pipeline, model performance, and limitations for future teams.

## Setup

### Requirements
- Python 3.9+
- Other dependencies listed in `requirements.txt`

### Getting Started

To get started with the project locally:

1. **Clone the repository**
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **(Optional) Create and activate a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## How to Run the Code

To run any Python script from this project, follow these steps in your terminal:

```
# Step 1: Navigate to the root of the project (where you cloned the repository)
cd path/to/your/project

# Step 2: List the contents to make sure you're in the right place
ls
# You should see something like:
# data/ dataset-documentation/ hypothesis_eda_scripts/ models/ README.md/ requirements.txt/

# Step 3: Change into the src folder where the scripts are
cd models

# Step 4: List the Python scripts available
Ls
# You should see something like this:
# LightGBM_model.py/ dtreemodel.py/ random_forest_model.py/ support_vector_machine.py/ xgboost_j.py/ xgmodel.py/

# Step 5: Run the desired Python script
python model1.py
# Replace 'model1.py' with the script you want to run
```

### Key Folder

- `models/` – Contains all model implementation files and utility scripts.

The most important files are the combine_dataset.ipynb in our combine_dataset folder as well as the files in the src folder.

## Dataset

### Discipline Specific Pathways

| Column Name                                 | Data Type | Description |
|--------------------------------------------|-----------|-------------|
| `Focal Award`                               | string    | Main award received |
| `Discipline`                                | string    | Category of award |
| `Pathway Award`                             | string    | Additional awards given after focal award |
| `Awardees w/ This Award`                    | integer   | How many people won this award |
| `Percentage of Awardees With This Award`    | float     | (Awardees with this award) / (Award nominees) — convert to float |
| `Mean Academic Age at Time of Award`        | integer   | Average academic age at time of winning the award (years in academia/research) |

### Awards Dataset Metadata

| Column Name                 | Data Type | Description |
|----------------------------|-----------|-------------|
| `awardgoverningsocietyid`  | integer   | ID for the governing society that grants the award |
| `awardgoverningsocietyname`| string    | Name of the governing society that grants the award |
| `awardid`                  | integer   | ID for the award |
| `awardname`                | string    | Name of the award |
| `awardreceivedid`          | integer   | ID for a specific instance of the award being received |
| `awardreceivedname`        | string    | Year in which the award was received |
| `AAUID`                    | integer   | ID assigned to the award recipient |
| `clientfacultyid`          | integer   | ID for faculty member within the institution |
| `orcid`                    | integer   | ORCID identifier for the award recipient (if available) |
| `personname`               | string    | Full name, formatted as "Last name, First name" |
| `institutionname`          | string    | Name of the institution associated with the recipient |
| `prestige`                 | string    | Classification of the award’s prestige |
| `asofdate`                 | Date/Time | Date when the record was last updated |

### Data Structure
```
├── data/                     # All dataset files
│   ├── clean_dataset/           # Cleaned dataset
│   ├── combine_dataset/         # Combined RI_Matches and Discipline_Pathways dataset
│   ├── raw_dataset/             # Original dataset files
├── dataset-documentation/    # Dataset documentation and metadata
├── scripts/                  # Hypothesis Testing and EDA Python Scripts
├── src/                      # Scripts for training model
├── README.md                 # Project overview and instructions
├── requirements.txt          # Dependencies for random forest model
```


### Proposed Solution

Goal : Develop a machine learning system that predicts which BU faculty are most likely to win a given award based on their academic history.

Approach :
Combined and cleaned two datasets (Discipline Specific Pathways + RI Matches) using similarity matching.
Trained 5 machine learning models (XGBoost, Random Forest, LightGBM, Support Vector Machine, Decision Tree) using features like academic age, discipline, and prior awards.
Final model returns a ranked list of top candidates for any given award.

Output : 
For any selected award, the model returns a ranked list of top BU faculty with the highest likelihood of receiving it.
Outputs include predicted probabilities and feature importances for interpretability.

Impact : Supports long-term goal of automating and scaling the award recommendation workflow.

### Challenges

One of the main challenges we faced was handling the RI_Matches file, which contained nearly 300,000 rows. Combining it with the discipline dataset significantly slowed down the process and required careful handling. Additionally, we encountered inconsistency in our model results. Despite using the same input, each model yielded varying outputs, making it difficult to pinpoint the most reliable model. We were also limited by the size of the Discipline Pathways dataset, only using a portion of it due to memory and processing constraints. This likely impacted the model's ability to capture the full range of award trajectories.

### Next Steps

For any future team reading this, it would be best to start with making sure the model does perform accurately. Right now, our models get about 80% accuracy, however, we are unable to validate whether this is in line with the client's model. Another next step is to create a UI interface since the project is designed for non-technical people. Having an interface would make it easier to demonstrate the results of the model.




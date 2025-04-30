# BU Awards Analysis

This repository contains the workflow, analysis, and visualizations for the project "BU Award Analysis" for CDS DS 539 Spring 2025, at Boston University.

## Overview
This project is about helping BU faculty determine which awards they should apply to for the highest chance. The goal is to create a model that works alongside the current excel model the Strategy & Innovation department currently uses.

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
# Step 1: Navigate to the root of the project
cd path/to/your/project

# Step 2: List the contents to make sure you're in the right place
ls
# You should see something like:
# combine_dataset/  src/  README.md

# Step 3: Change into the src folder where the scripts are
cd src

# Step 4: List the Python scripts available
ls

# Step 5: Run the desired Python script
python model1.py
# Replace 'model1.py' with the script you want to run
```

### Key Folder

- `src/` – Contains all model implementation files and utility scripts.

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

### Challenges

One of the main challenges we faced was handling the RI_Matches file, which contained nearly 300,000 rows. Combining it with the discipline dataset significantly slowed down the process and required careful handling. Additionally, we encountered inconsistency in our model results. Despite using the same input, each model yielded varying outputs, making it difficult to pinpoint the most reliable model. We were also limited by the size of the Discipline Pathways dataset, only using a portion of it due to memory and processing constraints. This likely impacted the model's ability to capture the full range of award trajectories.

### Next Steps

For any future team reading this, it would best to start with making sure the model does perform accurately. Right now, our models get about 80% accuracy, however, we are unable to validate whether this is in line with the client's model. Another next step is to create a UI interface since the project is designed for non-technical people. Having an interface would make it easier to demonstrate the results of the model.

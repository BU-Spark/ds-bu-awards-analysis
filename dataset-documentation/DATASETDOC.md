# Project Information

**What is the project name?**  
BU Awards Analysis

**What is the link to your project’s GitHub repository?**  
https://github.com/BU-Spark/ds-bu-awards-analysis/tree/main

**What is the link to your project’s Google Drive folder?**  
**This should be a Spark! Owned Google Drive folder - please contact your PM if you do not have access**  
https://drive.google.com/drive/u/1/folders/18gpUDCnC9EG4XMimIGOpqwzoJDv1zxW5

**In your own words, what is this project about? What is the goal of this project?**  
This project is about helping BU faculty determine which awards they should apply to for the highest chance. The goal is to create a model that works alongside the current excel model the Strategy & Innovation department currently uses.

**Who is the client for the project?**  
Phillip Lindsay, Director - Strategy & Innovation

**Who are the client contacts for the project?**  
Philip Lindsay - plindsay@bu.edu

**What class was this project part of?**  
DS 539

---

# Dataset Information

**What data sets did you use in your project? Please provide a link to the data sets, this could be a link to a folder in your GitHub Repo, Spark! owned Google Drive Folder for this project, or a path on the SCC, etc.**

[Dataset](https://github.com/BU-Spark/ds-bu-awards-analysis/tree/prepare_model/clean_dataset)

**Please provide a link to any data dictionaries for the datasets in this project.**  
If one does not exist, please create a data dictionary for the datasets used in this project. (Example of data dictionary)

[Data Dictionary](https://docs.google.com/document/d/1q97UhW8DeOmQDGFf5BQJ-wmBxqoAgfs3_MpOGPlSgDc/edit?usp=sharing)

---

# Code Documentation

**What keywords or tags would you attach to the data set?**  
Boston University, Award, Faculty Awards, Award Analysis, Machine Learning, Accuracy

---

## The following questions pertain to the datasets you used in your project.

### Motivation

**For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.**  
The main purpose of the dataset was to be able to get all related information about the faculty. Having an organized central place with all the data makes it easier to process all the information. There were a few IDs missing, but we were able to ignore it since it wasn’t relevant to predicting awards.

### Composition

**What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? What is the format of the instances (e.g., image data, text data, tabular data, audio data, video data, time series, graph data, geospatial data, multimodal (please specify), etc.)? Please provide a description.**  
Discipline Specific Pathways: The instances represent award pathways, specifically patterns of academic or professional recognition that individuals follow within a discipline. Each instance is centered on a focal award, and may include subsequent awards (called pathway awards) that are commonly received after it. Each record or row is one award that a faculty member has won including their pathway awards. Data types include strings, integers, and floats.  
The instances in this dataset represent individual award events, each corresponding to a specific person receiving a particular award from a governing society in a specific year. Each instance brings together the award, recipient, and award organization. You can identify the award by the award ID and name, recipient by AAUID, ORCID, name, and institution, and awarding organization by governing society. In this dataset, each row corresponds to an award recipient event or each person winning a specific award in a specific year. Data types include integers, strings, and datetime values.

**How many instances are there in total (of each type, if appropriate)?**  
In our combined dataset, there ended up being about 2,000 rows.

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).**  
Both datasets are not from a larger set and do not include all possible award events or pathways. The sample is representative of the award history since it came directly from the BU Strategy and Innovation department. This is the same dataset that they have been using when creating their own model.

**What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.**  
Awards Dataset: Each instance represents a structured record of an award event based on identifiers such as “ID” and descriptive fields such as “names, prestige, discipline”.  
Discipline-Specific Pathways Dataset: Each instance describes the relationship between a focal award and a pathway award. This includes fields such as “number of awardees, % of awardees, and mean academic age).

**Is there any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include redacted text.**  
Yes, ORCID ID’s are missing, but our client mentioned that this isn’t important and it doesn’t have an impact on our model.

**Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them**  
The majority of our team’s model was a 80:20 for train/test. We found this to work best across all of our models. We hope to validate our team’s model with the model the BU Strategy and Innovation team has. This is likely going to be a task for the future team since our client has not provided us with the information.

**Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.**  
The only errors have been dropped to allow the team to focus on creating a model. The client has stated that this is fine, but for any other teams, when combining the two datasets there are missing awards.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?**  
Yes, this dataset was given to us from the client. It is likely to change every year or when new awards are given out, so it is important to keep the model up to date with its datasets.

**Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?**  
Potentially, this would be a question for the client.

**Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer?**  
As long as VS Code is installed, there are no other restrictions.

**Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?**  
Yes, the dataset contains IDs associated with faculty and their past awards.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.**  
No, the dataset does not.

**Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.**  
Yes, like mentioned above, the ID column as well as the faculty name are in the dataset.

---

## Dataset Snapshot

**Discipline Specific Pathway:**
```
Property                         	Value
---------------------------------	------------------------------
Size of dataset                  	61.1mb
Number of instances              	~328,000
Number of fields                 	6
Labeled classes                  	No
Number of labels                 	N
```

**RI_Matches:**
```
Property                         	Value
---------------------------------	------------------------------
Size of dataset                  	548kb
Number of instances              	~2,400
Number of fields                 	13
Labeled classes                  	Yes — Award Prestige
Number of labels                 	Varies (e.g., High, Medium, Low)
```

---

# Collection Process

**What mechanisms or procedures were used to collect the data?**  
The data was given to us to use from the BU Strategy and Award Innovation department from our client Philip Lindsey. The datasets were created internally and passed along to us.

**If the dataset is a sample from a larger set, what was the sampling strategy?**  
The dataset is not a sample from a larger dataset. The dataset was given to us from the client. The original dataset is in GitHub and in our original project proposal.

**Over what timeframe was the data collected?**  
The data includes all awards faculty have won going back to the 1980s. This gives us about 2,000 award faculty entries to work with when designing our model and creating any visuals.

---

# Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**  
We dropped any null/empty values in the ID column for faculty from both the discipline_pathways and ri_matches csv files. After doing so, we then combined the two datasets based on the award name. Here is a list of columns that we dropped:  

```python
columns_to_remove = [ "awardgoverningsocietyid", "awardid", "awardreceivedid", "AAUID", "clientfacultyid", "orcid", "institutionname", "asofdate", "asofdate_iso", "is_historical", "possible_duplicate", "match_found", "matched_with", "match_type", "match_method", "similarity_score", "clean_award_name"]
```

**Were any transformations applied to the data?**  
When combining the two datasets, we faced a problem with the award names having a slight mismatch. If there was an extra space, comma, or extra filler, the script wouldn’t be able to match it. We ended up creating a script to combine the two datasets based on the award name. This was extremely time consuming since it needs to do a linear scan each time. It took nearly 24 hours for the script to run. There were about 50 or so unmatched awards.

**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data?**  
The “raw” data can be found in the original Google Doc.

**Is the code that was used to preprocess/clean the data available?**  
Yes, it can be found at our GitHub.

---

# Uses

**What tasks has the dataset been used for so far?**  
The cleaned, combined dataset has been used as our dataset for developing a model. Each of the members on the team used this dataset for their model. The combined dataset essentially combines the two datasets we were given so that we could use all of the features.

**What (other) tasks could the dataset be used for?**  
The dataset can also be used for data visualization, EDA, but the main purpose is for creating the model.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**  
Like mentioned above, combining the datasets was time consuming. So ideally, it should only be done once when the two datasets are most up to date.

**Are there tasks for which the dataset should not be used?**  
The only task the dataset shouldn’t be used is to give away information. The dataset needs to only be given to those that need to use it since we are dealing with faculty award history.

---

# Distribution

**Based on discussions with the client, what access type should this dataset be given?**  
The dataset is from the client, so only those working on the dataset should have access to it. We need to be able to read the data and open it in .csv format.

---

# Maintenance

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**  
The best way would be to update the .csv dataset and run the combine_dataset script, which can be found in our GitHub.

---

# Other

**Is there any other additional information that you would like to provide that has not already been covered in other sections?**  
No

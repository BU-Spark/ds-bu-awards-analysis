# BU Awards Analysis

This repository contains the workflow, analysis, and visualizations for the project "BU Award Analysis" for CDS DS 538 Spring 2025, at Boston University.

## Overview
This project is about helping BU faculty determine which awards they should apply to for the highest chance. The goal is to create a model that works alongside the current excel model the Strategy & Innovation department currently uses.

## Setup

### Requirements
- Python 3.9+
- Other dependencies listed in `requirements.txt`

### Dataset

## Discipline Specific Pathways - 

| Column Name                                 | Data Type | Description |
|--------------------------------------------|-----------|-------------|
| `Focal Award`                               | string    | Main award received |
| `Discipline`                                | string    | Category of award |
| `Pathway Award`                             | string    | Additional awards given after focal award |
| `Awardees w/ This Award`                    | integer   | How many people won this award |
| `Percentage of Awardees With This Award`    | float     | (Awardees with this award) / (Award nominees) — convert to float |
| `Mean Academic Age at Time of Award`        | integer   | Average academic age at time of winning the award (years in academia/research) |

## Awards Dataset Metadata

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
data/
├──
├──
└── 
```

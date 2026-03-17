# House Price Range Classification using Data Mining (Ames Dataset)

## Overview
This project applies data mining classification techniques to predict the
price range of residential homes in Ames, Iowa using the well-known Ames
Housing dataset. The original regression problem was transformed into a
classification task through discretization of the SalePrice variable into
five price bins. Four classifiers were evaluated across a two-factor
experiment design using Weka.

---

## Team
This project was completed as a group term paper.

Group Members: Amy Rettig, Chaitali Bonke, Swati Kohli,
Salman Mohammad, Tapan Patel

Course: INSY-5339-002 — Principles of Business Data Mining

> Note: This repository represents my personal documentation
> of our group project and is shared for portfolio purposes only.

---

## Problem Statement
House prices are influenced by a complex combination of structural,
locational, and qualitative factors. This project aims to build a
classification model that can accurately predict which price range a
given house falls into based on its attributes.

---

## Dataset

- **Source:** Kaggle — Ames Housing Dataset (compiled by Dean De Cock)
- **Coverage:** Residential property sales in Ames, Iowa from 2006 to 2010
- **Original size:** 1,460 instances, 80 attributes
- **Final cleaned size:** 1,460 instances, 65 attributes
- **Dataset link:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques

---

## Class Attribute — SalePrice Discretization

The original SalePrice was a continuous numeric variable ranging from
$34,900 to $755,000. Since this was not compatible with classification
algorithms, the problem was transformed from regression to classification
by applying equal height discretization to create 5 balanced price bins
averaging 292 records each:

| Bin | Price Range           | Count |
|-----|-----------------------|-------|
| 1   | $34,900 — $124,000    | 295   |
| 2   | $124,500 — $146,800   | 285   |
| 3   | $147,000 — $179,200   | 286   |
| 4   | $179,400 — $230,000   | 285   |
| 5   | $230,500 — $755,000   | 289   |

> Equal width discretization was rejected as it produced severely
> imbalanced bins (59.9% of records in the lowest bin alone).

---

## Data Cleaning

### Missing Values
- Numeric attributes MasVnrArea, LotFrontage, and GarageYrBlt had
  NA values replaced with "?" to be treated as missing numeric values

### Derived Attributes
- Total living area derived by adding basement square footage and
  above ground living area
- Four porch style attributes condensed into two: porch type (nominal)
  and porch size (numeric)

### Irrelevant Attributes Removed
14 attributes dominated by a single value were removed including
PoolQC, PoolArea, LowQualFinSF, MiscVal, MiscFeature, BsmtHalfBath,
Alley, BsmtFinSF2, Condition2, Utilities, RoofMatl, Street,
LandSlope, and Functional

### Effect of Cleaning
Average accuracy across all classifiers improved from 57.96% to 61.89%

---

## Experiment Design

### Factor 1 — Attribute Selection
- Full dataset — all attributes after cleaning
- Top 20 attributes — selected using Weka's OneRAttributeEval

### Factor 2 — Percentage Split
- 80% train / 20% test
- 30% train / 70% test

### Four Conditions
| Condition | Attributes  | Split      |
|-----------|-------------|------------|
| C1        | Full Set    | 80% / 20%  |
| C2        | Top 20      | 80% / 20%  |
| C3        | Full Set    | 30% / 70%  |
| C4        | Top 20      | 30% / 70%  |

Each condition was run 10 times with distinct seed values.
Total experiment runs: 4 conditions × 4 classifiers × 10 seeds = 160 runs

---

## Classifiers Used

| Classifier    | Description                                              |
|---------------|----------------------------------------------------------|
| OneR          | Benchmark — single rule based on best single attribute   |
| Naive Bayes   | Probabilistic classifier using Bayes rule                |
| K-Star        | Instance-based lazy learner with entropy distance metric |
| Random Forest | Ensemble of decision trees correcting for overfitting    |

---

## Top 20 Attributes by Predictive Power

| Rank | Attribute       | Merit  |
|------|-----------------|--------|
| 1    | totalLivingArea | 53.052 |
| 2    | OverallQual     | 52.534 |
| 3    | Neighborhood    | 47.633 |
| 4    | GarageArea      | 44.193 |
| 5    | GarageCars      | 43.364 |
| 6    | GrLivArea       | 42.070 |
| 7    | YearBuilt       | 40.982 |
| 8    | TotalBsmtSF     | 40.046 |
| 9    | BsmtQual        | 39.711 |
| 10   | KitchenQual     | 38.630 |

---

## Results

### Average Accuracy by Classifier (C1 — Full Set, 80/20 Split)

| Classifier    | Accuracy | Variance |
|---------------|----------|----------|
| Random Forest | 67.26%   | 0.07%    |
| Naive Bayes   | 65.10%   | 0.06%    |
| K-Star        | 61.34%   | 0.08%    |
| OneR          | 53.12%   | 0.11%    |

### Key Observations
- 80/20 split consistently outperformed 30/70 split across all classifiers
- Full attribute set outperformed Top 20 attribute set in all conditions
- Random Forest achieved the highest accuracy under all conditions
- OneR had the highest variance making it the least stable classifier
- Naive Bayes had the lowest variance making it the most stable classifier
- ROC curve analysis confirmed that All Attributes produced a larger
  area under the curve than the Top 20 set

---

## Key Findings
- The derived attribute totalLivingArea was the single strongest
  predictor of house price, ranking first in attribute selection
- Overall quality, neighborhood, and garage size were the next
  most influential factors after total living area
- Random Forest is the recommended classifier for this dataset
  due to its highest accuracy and strong stability
- Using all available attributes produced better models than
  attribute selection, suggesting that even lower-ranked attributes
  contribute meaningful signal
- Larger training sets (80/20 split) significantly improved
  prediction performance compared to smaller training sets (30/70)

---

## Reproducibility
This analysis was conducted entirely using the Weka GUI tool.
To reproduce the results:
- Download the Ames Housing dataset from Kaggle
- Apply the data cleaning steps described above
- Load the cleaned dataset into Weka
- Apply OneRAttributeEval for feature selection to identify Top 20 attributes
- Run all four classifiers: OneR, Naive Bayes, K-Star, Random Forest
- Use both 80/20 and 30/70 percentage splits
- Repeat each run 10 times with incrementally increasing random seeds
- Use the Weka Experimenter for F-score comparison
- Use the Weka Knowledge Flow for ROC curve generation

---

## Limitations
- Discretizing SalePrice into bins may lead to information loss
  compared to a direct regression approach
- Results depend on Weka configurations and random seed selection
- No external validation dataset was used beyond the train/test splits
- The dataset is limited to one city (Ames, Iowa) and may not
  generalize to other housing markets

---

## Tools Used
- Weka (classification, attribute evaluation, ROC curves)
- Weka Experimenter (classifier comparison using F-scores)
- Weka Knowledge Flow (ROC curve generation)

---

## Conclusion
Random Forest with all attributes and an 80/20 train/test split
produced the best predictive model for house price range classification,
achieving 67.26% accuracy. The most important predictors were total
living area, overall quality, neighborhood, and garage-related features.
This project demonstrates that ensemble methods with comprehensive
feature sets outperform simpler classifiers for real estate price
prediction tasks.

---

## Disclaimer
This project was completed as a group term paper for INSY-5339-002
at the University of Texas Arlington. The dataset is publicly
available on Kaggle. No original code files are included in this
repository as the analysis was conducted using the Weka GUI tool.

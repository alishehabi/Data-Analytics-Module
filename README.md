# Predicting Football/Soccer Players Transfer Market Value

## Overview
This project utilizes two machine learning techniques, **Linear Regression** and **Decision Trees**, to predict football players' transfer market values. The dataset used is derived from FIFA player statistics and contains a wide range of attributes, including player ratings, performance metrics, and other features influencing market value.

By leveraging machine learning, the project aims to provide accurate, data-driven valuations that can assist football clubs in identifying undervalued talent and making informed decisions during transfer windows.

---

## Objectives

1. **Feature Correlation**: Identify the features that most strongly correlate with transfer market value.
2. **Model Application**: Apply and compare the performance of Linear Regression and Decision Tree models.
3. **Evaluation**: Evaluate the models using performance metrics such as R² score and Mean Squared Error (MSE).
4. **Challenges & Improvements**: Document challenges faced, along with potential improvements and applications.

---

## Dataset

- **Source**: [Kaggle - Football Players Data](https://www.kaggle.com/datasets/maso0dahmed/football-players-data/data)
- **Description**: 
  - Over 17,000 entries.
  - Attributes include age, potential, skill ratings, and market value.
  - Features were pre-processed to remove irrelevant or non-numerical data, and Principal Component Analysis (PCA) was applied to reduce dimensionality.

---

## Installation

To run this project locally, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Google Colab (optional)

### Clone the Repository
```bash
git clone https://github.com/your-username/football-market-value-prediction.git
cd football-market-value-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements File Includes:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## Project Structure

```plaintext
Project Structure:
.
├── data/                          # Directory for dataset (not included in repo; download separately)
├── Code.py                        # Python script containing the core code
├── DataANALYTICS.ipynb            # Google Colab notebook exported as PDF with explanations
├── DataAnalytics_Report_200180517.pdf # Detailed report on methodology and result analysis
├── README.md                      # README file describing the project
├── dataanalytics.py               # Python script to upload on Colab for running the entire project
└── requirements.txt               # Dependencies for the project
```

---

## Usage

### Data Preprocessing
- Load and clean the dataset, removing missing values and non-numeric features.
- Perform feature selection and PCA to identify the most relevant predictors for market value.

### Training the Models
1. Run the `dataanalytics.py` script on Google Colab or any Python environment to execute the entire project.
2. Alternatively, use the `DataANALYTICS.ipynb` notebook for a more detailed explanation and walkthrough.

### Results
- **Linear Regression**:
  - R²: ~88%
  - MSE: ~0.0011
- **Decision Tree (Random Forest)**:
  - R²: ~93.8%
  - MSE: ~0.0006

Random Forest outperformed Linear Regression due to its ability to capture non-linear relationships and reduce overfitting.

---

## Key Insights
1. **Log Transformation**: Applied to the target variable (`value_euro`) to normalize skewed data.
2. **Feature Engineering**: Created aggregate features (e.g., `attack_rating`, `defence_rating`) for better prediction.
3. **Cross-Validation**: Used 5-fold cross-validation to ensure model robustness.

---

## Future Work

1. Incorporate real-time player data to refine model predictions.
2. Experiment with advanced algorithms like Gradient Boosting or Neural Networks.
3. Include additional factors such as injuries, market trends, and social media influence.

---

## References

1. [Predict the Value of Football Players Using FIFA Data](https://www.researchgate.net/publication/358871715_Predict_the_Value_of_Football_Players_Using_FIFA_Video_Game_Data_and_Machine_Learning_Techniques)
2. [Determinants of Transfers Fees: Evidence from Major European Leagues](https://www.researchgate.net/publication/331929212_Determinants_of_Transfers_Fees_Evidence_from_the_Five_Major_European_Football_Leagues)
3. Kaggle Dataset: [Football Players Data](https://www.kaggle.com/datasets/maso0dahmed/football-players-data/data)

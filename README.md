# Company-Profit-Prediction-ML
Analysis and profit prediction for the 50 Startups dataset using Python, Pandas, Scikit-learn, and Seaborn/Matplotlib..

## Description
This repository contains a Jupyter Notebook (`startup_profit_prediction.ipynb`) demonstrating how to predict the profit of startup companies based on several expenditure features and location. The project involves data preprocessing, exploratory data analysis (EDA), feature scaling, and the implementation and evaluation of two regression models: Linear Regression and a Multi-layer Perceptron (MLP) Regressor.

## Dataset
The analysis uses the `50_Startups.csv` dataset, which includes the following columns:
* `R&D Spend`: Amount spent on Research and Development.
* `Administration`: Amount spent on Administration.
* `Marketing Spend`: Amount spent on Marketing.
* `State`: The state where the startup is located (Categorical: New York, California, Florida).
* `Profit`: The resulting profit (Target Variable).

## Process
The notebook follows these key steps:
1.  **Import Libraries:** Importing necessary Python libraries (Pandas, Matplotlib, Seaborn, Scikit-learn).
2.  **Load & Prepare Data:**
    * Loading the `50_Startups.csv` dataset.
    * Randomly shuffling the dataset rows.
    * Encoding the categorical `State` column into numerical representations.
3.  **Exploratory Data Analysis (EDA):**
    * Visualizing the correlation between features using a Seaborn heatmap.
    * Analyzing and visualizing the sum and mean profit grouped by `State`.
4.  **Feature Scaling:** Standardizing the features using `StandardScaler` after splitting the data.
5.  **Train/Test Split:** Splitting the dataset into training (80%) and testing (20%) sets.
6.  **Model Training:**
    * Training a `LinearRegression` model.
    * Training an `MLPRegressor` (Neural Network) model.
7.  **Prediction & Evaluation:**
    * Making predictions on the test set using both trained models.
    * Evaluating model performance using Mean Absolute Error (MAE).
    * Calculating the error percentage for the Linear Regression model relative to the mean profit.

## Tools Used
* Python 3
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (for `train_test_split`, `LinearRegression`, `MLPRegressor`, `StandardScaler`, `metrics`)
* Jupyter Notebook

## How to Run
1.  Clone this repository:
    ```bash
    git clone (https://github.com/saamr6/Company-Profit-Prediction-ML)
    ```
2.  Navigate to the cloned directory.
3.  Ensure you have Python installed, along with the necessary libraries:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn notebook
    ```
4.  Make sure the dataset `50_Startups.csv` is present in the same directory as the notebook.
5.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
6.  Open `startup_profit_prediction.ipynb` (or your chosen filename) and run the cells.

## Results
The notebook includes the code implementation, visualizations (correlation heatmap, profit by state), and the evaluation metrics (MAE) for both the Linear Regression and MLP models. The Linear Regression model showed a Mean Absolute Error corresponding to approximately 10.43% of the average profit in the dataset.

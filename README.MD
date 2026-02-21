# NYC Taxi Trip Duration Prediction

An end-to-end Machine Learning regression project aimed at accurately predicting the total duration of taxi trips in New York City. This project utilizes spatial, temporal, and traffic-contextual feature engineering combined with hyperparameter-tuned Gradient Boosting models to solve the logistical challenge of predicting Estimated Time of Arrival (ETA).

##  Business Problem
Ride-hailing platforms and taxi dispatchers rely heavily on accurate ETAs to function effectively. Incorrect predictions lead to:
* **Poor Customer Experience:** Riders waiting longer than promised.
* **Inefficient Dispatching:** Cars sitting idle or being assigned to sub-optimal routes.
* **Lost Revenue:** Surge pricing algorithms failing to account for true time-in-transit.

**Objective:** Build a robust, highly generalizable regression model to predict the exact trip duration (in seconds) of NYC taxi rides, accounting for physical distance and complex non-linear traffic patterns.

---

##  Project Pipeline & Methodology

### 1. Data Cleaning & Preprocessing
* **Outlier Removal:** Filtered out physically impossible trip durations (e.g., multi-day meter errors) and restricted passenger counts to standard vehicle limits (1–6).
* **Geospatial Boxing:** Removed erroneous GPS coordinates by boxing the dataset within standard NYC latitude/longitude boundaries.
* **Speed Filtering:** Calculated average trip speed to remove "teleporter" anomalies (speeds > 160 km/h) and stationary data glitches.

### 2. Feature Engineering
* **Manhattan Distance (Taxicab Geometry):** Calculated the grid-based distance between pickup and dropoff coordinates using the Haversine formula, providing a much more realistic measurement of urban travel than straight-line Euclidean distance.
* **Cyclical Temporal Features:** Transformed hour, day, and month data using **Sine/Cosine transformations** to allow models to understand the continuous, circular nature of time (e.g., 23:00 is temporally adjacent to 00:00).
* **Daily Traffic Intensity:** Engineered a macro-level traffic context feature by tracking daily volume, allowing the model to detect and adjust for anomalies like blizzards or major holidays.
* **Airport Proximity:** Created boolean flags for trips originating or terminating within 2km of JFK, LaGuardia, or Newark airports.

### 3. Target Variable Transformation
* Applied a logarithmic transformation (`log1p`) to the highly right-skewed `trip_duration` target to satisfy normal distribution assumptions for baseline linear models.

---

##  Modeling Strategy

The project followed a progressive modeling approach, starting with interpretable baselines and advancing to complex ensembles.

* **Phase 1: Linear Baselines:** Trained standard `LinearRegression`, `Ridge` (L2), and `Lasso` (L1). Lasso performed the best by crushing useless features (like passenger count and vendor ID) to zero, proving that purely linear models struggle to map ETA solely based on distance without complex traffic context.
* **Phase 2: Tree-Based Ensembles:** Trained `XGBoost` and `LightGBM` regressors. These models successfully captured the non-linear "Rush Hour" effects, nearly halving the error of the linear models.
* **Phase 3: Hyperparameter Tuning:** Used `RandomizedSearchCV` with 3-fold Cross-Validation on the LightGBM model. Carefully restricted `max_depth` and `num_leaves` while applying feature sub-sampling to prevent overfitting.

---

##  Results & Evaluation

Evaluations were performed on the inverse-transformed data to provide business-interpretable metrics (actual minutes).

* **Baseline Linear Model (Lasso) RMSE:** ~12.44 minutes
* **Tuned LightGBM Model RMSE:** ~6.66 minutes
* **R-Squared ($R^2$):** ~0.70 (explaining 70% of the variance in NYC traffic)
* **Overfit Diagnostic:** The Train RMSE (6.87m) and Test RMSE (6.66m) gap was practically zero, mathematically proving the model generalized successfully to unseen data.

---

## Model Interpretability & Business Insights

To ensure the model's logic is transparent for dispatchers and stakeholders, we applied **SHAP (SHapley Additive exPlanations)** values and Feature Importance analysis:

1. **Distance is the Foundation:** Manhattan Distance is the highest magnitude driver of ETAs.
2. **Time is a Multiplier (The Rush Hour Effect):** SHAP plots revealed that cyclical time features (like the Cosine of the pickup hour) actively push the baseline ETA up or down depending on the time of day, accurately mimicking real-world traffic gridlock.
3. **Macro Traffic Matters:** Total daily trip volume (`daily_traffic_intensity`) served as a highly utilized feature in the tree splits, proving that the overall "busyness" of the city is critical for accurate forecasting.

---

##  Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost
* **Data Visualization:** Matplotlib, Seaborn, SHAP

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/nyc-taxi-duration-prediction.git](https://github.com/YourUsername/nyc-taxi-duration-prediction.git)

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Run the Jupyter Notebook `nyc_taxi_duration.ipynb` to execute the full pipeline.

*(Note: Data files should be placed in the `/data` directory. Due to size constraints, the original Kaggle dataset must be downloaded manually from the [NYC Taxi Trip Duration Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration).)*

```# nyc-reg-project

# Feature Engineering & Multicollinearity Analysis

This document provides a detailed analysis of the feature engineering step in the California Housing Price Prediction pipeline, specifically addressing its impact on multicollinearity and how it interacts with the Multi-Layer Perceptron (MLP) model.

---

## 1. Engineered Features Overview
In **Stage 09 (Feature Engineering)**, we created three domain-specific ratio features from the raw spatial and count variables to capture density and household characteristics rather than absolute scale:

1. **`rooms_per_household`** = `total_rooms / households`
   * *Rationale:* Normalizes the total rooms in a block group by the number of active households, representing the average size/room density per home.
2. **`bedrooms_per_room`** = `total_bedrooms / total_rooms`
   * *Rationale:* Measures the ratio of bedrooms to overall rooms. A high ratio usually signals smaller or denser housing units, while a low ratio indicates larger homes with more functional rooms (e.g., studies, living rooms).
3. **`population_per_household`** = `population / households`
   * *Rationale:* Represents the average number of residents per household, capturing residential crowding.

---

## 2. Multicollinearity Assessment

### The Problem in Raw Data
Multicollinearity occurs when independent variables are highly correlated with each other, which inflates the variance of model coefficients in linear models. In the raw California Housing dataset, absolute count features are highly collinear because block groups with more households naturally have more rooms, more bedrooms, and more population:

* Correlation between `total_bedrooms` and `households`: **0.97**
* Correlation between `total_rooms` and `total_bedrooms`: **0.93**
* Correlation between `population` and `households`: **0.91**

### The Solution: How Ratio Features Broke the Collinearity
By dividing these absolute counts by each other, we effectively normalized the scale factor. The engineered ratio features show significantly lower correlation with both their parent variables and each other:

| Feature 1 | Feature 2 | Correlation ($r$) | Multicollinearity Status |
|---|---|---|---|
| `rooms_per_household` | `total_rooms` | **0.15** | Broken / Resolved |
| `bedrooms_per_room` | `total_bedrooms` | **0.06** | Broken / Resolved |
| `population_per_household` | `population` | **0.07** | Broken / Resolved |
| `rooms_per_household` | `bedrooms_per_room` | **-0.38** | Low (Acceptable) |
| `rooms_per_household` | `population_per_household` | **-0.01** | Negligible |
| `bedrooms_per_room` | `population_per_household` | **0.00** | None |

*Conclusion:* The engineered features do not introduce new multicollinearity. In fact, they transform highly redundant raw metrics into independent, high-signal inputs.

---

## 3. Interaction with the MLP Model

While multicollinearity is a critical issue for Ordinary Least Squares (OLS) Linear Regression, it impacts Multi-Layer Perceptrons (MLPs) differently:

1. **No Matrix Inversion Failures:** OLS regression solves coefficients via closed-form matrix algebra $(X^T X)^{-1}$. If features are perfectly collinear, this matrix is singular (non-invertible) and the solver crashes or produces highly unstable weights. MLPs, however, optimize weights iteratively using **Gradient Descent (SGD)** via backpropagation, meaning they never calculate matrix inversions and will not fail mathematically.
2. **Hidden Layer Representations:** MLPs pass inputs through hidden layers and non-linear activation functions (like `ReLU`). The network naturally learns to combine, compress, or discount redundant inputs, creating its own robust representation of the feature space.
3. **Signal Enhancement:** Although the MLP can handle correlated raw features, providing the network with pre-calculated ratio features (like `bedrooms_per_room`) speeds up training convergence. It bypasses the need for a shallow network to mathematically discover these division-based relationships on its own.

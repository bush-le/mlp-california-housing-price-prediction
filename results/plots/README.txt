=== PLOTS DIRECTORY INVENTORY ===

This directory contains all visualizations generated during the pipeline execution.
Below is the interpretation of each plot:

1. 03_eda_histograms.png
   - Interpretation: Univariate distributions for all numerical features. Highlights skewness and scales.

2. 03_eda_ocean_proximity.png
   - Interpretation: Bar chart showing the distribution of the categorical 'ocean_proximity' feature.

3. 03_eda_correlation.png
   - Interpretation: Heatmap showing linear correlations (Pearson's r) between all numerical features. Useful for identifying multicollinearity and feature importance relative to the target.

4. 03_eda_geo_map.png
   - Interpretation: Scatter plot mapping longitude/latitude to median_house_value, showing the spatial distribution of house prices in California.

5. 05_outliers_before.png
   - Interpretation: Box plots of numerical features before outlier treatment, showing raw data spread and IQR-defined outliers.

6. 05_outliers_after.png
   - Interpretation: Box plots after treatment. Since we only removed censored target values and kept genuine feature outliers (relying on scaler bounds), the plots still show realistic variance but with the artificial $500k cap removed.

7. 08_target_distribution.png
   - Interpretation: Histogram of the target variable `median_house_value` showing its distribution and the $500,001 cap line.

8. 09_engineered_distributions.png
   - Interpretation: Histograms for newly engineered features (rooms_per_household, bedrooms_per_room, population_per_household) showing their density and spread.

9. 10_mlp_loss_curve.png
   - Interpretation: Training and validation loss curve across epochs. A decreasing trend with minimal gap between train/val indicates successful learning and good generalization without severe overfitting.

10. 12_confusion_matrix.png
    - Interpretation: A binarized confusion matrix mapping predictions above/below a threshold (e.g. $250k). Gives a classification perspective on the regression model's discriminatory power.

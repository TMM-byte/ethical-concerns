import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetricTextReport
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessor

# Step 1: Load the COMPAS dataset
compas = CompasDataset()
dataset = compas.convert_to_dataframe()

print("Dataset shape:", dataset.shape)
print("\nFirst few rows:")
print(dataset.head())

# Step 2: Analyze bias metrics
# Define protected attribute (race)
protected_attr = 'race'
favorable_label = 0  # No recidivism
unfavorable_label = 1  # Recidivism

# Calculate disparate impact
metric = BinaryLabelDatasetMetric(
    compas,
    unprivileged_groups=[{protected_attr: 1}],
    privileged_groups=[{protected_attr: 0}]
)

print("\n=== BIAS ANALYSIS ===")
print(f"Disparate Impact Ratio: {metric.disparate_impact():.4f}")
print(f"Mean Difference: {metric.mean_difference():.4f}")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")

# Step 3: Calculate false positive rates by race
df = dataset
false_positive_rate_by_race = df.groupby(protected_attr).apply(
    lambda x: ((x['two_year_recidivism'] == 0) & (x['predicted_recidivism'] == 1)).sum() / 
              (x['two_year_recidivism'] == 0).sum()
)

print("\n=== FALSE POSITIVE RATES BY RACE ===")
print(false_positive_rate_by_race)

# Step 4: Visualize disparities
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: False Positive Rates by Race
false_positive_rate_by_race.plot(kind='bar', ax=axes, color=['#1f77b4', '#ff7f0e'])
axes.set_title('False Positive Rates by Race')
axes.set_ylabel('False Positive Rate')
axes.set_xlabel('Race')
axes.set_ylim([0, 1])

# Plot 2: Recidivism Predictions by Race
recidivism_by_race = df.groupby(protected_attr)['predicted_recidivism'].mean()
recidivism_by_race.plot(kind='bar', ax=axes, color=['#2ca02c', '#d62728'])
axes.set_title('Predicted Recidivism Rate by Race')
axes.set_ylabel('Recidivism Rate')
axes.set_xlabel('Race')
axes.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('bias_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'bias_analysis.png'")

# Step 5: Apply mitigation technique (Reweighing)
print("\n=== APPLYING MITIGATION: REWEIGHING ===")
reweighing = Reweighing(
    unprivileged_groups=[{protected_attr: 1}],
    privileged_groups=[{protected_attr: 0}]
)
dataset_reweighted = reweighing.fit_transform(compas)

# Recalculate metrics after reweighing
metric_reweighted = BinaryLabelDatasetMetric(
    dataset_reweighted,
    unprivileged_groups=[{protected_attr: 1}],
    privileged_groups=[{protected_attr: 0}]
)

print(f"Disparate Impact (After Reweighing): {metric_reweighted.disparate_impact():.4f}")
print(f"Mean Difference (After Reweighing): {metric_reweighted.mean_difference():.4f}")
# Fair Top-k Ranking under Positional Fairness

Python implementation of "Fair Top-k Ranking under Positional Fairness". 
This repository provides the code for making fair rankings. 

## Example
```python
from dataset import *
from ranking import fair_top_k_ranking
from metrics import *

# Create Sample DataFrame with multiple protected groups
data, relevance_score, protected_attr = generate_dataset(n=1000,
                                                         protected_attributes={"gender": 2, "ethnicity": 3},
                                                         unfairness=0.1,
                                                         seed=42)

# Create fair top-k ranking
k = 100
z_unique = data[protected_attr].unique()
target_proportions = {key: 1 / len(z_unique) for key in z_unique} # define target proportions equally
fair_top_k_df = fair_top_k_positional_ranking(data, k, protected_attr, relevance_score)

# Get ranking metrics
metrics = get_metrics(top_k_df, data, k, target_proportions, protected_attr, relevance_score)
# >>> MPD: 0.056, NDCG_loss: 0.000, Selection_utility_loss: 0.256, Maximum_Relevance_loss: 0.000, MPoD: 0.017, kendall_tau: 1.000
```

## File Overview
- **ranking.py**: Contains the ranking algorithm.
- **metrics.py**: Contains all functions to calculate metrics for ranking evaluation.
- **dataset.py**: A set of utility functions to generate datasets

## Setup
This package relies on:
- `numpy`
- `pandas`

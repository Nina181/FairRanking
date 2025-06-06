import numpy as np
import pandas as pd


def normalization(df, relevance_score="y", protected_attr="z", alpha=1.0):
    """
    Applies min-max normalization to the "y" column within each group in "z".

    Parameters:
        df (pd.DataFrame): The input dataframe.
        relevance_score: column name of the relevance score column.
        alpha (float): Degree of normalization (0 = no normalization, 1 = full normalization).

    Returns:
        pd.DataFrame: DataFrame with a new column "y_normalized".
    """
    def normalize_with_alpha(x, alpha):
        min_x, max_x = x.min(), x.max()
        if max_x == min_x:  # Avoid division by zero
            return x
        normalized = (x - min_x) / (max_x - min_x)  # Standard min-max normalization
        return alpha * normalized + (1 - alpha) * x  # Interpolate with original values

    df[relevance_score + "_norm"] = (df.groupby(protected_attr)[relevance_score]
                                     .transform(lambda x: normalize_with_alpha(x, alpha)))
    return df

def generate_dataset(n=1000, protected_attributes={"gender": 2, "ethnicity": 3}, unfairness=0.5, seed=40):
    """
    Generates a dataset for the fair ranking problem with multiple protected attributes.

    Parameters:
        n (int): Number of candidates.
        protected_attributes (dict): A dictionary where keys are attribute names and values are the number of groups.
        unfairness (float): Degree of unfairness (0 = fair, higher values increase bias).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A dataset containing a "z" column with all protected attributes as a string,
                      and a "y" column representing the quality score.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate group assignments for each protected attribute
    groups = {attr: np.random.choice(range(num_groups), size=n) for attr, num_groups in protected_attributes.items()}

    # Create a combined group column (as a string with "_")
    df = pd.DataFrame(groups)
    df["z"] = df.astype(str).agg("_".join, axis=1)

    # Define base quality scores from a normal distribution
    base_quality = np.random.normal(loc=0.5, scale=0.15, size=n)

    # Introduce unfairness based on multiple attributes
    unfairness_effect = np.zeros(n)
    for attr, num_groups in protected_attributes.items():
        unfairness_effect += (df[attr] / (num_groups - 1)) * unfairness  # Higher groups get more advantage if u > 0

    quality_scores = np.clip(base_quality + unfairness_effect, 0, 1)  # Ensure scores stay in [0,1]

    # Assign quality score
    df["y"] = quality_scores

    # Sort by quality score (descending) for ranking
    df = df.sort_values(by="y", ascending=False).reset_index(drop=True)
    df["id"] = df.index  # Assign ranking ID
    df["y"] = np.linspace(1, 0, len(df))
    return df

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def calc_maximum_proportion_disparity(df, target_proportions, protected_attr="z"):
    """
    Calculates the Maximum Proportion Disparity for the top-k candidates. This metric is normalized to the
    maximum possible Proportion Disparity based on the target proportions.

    Parameters:
        df (pd.DataFrame): The ranking containing candidates, groups, and quality scores.
        protected_attr (str): The column of the social groups.
        target_proportions (dict): A dictionary mapping group labels to their target proportion in top-k.

    Returns:
        float: The Maximum proportion Disparity across all groups.
    """
    # Get observed proportions of groups in the top-k ranking
    group_counts = df[protected_attr].value_counts(normalize=True).to_dict()

    # Ensure all groups are accounted for, even if they have 0 representation
    observed_proportions = {group: group_counts.get(group, 0) for group in target_proportions.keys()}

    disparities = []
    for group in target_proportions:
        norm_factor = max(1 - target_proportions[group], target_proportions[group])
        disparities.append(abs(observed_proportions[group] - target_proportions[group]) / norm_factor)

    return max(disparities)


def calc_ndcg(df, k, relevance_col="y"):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) score.

    Parameters:
        df (pd.DataFrame): Dataframe containing the ranking with a quality score column.
        k (int): Number of top candidates to consider.
        relevance_col (str): Column name containing quality scores.

    Returns:
        float: NDCG score between 0 and 1.
    """
    # Compute DCG
    ranked_scores = np.array(df[relevance_col].values)
    dcg = np.sum(ranked_scores / np.log2(np.arange(1, k + 1) + 1))

    # Compute Ideal DCG (IDCG) with perfectly sorted ranking
    ideal_scores = np.sort(ranked_scores)[::-1]
    idcg = np.sum(ideal_scores / np.log2(np.arange(1, k + 1) + 1))

    # Normalize: If IDCG is 0, return 0 to avoid division by zero
    return dcg / idcg if idcg > 0 else 0.0


def calc_selection_utility_loss(df, rank_df, relevance_score="y"):
    """
    Computes the selection utility loss of a ranking.

    Parameters:
        df (pd.DataFrame): DataFrame containing candidates ranked from best to worst.
        rank_df (pd.DataFrame): Dataframe containing the top-k candidates.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: The selection utility loss of the ranking.
    """
    df_not_topk = df.loc[~df["id"].isin(rank_df["id"])]
    df_topk = df.loc[df["id"].isin(rank_df["id"])]

    max_relevance_not_topk = np.max(df_not_topk[relevance_score])
    min_relevance_topk = np.min(df_topk[relevance_score])

    if max_relevance_not_topk > min_relevance_topk:
        selection_utility = min_relevance_topk - max_relevance_not_topk
    else:
        selection_utility = 0

    selection_utility_loss = 0 - selection_utility
    return selection_utility_loss


def calc_maximum_relevance(df, rank_df, protected_attr="z", relevance_score="y"):
    """
    Computes the maximum relevance of a ranking defined as the maximum groupwise selection utility loss.

    Parameters:
        df (pd.DataFrame): DataFrame containing candidates ranked from best to worst.
        rank_df (pd.DataFrame): Dataframe containing the top-k candidates.
        protected_attr (str): The column name representing the protected group attribute.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: The maximum relevance loss of the ranking.
    """
    selection_utility_losses = []

    # calculate selection utility loss for each group
    for group in set(rank_df[protected_attr]):
        group_rank = rank_df[rank_df[protected_attr] == group]
        group_df = df[df[protected_attr] == group]

        selection_utility_loss = calc_selection_utility_loss(group_df, group_rank, relevance_score)
        selection_utility_losses.append(selection_utility_loss)

    return np.nanmax(selection_utility_losses)

def positional_fairness(rank_df, groups, protected_attr="z"):
    """
        Computes the Normalized Maximum Positional Disparity (MPoD) for a ranked list.

        The metric evaluates fairness by measuring how different groups
        are distributed across ranking positions. It applies a logarithmic
        discount to positions (i.e., higher ranks contribute more to the score)
        and calculates the disparity between the most and least advantaged groups.
        The result is normalized by the maximum possible positional score.

        The result is normalized between 0 and 1:
            - 0 indicates perfect fairness (all groups are proportionally represented across rankings).
            - 1 indicates maximum unfairness (one group dominates the top ranks, while another is pushed to the bottom).

        Parameters:
            rank_df (pd.DataFrame): A DataFrame containing ranked candidates.
            groups (list): A list of all social groups.
            protected_attr (str): The column name representing the protected group attribute.

        Returns:
            float: A normalized positional fairness score between 0 and 1.
    """
    wg = lambda x: 1/np.log2(x+1)

    pfd_scores = []
    for group in groups:
        if group in rank_df[protected_attr].unique():
            positions = rank_df[rank_df[protected_attr] == group].index.to_numpy() + 1
            pfd_scores.append(np.sum(wg(positions)))
        else:
            pfd_scores.append(0)

    mpd = max(pfd_scores) - min(pfd_scores)

    # Normalize by maximum possible Positional Disparity
    mpfd = mpd / sum(wg(np.arange(1, len(rank_df) + 1)))

    return mpfd

def calc_kendall_tau(df, rank_df, k, relevance_score="y"):
    relevance_score_orig = "_".join(relevance_score.split("_")[:-1])
    rank_orig = df.sort_values(by=relevance_score_orig, ascending=False).head(k).reset_index(drop=True)
    original_rank = rank_orig["id"].tolist()
    fair_rank = rank_df["id"].tolist()

    kendall_corr, _ = kendalltau(original_rank, fair_rank)
    return kendall_corr


def calc_jaccard(df, rank_df, k, relevance_score="y"):
    relevance_score_orig = "_".join(relevance_score.split("_")[:-1])
    rank_orig = df.sort_values(by=relevance_score_orig, ascending=False).head(k).reset_index(drop=True)

    # Identify common IDs
    common_ids = set(rank_orig['id']) & set(rank_df['id'])

    jaccard = len(common_ids) / len(set(rank_orig['id']) | set(rank_df['id']))
    return jaccard

def get_metrics(rank_df, data, k, target_proportions, protected_attr="z", relevance_score="y"):
    mpd = calc_maximum_proportion_disparity(rank_df, target_proportions, protected_attr=protected_attr)
    ndcg = calc_ndcg(rank_df, k=k, relevance_col=relevance_score)
    selection_utility_loss = calc_selection_utility_loss(data, rank_df, relevance_score=relevance_score)
    maximum_relevance_loss = calc_maximum_relevance(data, rank_df, protected_attr, relevance_score=relevance_score)
    pf_score = positional_fairness(rank_df, data[protected_attr].unique(), protected_attr=protected_attr)
    kendall_tau = calc_kendall_tau(data, rank_df, k, relevance_score=relevance_score)
    jaccard = calc_jaccard(data, rank_df, k, relevance_score=relevance_score)

    metrics = {"MPD": mpd,
               "NDCG_loss": 1 - ndcg,
               "Selection_utility_loss": selection_utility_loss,
               "Maximum_Relevance_loss": maximum_relevance_loss,
               "MPoD": pf_score,
               "kendall_tau": kendall_tau,
               "jaccard": jaccard}

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    return metrics

import numpy as np
import pandas as pd
import math

def fair_top_k_ranking(df, k, target_proportions, protected_attr="z", relevance_score="y"):
    # Create a ranking for each group separately by quality score
    df = df.sort_values(by=relevance_score, ascending=False).reset_index(drop=True)
    group_rankings = {group: df[df[protected_attr] == group][:k].to_dict('records') for group in
                      df[protected_attr].unique()}

    # Select top candidates from each group according to target proportions
    top_ranking = []
    residual_x = []
    for group, ranking in group_rankings.items():
        n_group = math.floor(target_proportions[group] * k)
        s_group = ranking[:n_group]
        top_ranking += s_group
        residual_x += [ranking[n_group]]  # store next best candidate from each group

    # Select k-best candidates using top_ranking and residual_x
    residual_x.sort(key=lambda d: d[relevance_score], reverse=True)
    top_k_ranking = top_ranking + residual_x[:k - len(top_ranking)]   # compensate rounding
    top_k_ranking.sort(key=lambda d: d[relevance_score], reverse=True)
    top_k_ranking = pd.DataFrame(top_k_ranking).reset_index(drop=True)

    return top_k_ranking

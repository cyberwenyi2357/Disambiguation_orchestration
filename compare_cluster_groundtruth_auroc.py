"""
Compare ranking consistency between Cluster.db and groundTruth.db using Pairwise Ranking AUROC
"""

import sqlite3
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm

CLUSTER_DB = "cluster_results_llama2_sorted_cleaned.db"
GROUNDTRUTH_DB = "groundTruth_llama.db"

def load_cluster_records(db_path):
    """Load all records from cluster_results_llama2_sorted.db"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source_id, ambiguous_question, n_clusters, entropy
        FROM cluster_results
        ORDER BY 
            CASE WHEN entropy IS NULL THEN 1 ELSE 0 END,
            entropy ASC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    records = {}
    for row in rows:
        source_id, ambiguous_question, n_clusters, entropy = row
        records[str(source_id)] = {
            "source_id": source_id,
            "ambiguous_question": ambiguous_question,
            "n_clusters": n_clusters,
            "entropy": entropy,
            "rank": None  # Will be calculated later
        }
    
    # Calculate rank (based on entropy, smaller values rank higher)
    sorted_records = sorted(
        records.items(),
        key=lambda x: (
            x[1]["entropy"] is None,
            x[1]["entropy"] if x[1]["entropy"] is not None else float('inf')
        )
    )
    
    for rank, (source_id, record) in enumerate(sorted_records, 1):
        records[source_id]["rank"] = rank
    
    return records

def load_groundtruth_records(db_path):
    """Load all records from groundTruth.db"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, question, qa_pairs, llm_answer, uncertainty, evaluation_score
        FROM results
        ORDER BY 
            CASE WHEN evaluation_score IS NULL THEN 1 ELSE 0 END,
            evaluation_score DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    records = {}
    for row in rows:
        record_id, question, qa_pairs_json, llm_answer, uncertainty, evaluation_score = row
        records[str(record_id)] = {
            "id": record_id,
            "question": question,
            "qa_pairs": qa_pairs_json,
            "llm_answer": llm_answer,
            "uncertainty": uncertainty,
            "evaluation_score": evaluation_score,
            "rank": None  # Will be calculated later
        }
    
    # Calculate rank (based on evaluation_score, larger values rank higher)
    sorted_records = sorted(
        records.items(),
        key=lambda x: (
            x[1]["evaluation_score"] is None,
            -x[1]["evaluation_score"] if x[1]["evaluation_score"] is not None else float('inf')
        )
    )
    
    for rank, (record_id, record) in enumerate(sorted_records, 1):
        records[record_id]["rank"] = rank
    
    return records

def get_common_ids(cluster_records, groundtruth_records):
    """Get common source_ids from both databases"""
    cluster_ids = set(cluster_records.keys())
    groundtruth_ids = set(groundtruth_records.keys())
    common_ids = list(cluster_ids & groundtruth_ids)
    return common_ids

def compute_pairwise_auroc(cluster_records, groundtruth_records, common_ids):
    """
    Compute pairwise ranking AUROC
    
    For each pair of records (i, j):
    - If evaluation_score of i > evaluation_score of j in groundtruth, then label = 1
    - If evaluation_score of i < evaluation_score of j in groundtruth, then label = 0
    - Use entropy difference as prediction score (note: smaller entropy is better, so use -entropy)
    """
    print("Generating pairs and computing labels...")
    
    # Generate all possible pairs
    pairs = list(combinations(common_ids, 2))
    print(f"Total pairs: {len(pairs)}\n")
    
    labels = []
    scores = []
    valid_pairs = 0
    
    for id1, id2 in tqdm(pairs, desc="Processing pairs"):
        gt1 = groundtruth_records[id1]
        gt2 = groundtruth_records[id2]
        cluster1 = cluster_records[id1]
        cluster2 = cluster_records[id2]
        
        # Check if evaluation_scores exist and are not equal
        if (gt1["evaluation_score"] is not None and 
            gt2["evaluation_score"] is not None and
            gt1["evaluation_score"] != gt2["evaluation_score"]):
            
            # Label: 1 if gt1 > gt2, 0 if gt1 < gt2
            label = 1 if gt1["evaluation_score"] > gt2["evaluation_score"] else 0
            
            # Score: Use entropy difference as score
            # If cluster1's entropy < cluster2's entropy, then cluster1 is better
            # So score = cluster2["entropy"] - cluster1["entropy"]
            if (cluster1["entropy"] is not None and cluster2["entropy"] is not None):
                # Smaller entropy is better, so if cluster1's entropy is smaller, score should be positive
                score = cluster2["entropy"] - cluster1["entropy"]
                
                labels.append(label)
                scores.append(score)
                valid_pairs += 1
    
    print(f"\nValid pairs (with both evaluation_score and entropy): {valid_pairs}")
    
    if valid_pairs == 0:
        print("Error: No valid pairs found!")
        return None, None, None
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Compute AUROC
    # Note: If all labels are the same, AUROC cannot be computed
    if len(np.unique(labels)) < 2:
        print("Warning: All labels are the same. Cannot compute AUROC.")
        return None, labels, scores
    
    auroc = roc_auc_score(labels, scores)
    
    return auroc, labels, scores

def analyze_rankings(cluster_records, groundtruth_records, common_ids):
    """Analyze ranking consistency"""
    print("=" * 80)
    print("Ranking Analysis")
    print("=" * 80 + "\n")
    
    # Sort by evaluation_score (groundtruth's ranking)
    sorted_by_gt = sorted(
        common_ids,
        key=lambda x: (
            groundtruth_records[x]["evaluation_score"] is None,
            -groundtruth_records[x]["evaluation_score"] if groundtruth_records[x]["evaluation_score"] is not None else 0
        )
    )
    
    # Sort by entropy (cluster's ranking)
    sorted_by_cluster = sorted(
        common_ids,
        key=lambda x: (
            cluster_records[x]["entropy"] is None,
            cluster_records[x]["entropy"] if cluster_records[x]["entropy"] is not None else float('inf')
        )
    )
    
    # Compute Spearman correlation coefficient
    from scipy.stats import spearmanr
    
    # Get ranks
    gt_ranks = {id: rank for rank, id in enumerate(sorted_by_gt, 1)}
    cluster_ranks = {id: rank for rank, id in enumerate(sorted_by_cluster, 1)}
    
    # Only consider records with both evaluation_score and entropy
    valid_ids = [
        id for id in common_ids
        if (groundtruth_records[id]["evaluation_score"] is not None and
            cluster_records[id]["entropy"] is not None)
    ]
    
    if len(valid_ids) < 2:
        print("Not enough valid records for correlation analysis.")
        return None
    
    gt_rank_values = [gt_ranks[id] for id in valid_ids]
    cluster_rank_values = [cluster_ranks[id] for id in valid_ids]
    
    spearman_corr, spearman_p = spearmanr(gt_rank_values, cluster_rank_values)
    
    print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Number of valid records: {len(valid_ids)}\n")
    
    return spearman_corr

def main():
    print("=" * 80)
    print("Pairwise Ranking AUROC Comparison")
    print("Cluster.db (sorted by entropy) vs groundTruth.db (sorted by evaluation_score)")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading Cluster.db...")
    cluster_records = load_cluster_records(CLUSTER_DB)
    print(f"Loaded {len(cluster_records)} records from Cluster.db\n")
    
    print("Loading groundTruth.db...")
    groundtruth_records = load_groundtruth_records(GROUNDTRUTH_DB)
    print(f"Loaded {len(groundtruth_records)} records from groundTruth.db\n")
    
    # Get common IDs
    common_ids = get_common_ids(cluster_records, groundtruth_records)
    print(f"Common records: {len(common_ids)}\n")
    
    if len(common_ids) < 2:
        print("Error: Need at least 2 common records to compute AUROC!")
        return
    
    # Analyze rankings
    spearman_corr = analyze_rankings(cluster_records, groundtruth_records, common_ids)
    
    # Compute pairwise AUROC
    print("=" * 80)
    print("Computing Pairwise Ranking AUROC")
    print("=" * 80 + "\n")
    
    auroc, labels, scores = compute_pairwise_auroc(
        cluster_records, groundtruth_records, common_ids
    )
    
    if auroc is not None:
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Pairwise Ranking AUROC: {auroc:.4f}")
        print(f"\nInterpretation:")
        print(f"  - AUROC = 1.0: Perfect agreement (entropy ranking perfectly matches evaluation_score ranking)")
        print(f"  - AUROC = 0.5: Random agreement (no correlation)")
        print(f"  - AUROC < 0.5: Negative correlation (opposite ranking)")
        print("=" * 80)
        
        if spearman_corr is not None:
            print(f"\nSpearman Rank Correlation: {spearman_corr:.4f}")
            print(f"  (Higher correlation indicates better agreement between rankings)")
    else:
        print("\nCould not compute AUROC (insufficient data or all labels are the same).")

if __name__ == "__main__":
    main()


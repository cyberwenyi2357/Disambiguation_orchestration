"""
Compare Uncertainty.db and groundTruth.db using Pairwise Ranking AUROC
Records with lower uncertainty should have higher evaluation_score
"""

import sqlite3
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm

GROUNDTRUTH_DB = "groundTruth_llama.db"
UNCERTAINTY_DB = "Uncertainty_llama.db"

def load_records_from_db(db_path):
    """Load all records from database, return dictionary {id: record}"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, question, qa_pairs, llm_answer, uncertainty, evaluation_score, created_at
        FROM results
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    records = {}
    for row in rows:
        records[row[0]] = {
            "id": row[0],
            "question": row[1],
            "qa_pairs": row[2],
            "llm_answer": row[3],
            "uncertainty": row[4],
            "evaluation_score": row[5],
            "created_at": row[6],
        }
    
    return records

def get_common_ids(groundtruth_records, uncertainty_records):
    """Get common IDs from both databases"""
    gt_ids = set(groundtruth_records.keys())
    uncertainty_ids = set(uncertainty_records.keys())
    common_ids = list(gt_ids & uncertainty_ids)
    return common_ids

def compute_pairwise_auroc(groundtruth_records, uncertainty_records, common_ids):
    """
    Compute Pairwise Ranking AUROC
    
    For each pair of records (i, j):
    - If evaluation_score of i > evaluation_score of j in groundtruth, then label = 1
    - If evaluation_score of i < evaluation_score of j in groundtruth, then label = 0
    - Use uncertainty difference as prediction score (lower uncertainty is better, so use uncertainty difference)
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
        unc1 = uncertainty_records[id1]
        unc2 = uncertainty_records[id2]
        
        # Check if evaluation_scores exist and are not equal
        if (gt1["evaluation_score"] is not None and 
            gt2["evaluation_score"] is not None and
            gt1["evaluation_score"] != gt2["evaluation_score"]):
            
            # Label: 1 if gt1 > gt2, 0 if gt1 < gt2
            label = 1 if gt1["evaluation_score"] > gt2["evaluation_score"] else 0
            
            # Score: Use uncertainty difference as score
            # If unc1 < unc2, then unc1 is better, score should be positive
            # So score = unc2["uncertainty"] - unc1["uncertainty"]
            if (unc1["uncertainty"] is not None and unc2["uncertainty"] is not None):
                # Lower uncertainty is better, so if unc1's uncertainty is smaller, score should be positive
                score = unc2["uncertainty"] - unc1["uncertainty"]
                
                labels.append(label)
                scores.append(score)
                valid_pairs += 1
    
    print(f"\nValid pairs (with both evaluation_score and uncertainty): {valid_pairs}")
    
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

def analyze_rankings(groundtruth_records, uncertainty_records, common_ids):
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
    
    # Sort by uncertainty (uncertainty's ranking)
    sorted_by_unc = sorted(
        common_ids,
        key=lambda x: (
            uncertainty_records[x]["uncertainty"] is None,
            uncertainty_records[x]["uncertainty"] if uncertainty_records[x]["uncertainty"] is not None else float('inf')
        )
    )
    
    # Compute Spearman correlation coefficient
    from scipy.stats import spearmanr
    
    # Get ranks
    gt_ranks = {id: rank for rank, id in enumerate(sorted_by_gt)}
    unc_ranks = {id: rank for rank, id in enumerate(sorted_by_unc)}
    
    # Only consider records with both evaluation_score and uncertainty
    valid_ids = [
        id for id in common_ids
        if (groundtruth_records[id]["evaluation_score"] is not None and
            uncertainty_records[id]["uncertainty"] is not None)
    ]
    
    if len(valid_ids) < 2:
        print("Not enough valid records for correlation analysis.")
        return None
    
    gt_rank_values = [gt_ranks[id] for id in valid_ids]
    unc_rank_values = [unc_ranks[id] for id in valid_ids]
    
    spearman_corr, spearman_p = spearmanr(gt_rank_values, unc_rank_values)
    
    print(f"Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Number of valid records: {len(valid_ids)}\n")
    
    return spearman_corr

def main():
    print("=" * 80)
    print("Pairwise Ranking AUROC Comparison")
    print("Uncertainty.db vs groundTruth.db")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading groundTruth.db...")
    groundtruth_records = load_records_from_db(GROUNDTRUTH_DB)
    print(f"Loaded {len(groundtruth_records)} records from groundTruth.db\n")
    
    print("Loading Uncertainty.db...")
    uncertainty_records = load_records_from_db(UNCERTAINTY_DB)
    print(f"Loaded {len(uncertainty_records)} records from Uncertainty.db\n")
    
    # Get common IDs
    common_ids = get_common_ids(groundtruth_records, uncertainty_records)
    print(f"Common records: {len(common_ids)}\n")
    
    if len(common_ids) < 2:
        print("Error: Need at least 2 common records to compute AUROC!")
        return
    
    # Analyze rankings
    spearman_corr = analyze_rankings(groundtruth_records, uncertainty_records, common_ids)
    
    # Compute pairwise ranking AUROC
    print("=" * 80)
    print("Computing Pairwise Ranking AUROC")
    print("=" * 80 + "\n")
    
    auroc, labels, scores = compute_pairwise_auroc(
        groundtruth_records, uncertainty_records, common_ids
    )
    
    if auroc is not None:
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Pairwise Ranking AUROC: {auroc:.4f}")
        print(f"\nInterpretation:")
        print(f"  - AUROC = 1.0: Perfect agreement (uncertainty ranking perfectly matches evaluation_score ranking)")
        print(f"  - AUROC = 0.5: Random agreement (no correlation)")
        print(f"  - AUROC < 0.5: Negative correlation (opposite ranking)")
        print(f"\nMethod:")
        print(f"  - For each pair of records (i, j):")
        print(f"    - Label = 1 if evaluation_score_i > evaluation_score_j")
        print(f"    - Label = 0 if evaluation_score_i < evaluation_score_j")
        print(f"    - Score = uncertainty_j - uncertainty_i (lower uncertainty = better)")
        print("=" * 80)
        
        if spearman_corr is not None:
            print(f"\nSpearman Rank Correlation: {spearman_corr:.4f}")
    else:
        print("\nCould not compute AUROC (insufficient data or all labels are the same).")

if __name__ == "__main__":
    main()


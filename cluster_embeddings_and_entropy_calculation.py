"""
Use HDBSCAN to cluster text embeddings for each question
"""

import sqlite3
import json
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any
from math import log

DB_PATH = "clarification_texts_llama2_cleaned.db"
OUTPUT_DB_PATH = "cluster_results_llama2_cleaned.db"
MIN_CLUSTER_SIZE = 2
# HDBSCAN parameters
USE_COSINE_DISTANCE = True  # Use cosine distance (via precomputed distance matrix)
MIN_SAMPLES = 1  # Lowering this value allows smaller clusters to be identified
CLUSTER_SELECTION_EPSILON = 0.0  # Can adjust this value to merge closer clusters

def get_texts_with_embeddings_by_source_id(source_id: str, db_path: str = DB_PATH):
    """
    Get all concatenated texts and their corresponding embeddings for specified source_id.
    Returns format: [{"text_id": 1, "search_result_idx": 1, "concatenated_text": "...", "embedding": [...]}, ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            id AS text_id,
            search_result_idx,
            concatenated_text,
            embedding
        FROM clarifications
        WHERE source_id = ?
        ORDER BY search_result_idx ASC, id ASC
        """,
        (source_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        text_id, search_result_idx, concatenated_text, embedding_blob = row
        # Parse embedding BLOB (stored as JSON string)
        embedding = None
        if embedding_blob:
            try:
                # BLOB could be JSON string or binary data
                if isinstance(embedding_blob, bytes):
                    embedding_str = embedding_blob.decode('utf-8')
                else:
                    embedding_str = embedding_blob
                embedding = json.loads(embedding_str)
            except:
                embedding = None
        
        result.append({
            "text_id": text_id,
            "search_result_idx": search_result_idx,
            "concatenated_text": concatenated_text,
            "embedding": embedding,
        })
    return result

def create_output_database(db_path: str = OUTPUT_DB_PATH):
    """Create output database to store clustering results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            ambiguous_question TEXT NOT NULL,
            n_clusters INTEGER NOT NULL,
            cluster_sizes TEXT NOT NULL,
            entropy REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Created output database: {db_path}")

def calculate_entropy(cluster_sizes: Dict[int, int]) -> float:
    """
    Calculate entropy: H = -Σ p_i * log(p_i)
    where p_i = cluster_size / total_size
    
    Args:
        cluster_sizes: Dictionary, {cluster_id: cluster_size}
    
    Returns:
        entropy value
    """
    if not cluster_sizes:
        return 0.0
    
    total_size = sum(cluster_sizes.values())
    if total_size == 0:
        return 0.0
    
    entropy = 0.0
    for cluster_size in cluster_sizes.values():
        if cluster_size > 0:
            p_i = cluster_size / total_size
            entropy -= p_i * log(p_i)
    
    return entropy

def get_all_source_ids(db_path: str = DB_PATH) -> List[str]:
    """Get all unique source_ids"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source_id FROM clarifications WHERE source_id IS NOT NULL ORDER BY source_id ASC")
    source_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return source_ids

def cluster_source_embeddings(source_id: str, db_path: str = DB_PATH, min_cluster_size: int = MIN_CLUSTER_SIZE):
    """
    Cluster embeddings for specified source_id and save results to database.
    
    Returns:
        dict: Clustering statistics
    """
    # Get all texts and embeddings for this source_id
    texts_data = get_texts_with_embeddings_by_source_id(source_id, db_path)
    
    if not texts_data:
        print(f"  Source ID {source_id}: No texts found, skipping")
        return None
    
    # Get ambiguous_question (from first record)
    ambiguous_question = texts_data[0].get('concatenated_text', '')[:100]  # Temporary use, will get from database later
    
    # Get ambiguous_question from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ambiguous_question FROM clarifications WHERE source_id = ? LIMIT 1", (source_id,))
    row = cursor.fetchone()
    if row:
        ambiguous_question = row[0]
    conn.close()
    
    # Filter texts that have embeddings
    valid_texts = [item for item in texts_data if item['embedding'] is not None]
    
    if len(valid_texts) < min_cluster_size:
        print(f"  Source ID {source_id}: Only {len(valid_texts)} texts with embeddings, skipping clustering (need at least {min_cluster_size})")
        return None
    
    # Extract embeddings and corresponding text_ids
    embeddings = np.array([item['embedding'] for item in valid_texts])
    text_ids = [item['text_id'] for item in valid_texts]
    
    # Perform HDBSCAN clustering
    if USE_COSINE_DISTANCE:
        # Method 1: Use precomputed cosine distance matrix
        # This bypasses sklearn BallTree's limitation of not supporting cosine
        distance_matrix = cosine_distances(embeddings)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=MIN_SAMPLES,
            metric='precomputed',  # Use precomputed distance matrix
            cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
    else:
        # Method 2: Normalize then use euclidean (equivalent to cosine)
        from sklearn.preprocessing import normalize
        embeddings_normalized = normalize(embeddings, norm='l2')
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=MIN_SAMPLES,
            metric='euclidean',
            cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        )
        cluster_labels = clusterer.fit_predict(embeddings_normalized)
    
    # Statistics
    unique_labels = set(cluster_labels)
    n_clusters_original = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 represents noise points
    n_noise = list(cluster_labels).count(-1)
    
    # Calculate cluster size distribution
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[label] = list(cluster_labels).count(label)
    
    # Treat noise points as a new cluster
    # If noise points exist, assign them to a new cluster ID (using n_clusters_original as the new cluster_id)
    if n_noise > 0:
        # Noise points form a new cluster, cluster_id = n_clusters_original
        cluster_sizes[n_clusters_original] = n_noise
        # Update cluster_labels: replace -1 with n_clusters_original
        cluster_labels = np.where(cluster_labels == -1, n_clusters_original, cluster_labels)
        n_clusters = n_clusters_original + 1  # Total clusters = original clusters + 1 (noise point cluster)
    else:
        n_clusters = n_clusters_original  # No noise points, total clusters = original clusters
    
    # Calculate entropy
    entropy = calculate_entropy(cluster_sizes)
    
    # Convert cluster_sizes to JSON string for storage
    # Need to convert numpy.int64 type keys to Python int
    cluster_sizes_python = {int(k): int(v) for k, v in cluster_sizes.items()}
    cluster_sizes_json = json.dumps(cluster_sizes_python, sort_keys=True)
    
    # Save results to output database
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO cluster_results 
        (source_id, ambiguous_question, n_clusters, cluster_sizes, entropy)
        VALUES (?, ?, ?, ?, ?)
    """, (source_id, ambiguous_question, n_clusters, cluster_sizes_json, entropy))
    
    conn.commit()
    conn.close()
    
    return {
        "source_id": source_id,
        "ambiguous_question": ambiguous_question,
        "total_texts": len(texts_data),
        "texts_with_embedding": len(valid_texts),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_labels": cluster_labels.tolist(),
        "cluster_sizes": cluster_sizes,
        "entropy": entropy,
    }

def main():
    print("=" * 80)
    print("HDBSCAN Clustering for Source ID Embeddings")
    print("=" * 80 + "\n")
    print(f"Configuration:")
    print(f"  Min cluster size: {MIN_CLUSTER_SIZE}")
    print(f"  Min samples: {MIN_SAMPLES}")
    print(f"  Distance metric: {'cosine (precomputed)' if USE_COSINE_DISTANCE else 'euclidean (on normalized embeddings)'}")
    print(f"  Cluster selection epsilon: {CLUSTER_SELECTION_EPSILON}\n")
    
    # Create output database
    print("Creating output database...")
    create_output_database(OUTPUT_DB_PATH)
    print()
    
    # Get all source_ids
    source_ids = get_all_source_ids(DB_PATH)
    print(f"Found {len(source_ids)} unique source_ids in database.\n")
    
    # Cluster each source_id
    print("Starting clustering...\n")
    results = []
    
    for i, source_id in enumerate(source_ids, 1):
        print(f"[{i}/{len(source_ids)}] Processing source_id {source_id}...", end=" ")
        result = cluster_source_embeddings(source_id, DB_PATH, MIN_CLUSTER_SIZE)
        
        if result:
            cluster_info = f"{result['n_clusters']} clusters"
            if result['n_clusters'] > 0:
                # Display cluster sizes (sorted by cluster_id)
                sizes = [result.get('cluster_sizes', {}).get(i, 0) for i in sorted(result.get('cluster_sizes', {}).keys())]
                cluster_info += f" (sizes: {sizes})"
            # Note: Noise points are now also included in clusters, so display original noise point count as reference
            if result.get('n_noise', 0) > 0:
                cluster_info += f" (including {result['n_noise']} noise points as a cluster)"
            # Display entropy
            entropy = result.get('entropy', 0.0)
            cluster_info += f", entropy: {entropy:.4f}"
            print(f"✓ {cluster_info}")
            results.append(result)
        else:
            print("✗ Skipped")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Clustering Summary:")
    print("=" * 80)
    print(f"Total source_ids processed: {len(source_ids)}")
    print(f"Source_ids successfully clustered: {len(results)}")
    print(f"Source_ids skipped: {len(source_ids) - len(results)}")
    
    if results:
        total_clusters = sum(r['n_clusters'] for r in results)
        total_noise = sum(r['n_noise'] for r in results)
        avg_clusters = total_clusters / len(results)
        avg_noise = total_noise / len(results)
        
        print(f"\nTotal clusters found: {total_clusters}")
        print(f"Total noise points: {total_noise}")
        print(f"Average clusters per source_id: {avg_clusters:.2f}")
        print(f"Average noise points per source_id: {avg_noise:.2f}")
    
    print("\n" + "=" * 80)
    print(f"Done! Cluster results have been saved to {OUTPUT_DB_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()


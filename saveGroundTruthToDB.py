import sqlite3
import json
import os

SOURCE_DB = "baseline_results_llama.db"
TARGET_DB = "groundTruth_llama.db"

def create_target_database(db_path):
    """Create target database with same table structure as source database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            qa_pairs TEXT NOT NULL,
            llm_answer TEXT,
            uncertainty REAL,
            evaluation_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Created target database: {db_path}")

def get_all_records_sorted(db_path):
    """Get all records, sorted by evaluation_score descending (NULL values last)"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Sort by evaluation_score descending (higher evaluation_score is better)
    # NULL values last
    cursor.execute("""
        SELECT id, question, qa_pairs, llm_answer, uncertainty, evaluation_score, created_at
        FROM results
        ORDER BY 
            CASE WHEN evaluation_score IS NULL THEN 1 ELSE 0 END,
            evaluation_score DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            "id": row[0],
            "question": row[1],
            "qa_pairs": row[2],
            "llm_answer": row[3],
            "uncertainty": row[4],
            "evaluation_score": row[5],
            "created_at": row[6],
        })
    
    return records

def save_records_to_database(records, db_path):
    """Save records to target database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for record in records:
        cursor.execute("""
            INSERT INTO results (id, question, qa_pairs, llm_answer, uncertainty, evaluation_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record["id"],
            record["question"],
            record["qa_pairs"],
            record["llm_answer"],
            record["uncertainty"],
            record["evaluation_score"],
            record["created_at"],
        ))
    
    conn.commit()
    conn.close()

def extract_sorted_by_evaluation_score():
    """Sort all records by evaluation_score descending and store them"""
    print("=" * 80)
    print("Sorting Records by Evaluation Score (Descending)")
    print("=" * 80 + "\n")
    
    # Check if source database exists
    if not os.path.exists(SOURCE_DB):
        print(f"Error: Source database '{SOURCE_DB}' not found!")
        return
    
    # Get all records (already sorted by evaluation_score descending)
    print(f"Loading records from {SOURCE_DB}...")
    all_records = get_all_records_sorted(SOURCE_DB)
    total_count = len(all_records)
    print(f"Total records: {total_count}\n")
    
    if total_count == 0:
        print("No records found in source database.")
        return
    
    # Display statistics
    print("Statistics of sorted records:")
    print("-" * 80)
    evaluation_scores = [r["evaluation_score"] for r in all_records if r["evaluation_score"] is not None]
    if evaluation_scores:
        print(f"  Evaluation Score Range: {min(evaluation_scores)} - {max(evaluation_scores)}")
        print(f"  Average Evaluation Score: {sum(evaluation_scores) / len(evaluation_scores):.2f}")
        print(f"  Records with NULL evaluation_score: {sum(1 for r in all_records if r['evaluation_score'] is None)}")
        print(f"  Highest Evaluation Score (Top 5):")
        for i, record in enumerate(all_records[:5], 1):
            if record["evaluation_score"] is not None:
                print(f"    {i}. ID: {record['id']}, Evaluation Score: {record['evaluation_score']}")
    else:
        print("  No evaluation scores found.")
    print("-" * 80 + "\n")
    
    # Create target database
    if os.path.exists(TARGET_DB):
        print(f"Warning: Target database '{TARGET_DB}' already exists. It will be overwritten.")
        os.remove(TARGET_DB)
    
    create_target_database(TARGET_DB)
    
    # Save all records (already sorted)
    print(f"Saving {len(all_records)} records to {TARGET_DB}...")
    save_records_to_database(all_records, TARGET_DB)
    
    print("\n" + "=" * 80)
    print("✓ Successfully sorted and saved records by evaluation_score!")
    print(f"  Source: {SOURCE_DB} ({total_count} records)")
    print(f"  Target: {TARGET_DB} ({len(all_records)} records, sorted by evaluation_score DESC)")
    print("=" * 80)

if __name__ == "__main__":
    extract_sorted_by_evaluation_score()


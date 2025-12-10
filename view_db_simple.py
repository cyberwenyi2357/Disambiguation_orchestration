import sqlite3
import json

DB_PATH = "baseline_results_llama.db"

def view_database(db_path, show_full_answer=False):
    """View all contents in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all records
    cursor.execute("SELECT id, question, llm_answer, uncertainty, evaluation_score, created_at FROM results ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    if not rows:
        print("No data in database.")
        conn.close()
        return
    
    print(f"\nTotal records in database: {len(rows)}\n")
    print("="*100)
    
    for idx, row in enumerate(rows, 1):
        record_id, question, llm_answer, uncertainty, evaluation_score, created_at = row
        print(f"\nRecord {idx}:")
        print(f"  ID: {record_id}")
        print(f"  Question: {question}")
        if show_full_answer:
            print(f"  LLM Answer: {llm_answer if llm_answer else 'None'}")
        else:
            print(f"  LLM Answer: {llm_answer[:200] if llm_answer else 'None'}...")
            if llm_answer and len(llm_answer) > 200:
                print(f"  (Truncated, use --full to see complete answer)")
        uncertainty_str = f"{uncertainty:.6f}" if uncertainty is not None else "None"
        print(f"  Uncertainty: {uncertainty_str}")
        evaluation_score_str = str(evaluation_score) if evaluation_score is not None else "None"
        print(f"  Evaluation Score: {evaluation_score_str}")
        print(f"  Created At: {created_at}")
        print("-"*100)
    
    conn.close()

def view_detailed_record(db_path, record_id=None):
    """View detailed information of a single record"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if record_id and record_id.lower() == "latest":
        # Get the latest record
        cursor.execute("SELECT * FROM results ORDER BY created_at DESC LIMIT 1")
    elif record_id and record_id.lower() == "first":
        # Get the first record (sorted by created_at ascending)
        cursor.execute("SELECT * FROM results ORDER BY created_at ASC LIMIT 1")
    elif record_id:
        # Find by ID
        cursor.execute("SELECT * FROM results WHERE id = ?", (str(record_id),))
    else:
        # Default: get the latest record
        cursor.execute("SELECT * FROM results ORDER BY created_at DESC LIMIT 1")
    
    row = cursor.fetchone()
    
    if not row:
        print("No record found.")
        conn.close()
        return
    
    column_names = [description[0] for description in cursor.description]
    
    print("\n" + "="*100)
    print("Detailed Record Information:")
    print("="*100 + "\n")
    
    for i, col_name in enumerate(column_names):
        value = row[i]
        print(f"{col_name}:")
        
        if col_name == "qa_pairs" and value:
            try:
                qa_pairs = json.loads(value)
                print(json.dumps(qa_pairs, ensure_ascii=False, indent=2))
            except:
                print(f"  {value}")
        elif col_name == "llm_answer" and value:
            print(f"  {value}")
        else:
            print(f"  {value}")
        print()
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg == "--full" or first_arg == "-f":
            # View all records (full answers)
            view_database(DB_PATH, show_full_answer=True)
        elif first_arg == "--help" or first_arg == "-h":
            # Display help information
            print("\nUsage:")
            print("  python view_db_simple.py                    # View all records (truncated answers)")
            print("  python view_db_simple.py --full            # View all records (full answers)")
            print("  python view_db_simple.py first              # View first record details")
            print("  python view_db_simple.py latest            # View latest record details")
            print("  python view_db_simple.py <id>              # View specific record by ID")
            print()
        else:
            # View detailed record for specified ID
            record_id = first_arg
            view_detailed_record(DB_PATH, record_id)
    else:
        # View all records (default: truncated)
        view_database(DB_PATH, show_full_answer=False)
        
        print("\n" + "-"*100)
        print("Usage:")
        print("  - View all records (truncated): python view_db_simple.py")
        print("  - View all records (full): python view_db_simple.py --full")
        print("  - View first record details: python view_db_simple.py first")
        print("  - View latest record details: python view_db_simple.py latest")
        print("  - View specific record by ID: python view_db_simple.py <id>")
        print("-"*100)

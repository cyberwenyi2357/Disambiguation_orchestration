import sqlite3
import json
from typing import List, Dict, Any

DB_PATH = "clarification_texts_llama.db"

def get_total_questions_count(db_path: str = DB_PATH) -> int:
    """Return total number of records in ambiguous_questions table (i.e., how many questions)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ambiguous_questions")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def find_empty_concatenated_texts(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Find all records where concatenated_text is empty or contains only whitespace.
    Returns format: [{"text_id": 1, "question_id": 1, "source_id": "...", "question": "...", "text": ""}, ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT
            ct.id AS text_id,
            cq.id AS question_id,
            cq.source_id,
            cq.ambiguous_question,
            ct.concatenated_text,
            ct.search_result_idx
        FROM clarification_texts AS ct
        JOIN ambiguous_questions AS cq
            ON ct.question_id = cq.id
        WHERE TRIM(COALESCE(ct.concatenated_text, '')) = ''
        ORDER BY cq.id ASC, ct.id ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        text_id, question_id, source_id, ambiguous_question, concatenated_text, search_result_idx = row
        result.append({
            "text_id": text_id,
            "question_id": question_id,
            "source_id": source_id,
            "ambiguous_question": ambiguous_question,
            "concatenated_text": concatenated_text,
            "search_result_idx": search_result_idx,
        })
    return result

def get_texts_count_per_question(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Count how many concatenated texts each question has, return list sorted by count in descending order.
    Returns format: [{"question_id": 1, "count": 10, "source_id": "...", "question": "..."}, ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            cq.id AS question_id,
            cq.source_id,
            cq.ambiguous_question,
            COUNT(ct.id) AS text_count
        FROM ambiguous_questions AS cq
        LEFT JOIN clarification_texts AS ct
            ON ct.question_id = cq.id
        GROUP BY cq.id, cq.source_id, cq.ambiguous_question
        ORDER BY text_count DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        question_id, source_id, ambiguous_question, text_count = row
        result.append({
            "question_id": question_id,
            "source_id": source_id,
            "ambiguous_question": ambiguous_question,
            "count": text_count
        })
    return result

def get_question_by_index(index: int):
    """index starts from 1: 1 means the first question, 101 means the 101st question."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, source_id, ambiguous_question
        FROM ambiguous_questions
        ORDER BY id ASC
        LIMIT 1 OFFSET ?
        """,
        (index - 1,),
    )
    row = cursor.fetchone()
    conn.close()
    return row  # (question_id, source_id, ambiguous_question) or None
def get_concatenated_texts_by_question_id(question_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ct.concatenated_text
        FROM ambiguous_questions AS cq
        JOIN clarification_texts AS ct
            ON ct.question_id = cq.id
        WHERE cq.id = ?
        ORDER BY ct.search_result_idx ASC, ct.id ASC
        """,
        (question_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_texts_with_clusters_by_question_id(question_id: int):
    """
    Get all concatenated texts and their corresponding cluster_id for specified question.
    Returns format: [{"text_id": 1, "search_result_idx": 1, "concatenated_text": "...", "cluster_id": 0}, ...]
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            ct.id AS text_id,
            ct.search_result_idx,
            ct.concatenated_text,
            ct.cluster_id
        FROM ambiguous_questions AS cq
        JOIN clarification_texts AS ct
            ON ct.question_id = cq.id
        WHERE cq.id = ?
        ORDER BY ct.cluster_id ASC NULLS LAST, ct.search_result_idx ASC, ct.id ASC
        """,
        (question_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        text_id, search_result_idx, concatenated_text, cluster_id = row
        result.append({
            "text_id": text_id,
            "search_result_idx": search_result_idx,
            "concatenated_text": concatenated_text,
            "cluster_id": cluster_id,
        })
    return result
def fetch_all_rows(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Read all data from database and return as list of dictionaries.

    Current structure has two tables:
      - ambiguous_questions(id, source_id, ambiguous_question)
      - clarification_texts(id, question_id, search_result_idx, concatenated_text, embedding)

    Here we do a JOIN to read each concatenated_text along with its corresponding ambiguous question.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            cq.id AS question_id,
            cq.source_id,
            cq.ambiguous_question,
            ct.id AS text_id,
            ct.search_result_idx,
            ct.concatenated_text,
            ct.embedding
        FROM ambiguous_questions AS cq
        JOIN clarification_texts AS ct
            ON ct.question_id = cq.id
        ORDER BY cq.id ASC, ct.search_result_idx ASC, ct.id ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        (
            question_id,
            source_id,
            ambiguous_question,
            text_id,
            search_result_idx,
            concatenated_text,
            embedding,
        ) = row
        result.append(
            {
                "question_id": question_id,
                "text_id": text_id,
                "source_id": source_id,
                "search_result_idx": search_result_idx,
                "ambiguous_question": ambiguous_question,
                "concatenated_text": concatenated_text,
                "embedding": embedding,
            }
        )
    return result


def main():
    try:
        # First count total questions
        total_questions = get_total_questions_count(DB_PATH)
        print(f"Total questions in database: {total_questions}\n")
        
        rows = fetch_all_rows(DB_PATH)
    except sqlite3.OperationalError as e:
        print(f"Failed to read database {DB_PATH}: {e}")
        return

    if not rows:
        print("No data in clarification_texts table.")
        return

    print(f"Total concatenated texts: {len(rows)}\n")
    
    # Statistics of texts count distribution per question
    texts_per_question = get_texts_count_per_question(DB_PATH)
    print(f"Statistics of texts per question:")
    print(f"  Questions with more than 10 texts: {sum(1 for item in texts_per_question if item['count'] > 10)}")
    print(f"  Questions with exactly 10 texts: {sum(1 for item in texts_per_question if item['count'] == 10)}")
    print(f"  Questions with less than 10 texts: {sum(1 for item in texts_per_question if item['count'] < 10)}")
    print(f"  Questions with 0 texts: {sum(1 for item in texts_per_question if item['count'] == 0)}")
    print(f"  Max texts per question: {max(item['count'] for item in texts_per_question) if texts_per_question else 0}")
    print(f"  Average texts per question: {sum(item['count'] for item in texts_per_question) / len(texts_per_question) if texts_per_question else 0:.2f}\n")
    
    # Check if there are empty concatenated texts
    empty_texts = find_empty_concatenated_texts(DB_PATH)
    if empty_texts:
        print(f"⚠️  Found {len(empty_texts)} empty concatenated texts:")
        for item in empty_texts[:10]:  # Only show first 10
            print(f"  Text ID {item['text_id']} (Question ID {item['question_id']}, Search Result {item['search_result_idx']})")
            print(f"    Question: {item['ambiguous_question'][:60]}...")
        if len(empty_texts) > 10:
            print(f"  ... and {len(empty_texts) - 10} more empty texts")
        print()
    else:
        print("✓ No empty concatenated texts found.\n")
    
    # Display questions with more than 10 texts
    over_10 = [item for item in texts_per_question if item['count'] > 10]
    if over_10:
        print(f"Questions with more than 10 texts (showing first 10):")
        for item in over_10[:10]:
            print(f"  Question ID {item['question_id']}: {item['count']} texts")
            print(f"    Question: {item['ambiguous_question'][:80]}...")
        print()

    # Only print first 10 different questions (grouped by question_id)
    current_qid = None
    shown_questions = set()
    for row in rows:
        qid = row["question_id"]
        if qid not in shown_questions:
            if len(shown_questions) >= 0:
                break
            # New question header
            shown_questions.add(qid)
            print("=" * 80)
            print(f"Question ID: {row['question_id']}")
            print(f"Source ID : {row['source_id']}")
            print(f"Ambiguous Question:\n  {row['ambiguous_question']}\n")

        print(f"  Text ID: {row['text_id']}")
        print(f"  Search Result Index: {row['search_result_idx']}")
        print(f"  Concatenated Text:\n    {row['concatenated_text']}")
        print(f"  Embedding is None? {row['embedding'] is None}")
        print("-" * 80)

    # Also export as JSON file for use in Colab
    with open("clarification_texts_export.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nExported all rows to clarification_texts_export.json")


if __name__ == "__main__":
    main()
    
    # Display ambiguous question and all concatenated texts for specified question
    question_id = 1

    row = get_question_by_index(question_id)
    if row is None:
        print(f"Question ID {question_id} not found.")
    else:
        qid, source_id, question = row
        texts = get_concatenated_texts_by_question_id(qid)
        
        print("\n" + "=" * 80)
        print(f"Question ID: {qid}")
        print(f"Source ID: {source_id}")
        print(f"Ambiguous Question: {question}")
        print(f"\nTotal concatenated texts: {len(texts)}\n")
        print("-" * 80)
        
        for i, t in enumerate(texts, 1):
            print(f"{i}. {t}\n")



"""
Read ambiguous_question, qa_pairs, and llm_answer from database,
then call GPT-4o-mini to score the llm_answer and update back to database
"""

import os
import json
import re
import requests
import sqlite3
from dotenv import load_dotenv
from tqdm import tqdm

DB_PATH = "baseline_results_llama.db"

def call_gpt4o_mini(prompt):
    """Call GPT-4o-mini API"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Please set OPENAI_API_KEY in .env file!")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")

def evaluate_answer(question, qa_pairs, llm_answer):
    """Use GPT-4o-mini to score the LLM answer"""
    interpretations_text = "\n".join([
        f"{idx + 1}. Question: {qa.get('question', '')}\n   Answer: {', '.join(qa.get('answer', []))}" 
        for idx, qa in enumerate(qa_pairs)
        if qa.get('question')
    ])
    
    # Use string formatting instead of f-string to avoid errors from curly braces in llm_answer
    evaluation_prompt = """You are an evaluator. Given an ambiguous question, its possible interpretations with their answers, and an LLM's answer, evaluate whether the LLM answer covers all interpretations.

Ambiguous Question: {question}

Possible Interpretations:
{interpretations_text}

LLM Answer:
{llm_answer}

Please evaluate the LLM answer and provide score (1-5):

 **Coverage Score (1-5)**: How well does the LLM answer cover all the possible interpretations?
   - 5: The answer comprehensively addresses all interpretations
   - 4: The answer addresses most interpretations well
   - 3: The answer addresses some interpretations but misses others
   - 2: The answer only addresses a small portion of interpretations
   - 1: The answer barely addresses any interpretations or does not address any interpretations

   Please just give me the score (1-5), no other text.
""".format(
        question=question,
        interpretations_text=interpretations_text,
        llm_answer=llm_answer
    )
    
    # Print the evaluation prompt sent to LLM
    print("Evaluation Prompt:")
    print("-" * 60)
    print(evaluation_prompt)
    print("-" * 60 + "\n")
    
    try:
        evaluation_result = call_gpt4o_mini(evaluation_prompt)
        return evaluation_result
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None

def load_records_from_db(db_path):
    """Load all records from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, question, qa_pairs, llm_answer, evaluation_score
        FROM results
        WHERE llm_answer IS NOT NULL AND llm_answer != ''
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        record_id, question, qa_pairs_json, llm_answer, evaluation_score = row
        
        # Parse qa_pairs JSON
        try:
            qa_pairs = json.loads(qa_pairs_json) if qa_pairs_json else []
        except json.JSONDecodeError:
            print(f"Warning: Could not parse qa_pairs for ID {record_id}")
            qa_pairs = []
        
        records.append({
            "id": record_id,
            "question": question,
            "qa_pairs": qa_pairs,
            "llm_answer": llm_answer,
            "evaluation_score": evaluation_score
        })
    
    return records

def update_evaluation_score(db_path, record_id, evaluation_score):
    """Update evaluation_score in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE results
        SET evaluation_score = ?
        WHERE id = ?
    """, (evaluation_score, record_id))
    
    conn.commit()
    conn.close()

def extract_score_from_result(evaluation_result):
    """Extract score from evaluation_result"""
    evaluation_score = None
    try:
        # Try to parse JSON
        # Extract coverage_score from JSON
        json_match = re.search(r'"coverage_score"\s*:\s*(\d+)', evaluation_result)
        if json_match:
            evaluation_score = int(json_match.group(1))
        else:
            # If not JSON, try to extract the first number (1-5)
            score_match = re.search(r'\b([1-5])\b', evaluation_result)
            if score_match:
                evaluation_score = int(score_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse evaluation score: {str(e)}")
        evaluation_score = None
    
    return evaluation_score

def main():
    print("=" * 80)
    print("Evaluating Existing Answers")
    print("=" * 80 + "\n")
    
    # Load database records
    print(f"Loading records from {DB_PATH}...")
    records = load_records_from_db(DB_PATH)
    print(f"Loaded {len(records)} records with llm_answer\n")
    
    if not records:
        print("No records found with llm_answer. Exiting.")
        return
    
    # Process each record
    evaluated_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, record in enumerate(tqdm(records, desc="Evaluating answers"), start=1):
        record_id = record["id"]
        question = record["question"]
        qa_pairs = record["qa_pairs"]
        llm_answer = record["llm_answer"]
        existing_score = record["evaluation_score"]
        
        # If evaluation_score already exists, can choose to skip or re-evaluate
        # Here we choose to skip records with existing scores, if re-evaluation is needed, comment out the following lines
        if existing_score is not None:
            skipped_count += 1
            continue
        
        print(f"\n[{idx}/{len(records)}] Processing record ID: {record_id}")
        print(f"Question: {question}\n")
        
        if not qa_pairs:
            print("Warning: No qa_pairs found. Skipping.\n")
            skipped_count += 1
            continue
        
        try:
            # Call GPT-4o-mini for scoring
            print("=" * 60)
            print("Evaluating answer with GPT-4o-mini...")
            print("=" * 60 + "\n")
            
            evaluation_result = evaluate_answer(question, qa_pairs, llm_answer)
            
            if evaluation_result:
                print("Evaluation Result:")
                print(evaluation_result)
                print("\n" + "=" * 60 + "\n")
                
                # Extract score
                evaluation_score = extract_score_from_result(evaluation_result)
                
                if evaluation_score is not None:
                    # Update database
                    update_evaluation_score(DB_PATH, record_id, evaluation_score)
                    print(f"Updated evaluation_score: {evaluation_score}\n")
                    evaluated_count += 1
                else:
                    print("Warning: Could not extract evaluation score. Skipping update.\n")
                    error_count += 1
            else:
                print("Warning: Evaluation returned None. Skipping.\n")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing record {record_id}: {str(e)}\n")
            error_count += 1
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total records: {len(records)}")
    print(f"Evaluated: {evaluated_count}")
    print(f"Skipped (already have score): {skipped_count}")
    print(f"Errors: {error_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()


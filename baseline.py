import os
import json
import re
import requests
import sqlite3
from dotenv import load_dotenv
from datetime import datetime

DATASET_PATH = "train_light.json"
# Modify this value to control the number of questions to process; set to None to process all
MAX_REQUESTS = 500
DB_PATH = "baseline_results.db"

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

def call_gpt35_turbo(prompt, return_logprobs=True):
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
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    # Add logprobs parameter to get log probability for each token
    if return_logprobs:
        data["logprobs"] = True  # Return logprob for each generated token
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        choice = response_data["choices"][0]
        
        result = {
            "content": choice["message"]["content"],
            "logprobs": choice.get("logprobs")  # Contains logprobs information for each token
        }
        return result
    else:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")

def init_database(db_path):
    """Initialize database, create table structure"""
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
    
    # If table exists but doesn't have evaluation_score column, add it
    cursor.execute("PRAGMA table_info(results)")
    columns = [column[1] for column in cursor.fetchall()]
    if "evaluation_score" not in columns:
        cursor.execute("ALTER TABLE results ADD COLUMN evaluation_score INTEGER")
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}\n")

def save_to_database(db_path, entry_id, question, qa_pairs, llm_answer, uncertainty, evaluation_score=None):
    """Save results to database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert qa_pairs to JSON string
    qa_pairs_json = json.dumps(qa_pairs, ensure_ascii=False)
    
    cursor.execute("""
        INSERT OR REPLACE INTO results 
        (id, question, qa_pairs, llm_answer, uncertainty, evaluation_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (str(entry_id), question, qa_pairs_json, llm_answer, uncertainty, evaluation_score))
    
    conn.commit()
    conn.close()
    print(f"Data saved to database (ID: {entry_id})\n")

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

def load_multipleqa_entries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data:
        ambiguous_question = (item.get("question") or "").strip()
        if not ambiguous_question:
            continue
        annotations = item.get("annotations", [])
        for ann in annotations:
            if ann.get("type") != "multipleQAs":
                continue
            qa_pairs = ann.get("qaPairs", [])
            interpretations = [
                qa.get("question").strip()
                for qa in qa_pairs
                if qa.get("question")
            ]
            if interpretations:
                entries.append(
                    {
                        "id": item.get("id"),
                        "question": ambiguous_question,
                        "interpretations": interpretations,
                        "qa_pairs": qa_pairs,  # Save complete qa_pairs
                    }
                )
    return entries

if __name__ == "__main__":
    # Initialize database
    init_database(DB_PATH)
    
    try:
        entries = load_multipleqa_entries(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Cannot find data file {DATASET_PATH}!")
        exit(1)

    if not entries:
        print("No multipleQAs items found in data.")
        exit(1)

    total = len(entries)
    limit = MAX_REQUESTS if isinstance(MAX_REQUESTS, int) and MAX_REQUESTS > 0 else total
    print(f"Found {total} multipleQAs questions, processing first {limit}.\n")

    for idx, entry in enumerate(entries, start=1):
        if idx > limit:
            break
        print(f"[{idx}] Processing question (ID: {entry['id']}): {entry['question']}\n")
        
        ambiguous_question = entry["question"]
        qa_pairs = entry.get("qa_pairs", [])
        interpretations = entry.get("interpretations", [])
        prompt = (
            "please answer the following question asked by a user \n"
            f"Question: {ambiguous_question}\n"
        )
        
        llm_answer = None
        uncertainty = None
        evaluation_score = None
        
        try:
            result = call_gpt35_turbo(prompt)
            llm_answer = result["content"]
            print("\nGPT-3.5-turbo response:")
            print(llm_answer)
            
            # Calculate uncertainty
            if result.get("logprobs"):
                logprobs_data = result["logprobs"]
                # Extract logprob value for each token from content list
                content_list = logprobs_data.get("content", [])
                
                if content_list:
                    # Extract logprob values for all tokens
                    token_logprobs = [
                        item.get("logprob") 
                        for item in content_list 
                        if item.get("logprob") is not None
                    ]
                    
                    if token_logprobs:
                        # Calculate sum
                        LL_sum = sum(token_logprobs)
                        # Calculate average
                        LL_avg = LL_sum / len(token_logprobs)
                        # Calculate uncertainty
                        uncertainty = -LL_avg
                        
                        print(f"\n{'='*60}")
                        print(f"Token count: {len(token_logprobs)}")
                        print(f"Log-likelihood sum (LL_sum): {LL_sum:.6f}")
                        print(f"Log-likelihood average (LL_avg): {LL_avg:.6f}")
                        print(f"Uncertainty: {uncertainty:.6f}")
                        print(f"{'='*60}\n")
                    else:
                        print("Warning: No valid logprob values\n")
                else:
                    print("Warning: No content data in logprobs\n")
            else:
                print("Warning: No logprobs data\n")
            
            # Use GPT-4o-mini to score the answer
            if llm_answer and qa_pairs:
                print("\n" + "="*60)
                print("Evaluating answer with GPT-4o-mini...")
                print("="*60 + "\n")
                evaluation_result = evaluate_answer(ambiguous_question, qa_pairs, llm_answer)
                if evaluation_result:
                    print("Evaluation Result:")
                    print(evaluation_result)
                    print("\n" + "="*60 + "\n")
                    
                    # Extract score from evaluation_result (could be JSON or plain number)
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
            
            # Save to database
            save_to_database(
                DB_PATH,
                entry["id"],
                ambiguous_question,
                qa_pairs,
                llm_answer,
                uncertainty,
                evaluation_score
            )
        
        except Exception as e:
            print(f"Error: {str(e)}\n")
            # Try to save even if failed (uncertainty and evaluation_score are None)
            save_to_database(
                DB_PATH,
                entry["id"],
                ambiguous_question,
                qa_pairs,
                None,
                None,
                None
            )

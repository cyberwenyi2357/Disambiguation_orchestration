import requests
import json
from dotenv import load_dotenv
import os
import sqlite3

DATASET_PATH = "train_light.json"
# Total number of ambiguous questions to process (optional, used with start/end index)
MAX_REQUESTS = 1000  # Set to None to not limit by count, only by start/end index

# According to order in entries, from which ambiguous question to start and end (1-based)
START_INDEX = 77   # Inclusive
END_INDEX = 1000    # Inclusive

# Local database for storing ambiguous questions and their concatenated Interpretation+Answer texts
TEXT_DB_PATH = "clarification_texts.db"


def init_text_database(db_path: str = TEXT_DB_PATH):
    """
    Create (if not exists) a SQLite database with two tables:
      - ambiguous_questions: One row per ambiguous question
      - clarification_texts: Multiple concatenated 'interpretation + answer' texts for each question
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If there was an old structure before, you can choose to DROP old tables here if needed.
    # Currently we directly create new tables (if not exists).

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ambiguous_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            ambiguous_question TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clarification_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            search_result_idx INTEGER,
            concatenated_text TEXT NOT NULL,
            embedding TEXT,
            FOREIGN KEY(question_id) REFERENCES ambiguous_questions(id)
        )
        """
    )

    conn.commit()
    conn.close()


def save_concatenated_text(
    ambiguous_question: str,
    concatenated_text: str,
    source_id: str = None,
    search_result_idx: int = None,
    db_path: str = TEXT_DB_PATH,
):
    """
    Attach a concatenated text under the corresponding ambiguous question:
      - ambiguous_questions: one record per ambiguous question
      - clarification_texts: can have multiple concatenated_text entries for the same question_id
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First find or create ambiguous_questions record
    cursor.execute(
        """
        SELECT id FROM ambiguous_questions
        WHERE ambiguous_question = ? AND (source_id = ? OR (? IS NULL AND source_id IS NULL))
        LIMIT 1
        """,
        (ambiguous_question, str(source_id) if source_id is not None else None,
         str(source_id) if source_id is not None else None),
    )
    row = cursor.fetchone()

    if row:
        question_id = row[0]
    else:
        cursor.execute(
            """
            INSERT INTO ambiguous_questions (source_id, ambiguous_question)
            VALUES (?, ?)
            """,
            (str(source_id) if source_id is not None else None, ambiguous_question),
        )
        question_id = cursor.lastrowid

    # Then insert clarification_texts record
    cursor.execute(
        """
        INSERT INTO clarification_texts
        (question_id, search_result_idx, concatenated_text, embedding)
        VALUES (?, ?, ?, NULL)
        """,
        (question_id, search_result_idx, concatenated_text),
    )

    conn.commit()
    conn.close()

def extract_interpretations_and_answers(gpt_text):
    """
    Extract Interpretation and Answer from GPT response, return as ordered list [(interpretation, answer), ...]
    Expected format similar to:
        Interpretation: ...
        Answer: ...
    Or with numbering:
        Interpretation 1: ...
        Answer 1: ...
    """
    pairs = []
    current_interpretation = None

    lines = [line.strip() for line in gpt_text.splitlines() if line.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("interpretation"):
            # Get content after colon
            parts = line.split(":", 1)
            current_interpretation = parts[1].strip() if len(parts) > 1 else ""
        elif lower.startswith("answer") and current_interpretation is not None:
            parts = line.split(":", 1)
            answer = parts[1].strip() if len(parts) > 1 else ""
            pairs.append((current_interpretation, answer))
            current_interpretation = None

    return pairs

def call_google_search_api(query, num_results=10):
    """Call Google Custom Search API to search for questions"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key:
        raise Exception("Please set GOOGLE_API_KEY in .env file!")
    if not search_engine_id:
        raise Exception("Please set GOOGLE_SEARCH_ENGINE_ID in .env file!")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": num_results
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return results
    else:
        raise Exception(f"Google Search API returned error: {response.status_code}: {response.text}")

def call_gpt4o_api(prompt):
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
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")


def relax_question(ambiguous_question):
    """Use GPT to relax ambiguous question, generating a broader search query"""
    relax_prompt = f"""Given an ambiguous question, relax it to turn it into a suitable query for web search engines. Please do not add restrictions or extra details. Return only the relaxed query text, nothing else.

Original question: {ambiguous_question}

Relaxed query:"""
    
    try:
        print("Relax prompt sent to LLM:")
        print("-" * 60)
        print(relax_prompt)
        print("-" * 60 + "\n")
        relaxed_query = call_gpt4o_api(relax_prompt).strip()
        # Clean possible extra text (only take first line, remove possible explanations)
        relaxed_query = relaxed_query.split('\n')[0].strip()
        # Remove possible quotes
        relaxed_query = relaxed_query.strip('"').strip("'")
        return relaxed_query
    except Exception as e:
        print(f"Relax failed, using original question: {str(e)}")
        return ambiguous_question

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
                    }
                )
    return entries

def build_prompt(ambiguous_question, passage):
    """Build prompt for a single search result"""
    intro = """Given an ambiguous query and one of the passages from retrieval results, provide a disambiguated query which can be answered by the passage. Try to infer the user's intent with the ambiguous query and think of possible concrete, non-ambiguous rewritten questions. If you cannot find any of
them, which can be answered by the provided document, simply abstain by replying with 'null'. You should provide at most one subquestion, the most relevant one you can think of.
Here are the rules to follow when generating the question and answer:
1. The generated question must be a disambiguation of the original ambiguous query.
2. The question should be fully answerable from information present in given passage. Even
if the passage is relevant to the original ambiguous query, if it is not self-contained, abstain by
responding with 'null'.
3. Make sure the question is clear and unambiguous, while clarifying the intent of the original
ambiguous question.
4. Phrases like 'based on the provided context', 'according to the passage', etc., are not allowed to
appear in the question. Similarly, questions such as "What is not mentioned about something in the
passage?" are not acceptable.
5. When addressing questions tied to a specific moment, provide the clearest possible time
reference. Avoid ambiguous questions such as "Which country has won the most recent World
Cup?" since the answer varies depending on when the question is asked.
6. The answer must be specifically based on the information provided in the passage. Your prior
knowledge should not intervene in answering the identified clarification question.
Input fields are:Question: {ambiguous question (q)}Passage: {passage (p)}
Output fields are:Interpretation: {generated interpretation (ˆq)}Answer: {generated answer (ˆy)}""".strip()
    
    prompt = [intro]
    prompt.append(f"\nQuestion: {ambiguous_question}")
    prompt.append(f"\nPassage: {passage}")
    prompt.append("\n")
    return "\n".join(prompt)

if __name__ == "__main__":
    # Initialize local text database
    init_text_database(TEXT_DB_PATH)
    try:
        entries = load_multipleqa_entries(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Cannot find data file {DATASET_PATH}!")
        exit(1)

    if not entries:
        print("No matching multipleQAs items found in data.")
        exit(1)

    total = len(entries)
    # Calculate actual start and end indices to use
    start_idx = max(1, START_INDEX)
    end_idx = min(END_INDEX if END_INDEX is not None else total, total)

    # If MAX_REQUESTS is also set, truncate end_idx by count
    if isinstance(MAX_REQUESTS, int) and MAX_REQUESTS > 0:
        end_idx = min(end_idx, start_idx + MAX_REQUESTS - 1)

    print(f"Found {total} multipleQAs questions, will process questions {start_idx} to {end_idx}.\n")

    for idx, entry in enumerate(entries, start=1):
        # Filter by start/end index
        if idx < start_idx:
            continue
        if idx > end_idx:
            break
        print(f"[{idx}] Processing question (ID: {entry['id']}): {entry['question']}\n")
        
        # First use GPT to relax the question
        print("Using GPT to relax the question, generating broader search query...\n")
        relaxed_query = relax_question(entry["question"])
        print(f"Original question: {entry['question']}")
        print(f"Relaxed query: {relaxed_query}\n")
        
        # Use relaxed query to call Google Search API
        print("Calling Google Search API, please wait...\n")
        try:
            search_results = call_google_search_api(relaxed_query)
            print(f"Found {len(search_results)} search results\n")
        except Exception as e:
            print(f"Google Search call failed: {str(e)}\n")
            search_results = []
        
        # Call GPT separately for each search result
        if search_results:
            for result_idx, result in enumerate(search_results, 1):
                print(f"\nProcessing search result {result_idx}/{len(search_results)}:")
                print(f"  Title: {result['title']}")
                print(f"  Snippet: {result['snippet']}")
                print(f"  Link: {result['link']}\n")
                
                # Build passage (using title and snippet)
                passage = f"{result['title']}\n{result['snippet']}"
                
                # Build prompt for current search result
                prompt = build_prompt(entry["question"], passage)
                
                print("Calling GPT-3.5-turbo, please wait...\n")
                try:
                    gpt_result = call_gpt4o_api(prompt)
                    print("GPT-3.5-turbo response:\n")
                    print(gpt_result)

                    # Extract Interpretation and Answer from response and concatenate
                    ia_pairs = extract_interpretations_and_answers(gpt_result)
                    if ia_pairs:
                        concatenated = " || ".join(
                            # Directly concatenate into one sentence: <interpretation> <answer>
                            [f"{interp} {ans}" for interp, ans in ia_pairs]
                        )
                        print("\nConcatenated Interpretation + Answer (labels removed):")
                        print(concatenated)

                        # Save ambiguous question and concatenated text to local SQLite database
                        save_concatenated_text(
                            ambiguous_question=entry["question"],
                            concatenated_text=concatenated,
                            source_id=entry.get("id"),
                            search_result_idx=result_idx,
                            db_path=TEXT_DB_PATH,
                        )

                    print("\n" + "-" * 60 + "\n")
                except Exception as e:
                    print(f"Call failed: {str(e)}\n")
        else:
            print("No search results, skipping GPT call\n")
        
        print("=" * 60 + "\n")

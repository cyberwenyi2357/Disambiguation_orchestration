import requests
import json
from dotenv import load_dotenv
import os

DATASET_PATH = "train_light.json"
# 修改该值可以控制要处理的问题数量；设为 None 表示处理全部
MAX_REQUESTS = 1

def call_google_search_api(query, num_results=5):
    """调用 Google Custom Search API 搜索问题"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key:
        raise Exception("请在 .env 文件中设置 GOOGLE_API_KEY！")
    if not search_engine_id:
        raise Exception("请在 .env 文件中设置 GOOGLE_SEARCH_ENGINE_ID！")
    
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
        raise Exception(f"Google Search API 返回错误: {response.status_code}: {response.text}")

def call_gpt4o_api(prompt):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("请在 .env 文件中设置 OPENAI_API_KEY！")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o",
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
    """使用 GPT 对 ambiguous question 进行 relax，生成更宽泛的搜索查询"""
    relax_prompt = f"""Given an ambiguous question, relax it to turn it into a suitable query for web search engines. Please do not add restrictions or extra details. Return only the relaxed query text, nothing else.

Original question: {ambiguous_question}

Relaxed query:"""
    
    try:
        print("发送给 LLM 的 relax prompt:")
        print("-" * 60)
        print(relax_prompt)
        print("-" * 60 + "\n")
        relaxed_query = call_gpt4o_api(relax_prompt).strip()
        # 清理可能的额外文本（只取第一行，去除可能的解释）
        relaxed_query = relaxed_query.split('\n')[0].strip()
        # 移除可能的引号
        relaxed_query = relaxed_query.strip('"').strip("'")
        return relaxed_query
    except Exception as e:
        print(f"Relax 失败，使用原始问题: {str(e)}")
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
    """为单个搜索结果构建 prompt"""
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
    try:
        entries = load_multipleqa_entries(DATASET_PATH)
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {DATASET_PATH}！")
        exit(1)

    if not entries:
        print("数据中没有符合条件的 multipleQAs 项目。")
        exit(1)

    total = len(entries)
    limit = MAX_REQUESTS if isinstance(MAX_REQUESTS, int) and MAX_REQUESTS > 0 else total
    print(f"共找到 {total} 个 multipleQAs 问题，准备处理前 {limit} 个。\n")

    for idx, entry in enumerate(entries, start=1):
        if idx > limit:
            break
        print(f"[{idx}] 处理问题（ID: {entry['id']}）: {entry['question']}\n")
        
        # 先使用 GPT 对问题进行 relax
        print("正在使用 GPT 对问题进行 relax，生成更宽泛的搜索查询...\n")
        relaxed_query = relax_question(entry["question"])
        print(f"原始问题: {entry['question']}")
        print(f"Relaxed 查询: {relaxed_query}\n")
        
        # 使用 relaxed query 调用 Google Search API
        print("正在调用 Google Search API，请稍候...\n")
        try:
            search_results = call_google_search_api(relaxed_query)
            print(f"找到 {len(search_results)} 个搜索结果\n")
        except Exception as e:
            print(f"Google Search 调用失败：{str(e)}\n")
            search_results = []
        
        # 对每个搜索结果分别调用 GPT
        if search_results:
            for result_idx, result in enumerate(search_results, 1):
                print(f"\n处理搜索结果 {result_idx}/{len(search_results)}:")
                print(f"  标题: {result['title']}")
                print(f"  摘要: {result['snippet']}")
                print(f"  链接: {result['link']}\n")
                
                # 构建 passage（使用标题和摘要）
                passage = f"{result['title']}\n{result['snippet']}"
                
                # 为当前搜索结果构建 prompt
                prompt = build_prompt(entry["question"], passage)
                
                print("发送给 LLM 的 prompt:")
                print("-" * 60)
                print(prompt)
                print("-" * 60 + "\n")
                
                print("正在调用GPT-4o，请稍候...\n")
                try:
                    gpt_result = call_gpt4o_api(prompt)
                    print("GPT-4o的回复：\n")
                    print(gpt_result)
                    print("\n" + "-" * 60 + "\n")
                except Exception as e:
                    print(f"调用失败：{str(e)}\n")
        else:
            print("没有搜索结果，跳过 GPT 调用\n")
        
        print("=" * 60 + "\n")

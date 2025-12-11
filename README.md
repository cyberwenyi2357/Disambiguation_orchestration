## Search and then Clarify: An Agentic Method for Deciding Whether to Ask the User Clarifying Question

This is the codebase for Search and then Clarify, an agentic method for deciding whether to ask the user clarifying question.

## Overview

The five-step pipeline:

1. **Query Relaxation** to expand the search space
2. **Web Search** to generate grounded interpretations of an ambiguous question
3. **Generating Interpretations** to identify different interpretations of ambiguous questions
4. **Interpretations Clustering** to merge semantically equivalent or highly similar interpretations
5. **Calculating entropy** to quantify how ambiguous the question is

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Google Custom Search API key and Search Engine ID

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Trustworthy NLP project"
   ```
   
2. **Install dependencies**
   ```bash
   pip install requests python-dotenv sqlite3 numpy scikit-learn scipy tqdm hdbscan sentence-transformers torch transformers
   ```
   
3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
   ```

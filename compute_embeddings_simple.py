"""
Simplified Colab script: Assumes model is already loaded as 'model' variable
Directly compute embeddings and update database
"""

import sqlite3
import json
from tqdm import tqdm

# Configuration
DB_PATH = "clarification_texts.db"
BATCH_SIZE = 32  # Batch processing size, can be adjusted based on GPU memory

def get_texts_without_embedding(db_path: str = DB_PATH):
    """
    Get all concatenated texts that don't have embeddings yet.
    Returns: [(text_id, concatenated_text), ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, concatenated_text
        FROM clarification_texts
        WHERE embedding IS NULL OR TRIM(embedding) = ''
        ORDER BY id ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    
    return rows

def update_embeddings_batch(model, text_ids, texts, db_path: str = DB_PATH):
    """
    Batch compute embeddings and update to database.
    
    Args:
        model: Loaded SentenceTransformer model
        text_ids: List of text IDs
        texts: List of concatenated texts
        db_path: Database path
    """
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
    if not valid_indices:
        return 0
    
    valid_text_ids = [text_ids[i] for i in valid_indices]
    valid_texts = [texts[i] for i in valid_indices]
    
    # Compute embeddings - use more stable parameters, process one by one to avoid caching issues
    embeddings_list = []
    try:
        # Method 1: Try batch processing (if fails, process one by one)
        try:
            embeddings = model.encode(
                valid_texts,
                prompt_name="query",  # gte-Qwen2 recommends this for query side
                batch_size=min(BATCH_SIZE, len(valid_texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embeddings_list = embeddings
        except Exception as e1:
            # Method 2: If prompt_name fails, try without prompt_name
            try:
                embeddings = model.encode(
                    valid_texts,
                    batch_size=min(BATCH_SIZE, len(valid_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings_list = embeddings
            except Exception as e2:
                # Method 3: Process one by one (slowest but most stable)
                print(f"  Warning: Batch processing failed, processing one by one...")
                for text in valid_texts:
                    try:
                        emb = model.encode(
                            [text],
                            prompt_name="query",
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        embeddings_list.append(emb[0])
                    except:
                        # If prompt_name fails, try without it
                        emb = model.encode(
                            [text],
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        embeddings_list.append(emb[0])
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    if not embeddings_list:
        return 0
    
    # Update database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated_count = 0
    for text_id, embedding in zip(valid_text_ids, embeddings_list):
        try:
            # Convert embedding to JSON string for storage
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            embedding_json = json.dumps(embedding_list)
            cursor.execute(
                "UPDATE clarification_texts SET embedding = ? WHERE id = ?",
                (embedding_json, text_id),
            )
            updated_count += 1
        except Exception as e:
            print(f"  Warning: Failed to update text_id {text_id}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return updated_count

def get_statistics(db_path: str = DB_PATH):
    """Get statistics"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total number of texts
    cursor.execute("SELECT COUNT(*) FROM clarification_texts")
    total = cursor.fetchone()[0]
    
    # Number of texts with embeddings
    cursor.execute(
        "SELECT COUNT(*) FROM clarification_texts WHERE embedding IS NOT NULL AND TRIM(embedding) != ''"
    )
    with_embedding = cursor.fetchone()[0]
    
    # Number of texts without embeddings
    cursor.execute(
        "SELECT COUNT(*) FROM clarification_texts WHERE embedding IS NULL OR TRIM(embedding) = ''"
    )
    without_embedding = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total": total,
        "with_embedding": with_embedding,
        "without_embedding": without_embedding,
    }

# Main execution flow
print("=" * 80)
print("Computing Embeddings for Database")
print("=" * 80 + "\n")

# Check and display library versions
try:
    import transformers
    import torch
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  Warning: CUDA not available, using CPU (will be slow)\n")
except ImportError as e:
    print(f"⚠️  Warning: Could not check versions: {e}\n")

# Check if model is loaded
try:
    _ = model
    print("✓ Model is loaded")
    print(f"  Model type: {type(model).__name__}")
    if hasattr(model, 'max_seq_length'):
        print(f"  Max sequence length: {model.max_seq_length}\n")
    else:
        print()
except NameError:
    print("❌ Error: Model variable 'model' is not defined!")
    print("Please load the model first:")
    print("  from sentence_transformers import SentenceTransformer")
    print("  model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)")
    print("  model.max_seq_length = 8192")
    exit(1)

# Display initial statistics
stats = get_statistics(DB_PATH)
print("Initial Statistics:")
print(f"  Total texts: {stats['total']}")
print(f"  With embedding: {stats['with_embedding']}")
print(f"  Without embedding: {stats['without_embedding']}\n")

if stats['without_embedding'] == 0:
    print("✓ All texts already have embeddings. Nothing to do.")
else:
    # Get texts that need embedding computation
    print("Fetching texts without embeddings...")
    texts_to_process = get_texts_without_embedding(DB_PATH)
    print(f"Found {len(texts_to_process)} texts to process.\n")
    
    if texts_to_process:
        # Batch processing
        print("Computing embeddings...")
        text_ids = [row[0] for row in texts_to_process]
        texts = [row[1] for row in texts_to_process]
        
        # Process in batches
        total_updated = 0
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
            batch_ids = text_ids[i:i + BATCH_SIZE]
            batch_texts = texts[i:i + BATCH_SIZE]
            
            updated = update_embeddings_batch(model, batch_ids, batch_texts, DB_PATH)
            total_updated += updated
        
        print(f"\n✓ Successfully updated {total_updated} embeddings.\n")
        
        # Display final statistics
        stats = get_statistics(DB_PATH)
        print("Final Statistics:")
        print(f"  Total texts: {stats['total']}")
        print(f"  With embedding: {stats['with_embedding']}")
        print(f"  Without embedding: {stats['without_embedding']}\n")

print("=" * 80)
print("Done!")
print("=" * 80)


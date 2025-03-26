import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Load datasets
print("Loading datasets...")
df_prompts_train = pd.read_csv('train_prompts.csv')
df_prompts_dev = pd.read_csv('dev_prompts.csv')
df_answers = pd.read_csv('train_responses.csv')
df_answers_dev = pd.read_csv('dev_responses.csv')

# Enhanced preprocessing function with more careful handling
def preprocess_text(text):
    """
    Enhanced preprocessing for character n-grams
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Simple but effective preprocessing
    text = text.lower().strip()
    
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Add spaces at beginning and end to capture word boundaries
    # This is a key insight from the CHARAGRAM paper
    text = ' ' + text + ' '
    
    return text

# Apply preprocessing
print("Preprocessing data...")
df_prompts_train['processed'] = df_prompts_train['user_prompt'].apply(preprocess_text)
df_prompts_dev['processed'] = df_prompts_dev['user_prompt'].apply(preprocess_text)

# Create an enhanced TF-IDF vectorizer
print("Creating vectorizer...")
vectorizer = TfidfVectorizer(
    analyzer='char',        # Use character n-grams (without word boundaries)
    ngram_range=(2, 5),     # Expanded range to 2-5 for better semantic capture
    min_df=2,               # Reduced min_df to capture more rare but meaningful patterns
    max_df=0.9,             # Ignore n-grams that appear in more than 90% of documents
    sublinear_tf=True,      # Apply sublinear TF scaling
    use_idf=True,           # Use IDF weighting
    norm='l2',              # L2 normalization
    max_features=100000     # Increased feature space for better differentiation
)

print("Fitting vectorizer...")
tfidf_train_matrix = vectorizer.fit_transform(df_prompts_train['processed'])
print(f"Feature matrix shape: {tfidf_train_matrix.shape}")

# Process in batches to reduce memory usage
print("Finding similar prompts (using batching)...")
batch_size = 500
results = []

for i in range(0, len(df_prompts_dev), batch_size):
    # Create batch
    batch_end = min(i + batch_size, len(df_prompts_dev))
    dev_batch = df_prompts_dev.iloc[i:batch_end]
    
    print(f"Processing batch {i//batch_size + 1}/{(len(df_prompts_dev)-1)//batch_size + 1}")
    
    # Transform batch
    dev_vectors = vectorizer.transform(dev_batch['processed'])
    
    # Calculate similarities
    similarity_matrix = cosine_similarity(dev_vectors, tfidf_train_matrix)
    
    # Find best matches
    for j, (idx, row) in enumerate(dev_batch.iterrows()):
        most_similar_idx = np.argmax(similarity_matrix[j])
        dev_id = row['conversation_id']
        train_id = df_prompts_train.iloc[most_similar_idx]['conversation_id']
        
        # Get responses - with safety checks
        try:
            dev_response_row = df_answers_dev[df_answers_dev['conversation_id'] == dev_id]
            if len(dev_response_row) > 0:
                dev_response = str(dev_response_row['model_response'].iloc[0])
            else:
                dev_response = ""
                
            train_response_row = df_answers[df_answers['conversation_id'] == train_id]
            if len(train_response_row) > 0:
                train_response = str(train_response_row['model_response'].iloc[0])
            else:
                train_response = ""
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            dev_response = ""
            train_response = ""
        
        results.append({
            'conversation_id': dev_id,
            'response_id': train_id,
            'dev_prompt': row['user_prompt'],
            'train_prompt': df_prompts_train.iloc[most_similar_idx]['user_prompt'],
            'dev_response': dev_response,
            'train_response': train_response
        })

# Create results DataFrame
test_results_df = pd.DataFrame(results)

# Calculate BLEU score for evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction

    # Smoothing function
    smoothingfunction = SmoothingFunction()

    # BLEU score calculation with safeguards
    def safe_bleu(row):
        try:
            # Ensure values are strings and not empty
            dev_resp = str(row['dev_response']) if not pd.isna(row['dev_response']) else ""
            train_resp = str(row['train_response']) if not pd.isna(row['train_response']) else ""
            
            if not dev_resp or not train_resp:
                return 0.0
                
            return sentence_bleu(
                [dev_resp.split()], 
                train_resp.split(), 
                weights=(0.5, 0.5, 0, 0), 
                smoothing_function=smoothingfunction.method3
            )
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0

    test_results_df['bleu_score'] = test_results_df.apply(safe_bleu, axis=1)

    # Print average BLEU score
    avg_bleu = test_results_df['bleu_score'].mean()
    print(f"Average BLEU score: {avg_bleu:.5f}")
except Exception as e:
    print(f"Couldn't calculate BLEU scores: {e}")

# Save results for track 1 submission (only the required columns)
submission_df = test_results_df[['conversation_id', 'response_id']]
submission_df.to_csv('track_1_test.csv', index=False)
print(f"Saved submission file with {len(submission_df)} rows")

# Print some basic stats
print("\nSubmission summary:")
print(f"Total pairs: {len(submission_df)}")
print(f"Unique response IDs: {submission_df['response_id'].nunique()}")

# Optional: Print top 10 responses by frequency
print("\nTop 10 most frequently matched train responses:")
top_responses = submission_df['response_id'].value_counts().head(10)
print(top_responses)

# Sample of matches for review
print("\nSample of matches (first 5 entries):")
sample_df = test_results_df[['dev_prompt', 'train_prompt', 'bleu_score']].head(5)
for idx, row in sample_df.iterrows():
    print(f"DEV: {row['dev_prompt'][:50]}..." if len(row['dev_prompt']) > 50 else f"DEV: {row['dev_prompt']}")
    print(f"TRAIN: {row['train_prompt'][:50]}..." if len(row['train_prompt']) > 50 else f"TRAIN: {row['train_prompt']}")
    print(f"BLEU: {row['bleu_score']:.4f}")
    print("-" * 50)
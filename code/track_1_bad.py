import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Paths to the separate CSVs
    train_prompts_path = os.path.join("data", "train_prompts.csv")
    train_responses_path = os.path.join("data", "train_responses.csv")
    dev_prompts_path = os.path.join("data", "dev_prompts.csv")
    dev_responses_path = os.path.join("data", "dev_responses.csv")
    test_prompts_path = os.path.join("data", "test_prompts.csv")

    # Read prompts and responses
    train_prompts_df = pd.read_csv(train_prompts_path)
    train_responses_df = pd.read_csv(train_responses_path)
    dev_prompts_df = pd.read_csv(dev_prompts_path)
    dev_responses_df = pd.read_csv(dev_responses_path)
    test_prompts_df = pd.read_csv(test_prompts_path)
    
    # Merge on conversation_id to get a single DataFrame for train
    train_df = pd.merge(train_prompts_df, train_responses_df, on="conversation_id", how="left")
    # Merge on conversation_id to get a single DataFrame for dev
    dev_df = pd.merge(dev_prompts_df, dev_responses_df, on="conversation_id", how="left")
    # Test set only has prompts
    test_df = test_prompts_df
    
    # Drop dev rows with NaN responses if needed
    dev_df = dev_df.dropna(subset=["model_response"])

    return train_df, dev_df, test_df

def build_candidate_set(train_df, dev_df):
    # Combine train and dev into one candidate set
    candidate_df = pd.concat([train_df, dev_df], ignore_index=True)
    return candidate_df

def vectorize_texts(candidate_texts, test_texts):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    candidate_tfidf = vectorizer.fit_transform(candidate_texts)
    test_tfidf = vectorizer.transform(test_texts)
    return candidate_tfidf, test_tfidf

def retrieve_most_similar(candidate_tfidf, test_tfidf, candidate_ids):
    # Compute cosine similarity
    sim_matrix = cosine_similarity(test_tfidf, candidate_tfidf)
    best_candidate_indices = np.argmax(sim_matrix, axis=1)
    return best_candidate_indices

def main():
    # Load and merge data
    train_df, dev_df, test_df = load_data()
    
    # Build candidate set
    candidate_df = build_candidate_set(train_df, dev_df)
    
    # Prepare texts and IDs
    candidate_texts = candidate_df["user_prompt"].astype(str).tolist()
    candidate_ids = candidate_df["conversation_id"].tolist()
    test_texts = test_df["user_prompt"].astype(str).tolist()
    test_ids = test_df["conversation_id"].tolist()
    
    # Vectorize with TF-IDF
    candidate_tfidf, test_tfidf = vectorize_texts(candidate_texts, test_texts)
    
    # Retrieve most similar prompts
    best_candidate_indices = retrieve_most_similar(candidate_tfidf, test_tfidf, candidate_ids)
    
    # Map indices to conversation IDs
    retrieved_ids = [candidate_ids[idx] for idx in best_candidate_indices]
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        "conversation_id": test_ids,
        "response_id": retrieved_ids
    })
    
    # Save results
    os.makedirs("dump", exist_ok=True)
    output_path = os.path.join("dump", "track_1_test.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved Track 1 predictions to: {output_path}")

if __name__ == "__main__":
    main()

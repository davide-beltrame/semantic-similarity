import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """
    Load and combine TRAIN and DEV data (which each have conversation_id, user_prompt, model_response).
    Then load TEST data (which has conversation_id, user_prompt).
    """
    train_df = pd.read_csv("data/train_prompts.csv")  # or your actual path
    dev_df   = pd.read_csv("data/dev_prompts.csv")    # or your actual path
    
    # Combine TRAIN and DEV
    combined_df = pd.concat([train_df, dev_df], ignore_index=True)

    # Load TEST data
    test_df = pd.read_csv("data/test_prompts.csv")  # or your actual path

    return combined_df, test_df

def build_representation(train_prompts):
    """
    Build a TF-IDF representation for the TRAIN+DEV prompts.
    Returns both the trained vectorizer and the matrix of prompt vectors.
    """
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_prompts)
    return vectorizer, train_vectors

def find_best_match(test_prompt, vectorizer, train_vectors):
    """
    Given a single test prompt, compute its vector and return the index of
    the most similar prompt from the TRAIN+DEV set (highest cosine similarity).
    """
    test_vec = vectorizer.transform([test_prompt])
    sims = cosine_similarity(test_vec, train_vectors)
    best_idx = sims.argmax()
    return best_idx

def main():
    # 1) Load data
    combined_df, test_df = load_data()

    # 2) Build representation for TRAIN+DEV prompts
    vectorizer, train_vectors = build_representation(combined_df["user_prompt"])

    # 3) For each TEST prompt, find the most similar TRAIN+DEV question
    results = []
    for i, row in test_df.iterrows():
        test_conversation_id = row["conversation_id"]
        test_prompt = row["user_prompt"]

        # Retrieve index of best match
        best_idx = find_best_match(test_prompt, vectorizer, train_vectors)
        
        # Grab that best match’s conversation_id from combined_df
        best_response_id = combined_df.iloc[best_idx]["conversation_id"]

        # Store pair: (test conversation_id, best match conversation_id)
        results.append({
            "conversation_id": test_conversation_id,
            "response_id": best_response_id
        })

    # 4) Convert results to a DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("track_1_test.csv", index=False)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

class SemanticRetriever:
    def __init__(self, vectorizer_type="tfidf"):
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(2, 5),   # using character n-grams of length 2 to 5
                min_df=2,
                max_df=0.8,
                max_features=5000,
                analyzer='char'
            )
        else:
            raise ValueError(f"Vectorizer type {vectorizer_type} not implemented")
        self.reference_vectors = None
        self.reference_prompts = None
        self.id_to_response = None

    def fit(self, reference_prompts_df, reference_responses_df):
        self.reference_prompts = reference_prompts_df
        # Create a mapping from conversation_id to model_response
        self.id_to_response = dict(zip(
            reference_responses_df['conversation_id'],
            reference_responses_df['model_response']
        ))
        # Fit the vectorizer on the raw 'user_prompt' field.
        self.reference_vectors = self.vectorizer.fit_transform(reference_prompts_df['user_prompt'])

    def retrieve(self, query_df):
        query_vectors = self.vectorizer.transform(query_df['user_prompt'])
        sim_matrix = cosine_similarity(query_vectors, self.reference_vectors)
        best_indices = np.argmax(sim_matrix, axis=1)
        best_conv_ids = self.reference_prompts.iloc[best_indices]['conversation_id'].values
        result_df = pd.DataFrame({
            "conversation_id": query_df['conversation_id'],
            "response_id": best_conv_ids
        })
        return result_df

def main():
    # Choose dataset: default to "test", or pass "dev" as argument.
    dataset = "test"
    if len(sys.argv) > 1:
        dataset = sys.argv[1].strip().lower()
    
    data_dir = "data"
    dump_dir = "dump"
    os.makedirs(dump_dir, exist_ok=True)
    
    # Load CSV files.
    train_prompts = pd.read_csv(os.path.join(data_dir, "train_prompts.csv"))
    train_responses = pd.read_csv(os.path.join(data_dir, "train_responses.csv"))
    dev_prompts = pd.read_csv(os.path.join(data_dir, "dev_prompts.csv"))
    dev_responses = pd.read_csv(os.path.join(data_dir, "dev_responses.csv"))
    test_prompts = pd.read_csv(os.path.join(data_dir, "test_prompts.csv"))
    
    if dataset == "dev":
        # For dev evaluation, use only training data as the retrieval pool.
        combined_prompts = train_prompts.copy()
        combined_responses = train_responses.copy()
        target_prompts = dev_prompts.copy()
        output_file = os.path.join(dump_dir, "track_1_working.csv")
    else:
        # For test, combine train and dev into the retrieval pool.
        combined_prompts = pd.concat([train_prompts, dev_prompts], ignore_index=True)
        combined_responses = pd.concat([train_responses, dev_responses], ignore_index=True)
        target_prompts = test_prompts.copy()
        output_file = os.path.join(dump_dir, "track_1_test.csv")
    
    # Initialize and fit the retriever.
    retriever = SemanticRetriever(vectorizer_type="tfidf")
    retriever.fit(combined_prompts, combined_responses)
    
    # Retrieve predictions.
    predictions = retriever.retrieve(target_prompts)
    
    if dataset == "dev":
        # For dev evaluation, output predicted responses along with the true responses.
        # Build a mapping for dev responses.
        dev_response_map = dict(zip(dev_responses['conversation_id'], dev_responses['model_response']))
        # Get predicted response text from the retrieval pool mapping.
        predictions['predicted_response'] = predictions['response_id'].map(lambda x: retriever.id_to_response.get(x, ""))
        # Add the true dev response by matching conversation_id.
        predictions['model_response'] = predictions['conversation_id'].map(lambda x: dev_response_map.get(x, ""))
        predictions[['conversation_id', 'predicted_response', 'model_response']].to_csv(output_file, index=False)
        print(f"Dev evaluation file saved to {output_file}")
    else:
        # For test, simply output conversation_id and response_id.
        predictions.to_csv(output_file, index=False)
        print(f"Test predictions saved to {output_file}")

if __name__ == "__main__":
    main()

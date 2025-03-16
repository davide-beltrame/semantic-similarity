import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main():
    # 1) Load your DEV predictions (conversation_id, response_id)
    dev_pred = pd.read_csv("track_1_dev.csv")
    
    # 2) Load DEV ground truth
    dev_prompts = pd.read_csv("data/dev_prompts.csv")      # columns: conversation_id, user_prompt
    dev_responses = pd.read_csv("data/dev_responses.csv")  # columns: conversation_id, model_response
    # Merge to get the DEV gold data
    dev_gold = pd.merge(dev_prompts, dev_responses, on="conversation_id", how="left")
    # Rename model_response -> gold_response for clarity
    dev_gold.rename(columns={"model_response": "gold_response"}, inplace=True)

    # 3) Load TRAIN data (or TRAIN+DEV if that’s your retrieval pool)
    train_prompts = pd.read_csv("data/train_prompts.csv")
    train_responses = pd.read_csv("data/train_responses.csv")
    train_data = pd.merge(train_prompts, train_responses, on="conversation_id", how="left")
    # Now train_data has columns: conversation_id, user_prompt, model_response

    # 4) Merge predictions with DEV gold
    merged = pd.merge(dev_pred, dev_gold, on="conversation_id", how="left")
    # Now each row has: conversation_id, response_id, gold_response

    # 5) Merge again to get the retrieved response text from TRAIN data
    merged = pd.merge(
        merged, 
        train_data[["conversation_id", "model_response"]],
        left_on="response_id",
        right_on="conversation_id",
        how="left",
        suffixes=("", "_train")
    )
    # Rename the retrieved model_response for clarity
    merged.rename(columns={"model_response": "retrieved_response"}, inplace=True)

    # 6) Compute BLEU
    smoothing = SmoothingFunction().method3
    merged["bleu_score"] = merged.apply(
        lambda row: sentence_bleu(
            [str(row["gold_response"]).split()],
            str(row["retrieved_response"]).split(),
            weights=(0.5, 0.5, 0, 0),  # bigram BLEU example
            smoothing_function=smoothing
        ),
        axis=1
    )

    # 7) Print some results + save
    print(merged[["conversation_id", "gold_response", "retrieved_response", "bleu_score"]].head(10))
    merged.to_csv("evaluation_dev_track_1.csv", index=False)

if __name__ == "__main__":
    main()

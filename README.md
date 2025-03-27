# Methodology Description

I implemented minimal preprocessing across all tracks, removing extra spaces, normalizing whitespace, and filtering out responses that could be considered invalid with high likelihood (< 3 characters, since the shortest answer is "No."). I found that both aggressive preprocessing and no preprocessing at all worsened BLEU scores. My approach focused on optimizing each representation method through careful hyperparameter tuning and trial-and-error, rather than extensive preprocessing.

## Track 1: TF-IDF Character N-grams

After evaluating various discrete representation methods, TF-IDF clearly outperformed other options in maximizing BLEU scores. I conducted extensive hyperparameter optimization via grid search, finding optimal results with character-level n-grams (range 2-4), minimum document frequency of 2, maximum document frequency of 0.8, and sublinear term frequency scaling. Using character-level analysis rather than word-level captured subtle textual patterns and helped handle misspellings and morphological variations in the conversational data.

## Track 2: FastText with TF-IDF Weighting

For distributed representations, I implemented FastText with TF-IDF weighting to combine the advantages of word embeddings with document frequency information. The model uses 300-dimensional vectors, context window of 3, and minimum word count of 5. I enhanced the standard approach by implementing TF-IDF weighted sentence embeddings, where each word vector is weighted by its TF-IDF score before aggregation. This significantly improved performance by emphasizing semantically important terms and reducing the influence of common words.

## Track 3: Sentence Transformers

I leveraged Sentence Transformers from Hugging Face, evaluating multiple pre-trained models including MiniLM, RoBERTa, and MPNet variants. After comprehensive testing, "all-mpnet-base-v2" delivered the best performance with a BLEU score of 0.1079. This model effectively captures semantic relationships through its transformer architecture and pre-training on diverse textual data. I explored fine-tuning the model on our specific dataset but found negligible improvements over the pre-trained version, suggesting the model was already well-optimized for semantic similarity tasks.
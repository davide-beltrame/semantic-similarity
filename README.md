# Semantic Similarity Retrieval Methods
### Language Technology Assignment #2
#### Spring 2025

The purpose of this assignment is to explore different semantic similarity retrieval methods and evaluate their performance using BLEU scores. The task involves implementing three different tracks: TF-IDF character n-grams, FastText embeddings, and Sentence Transformers.

I implemented minimal preprocessing across all tracks, removing extra spaces, normalizing whitespace, and filtering out responses that could be considered invalid with high likelihood (< 3 characters, since the shortest answer are "No." and "Yes"). I found that both aggressive preprocessing and no preprocessing at all worsened BLEU scores. My approach focused on optimizing each representation method through careful hyperparameter tuning and trial-and-error, rather than extensive preprocessing.

## Track 1: TF-IDF Character N-grams

After evaluating various discrete representation methods, TF-IDF clearly outperformed other options in maximizing BLEU scores. I conducted extensive hyperparameter optimization via grid search, finding optimal results with character-level n-grams (range 2-4), minimum document frequency of 3, maximum document frequency of 0.7, and sublinear term frequency scaling with L1 normalization. Using character-level analysis with word boundaries captured subtle textual patterns and helped handle misspellings and morphological variations in the conversational data.

## Track 2: FastText Word Embeddings

For distributed representations, I implemented FastText to capture semantic relationships between words in context. The model uses 300-dimensional vectors, context window of 5, and minimum word count of 2, trained for 20 epochs on both prompts and responses. FastText's subword modeling effectively handles out-of-vocabulary words and morphological variations, which is valuable for conversational data containing misspellings and informal language. For sentence embeddings, I used mean-pooling of word vectors followed by normalization, which produced well-balanced semantic representations that capture the overall meaning of prompts.

## Track 3: Sentence Transformers

I leveraged Sentence Transformers from Hugging Face, evaluating multiple pre-trained models. After comprehensive testing, the "all-mpnet-base-v2" model delivered the best performance with a BLEU score of approximately 0.108. This model effectively captures semantic relationships through its transformer architecture and pre-training on diverse textual data. Using standard preprocessing with this model proved optimal, as the transformer's contextual embeddings already handle semantic variations effectively without requiring additional preprocessing.
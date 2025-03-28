# Semantic Similarity Retrieval Methods
### Language Technology Assignment #2
#### Spring 2025

The purpose of this assignment is to explore different semantic similarity retrieval methods and evaluate their performance using BLEU scores. The task involves implementing three different tracks: TF-IDF character n-grams, FastText embeddings, and Sentence Transformers.

I implemented minimal preprocessing across all tracks, removing extra spaces, normalizing whitespace, and filtering out responses that could be considered invalid with high likelihood (< 3 characters, since the shortest answer are "No." and "Yes"). I found that both aggressive preprocessing and no preprocessing at all worsened BLEU scores. My approach focused on optimizing each representation method through careful hyperparameter tuning and trial-and-error, rather than extensive preprocessing.

## Track 1: TF-IDF Character N-grams

After testing several methods, TF-IDF gave the best results in terms of BLEU scores. 
I used extensive grid search to find the best hyperparameters, which turned out to be character-level n-grams (2-4 characters), 
a minimum document frequency of 3, a maximum document frequency of 0.7, sublinear TF scaling, and L1 normalization. 
Using characters instead of whole words helped capture minor textual differences, including misspellings and variations in word forms common in conversational data.

## Track 2: FastText Word Embeddings

For distributed representations, I chose FastText because it handles semantic relationships between words effectively. 
I set the vector size to 300, used a context window of 5 words, set the minimum word frequency at 2, and trained the model for 20 epochs on both prompts and responses. 
FastText's ability to break words into subwords meant it could easily handle informal language and misspellings, which are frequent in chat data. 
For sentence embeddings, averaging the word vectors followed by normalization gave balanced results and effectively captured the overall meaning.

## Track 3: Sentence Transformers

I explored multiple Sentence Transformer models from Hugging Face and eventually settled on the "all-mpnet-base-v2" model, 
which achieved the highest BLEU score of around 0.108. 
This model was particularly effective due to its transformer-based architecture and pre-training on diverse data. 
Simple preprocessing worked best here since the transformer model already captures semantic nuances well without needing additional text processing.
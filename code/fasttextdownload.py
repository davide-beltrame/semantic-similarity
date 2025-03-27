from gensim.models import FastText
import os

# Create a minimal corpus for testing
sentences = [['hello', 'world'], ['testing', 'fasttext']]

# Initialize model directory
os.makedirs("models", exist_ok=True)

# Train a tiny model just to test FastText works
print("Initializing FastText...")
model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=1)
model.save("models/fasttext_test.bin")
print("FastText initialized successfully! You can now run the grid search.")
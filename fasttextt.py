from pynn import *
import fasttext

"""
@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
"""

print("Loading model...")
# Load the pre-trained model (ensure the path is correct)
fasttext_model = fasttext.load_model("/Users/jamesmoore/python/fastText/cc.en.300.bin")
print("Done\n")

# Define a fastText embedding node using dictionary-based customization
nFastTextEmbedding = {
    "name": "fastText embedding",
    "embedding_model": fasttext_model,
    "vector_size": 300,
    # The term function receives a list of tokens and returns a list of vectors.
    "term": lambda self, tokens: [self.embedding_model.get_word_vector(token) for token in tokens]
}

# Tokenization node (simple whitespace split)
nTokenize = {
    "name": "tokenize",
    "term": lambda self, input: input.lower().split()
}

# Create the input node.
nIO = Base(name="i/o node", term=lambda self, input: input)

# Chain tokenization and fastText embedding nodes.
nIO.connect(Base, nTokenize) \
   .connect(Base, nFastTextEmbedding)

output_vector = nIO("How are you?")

print(output_vector)

"""def vector_to_text(vector, model):
    \"""
    Given an output vector and a fastText model, find the word in the model's vocabulary
    whose embedding is most similar to the vector (using cosine similarity).
    \"""
    # Get all words in the model's vocabulary.
    words = model.get_words()
    # Normalize the output vector.
    norm_vector = vector / np.linalg.norm(vector)
    
    best_word = None
    best_similarity = -1

    # Iterate over the vocabulary (this can be slow for large models).
    for word in words:
        word_vec = model.get_word_vector(word)
        # Normalize the word vector.
        norm_word_vec = word_vec / np.linalg.norm(word_vec)
        similarity = np.dot(norm_vector, norm_word_vec)
        if similarity.any() > best_similarity.any():
            best_similarity = similarity
            best_word = word

    return best_word

# Example usage:
# Assume `fasttext_model` is already loaded via fasttext.load_model("cc.en.300.bin")
# and `output_vector` is the vector produced by your network.
word = vector_to_text(output_vector, fasttext_model)
print("Translated output:", word)
"""

def vector_to_text(vector, model, k=1):
    """
    Convert a single vector to its nearest word using fastText.
    
    Parameters:
        vector (list or np.ndarray): The token vector.
        model: The fastText model.
        k (int): Number of nearest neighbors to consider.
    
    Returns:
        The nearest word (if k == 1) or a list of (similarity, word) tuples.
    """
    neighbors = model.get_nearest_neighbors(vector, k=k)
    if k == 1:
        return neighbors[0][1] if neighbors else ""
    else:
        return neighbors

def vector_to_text_sequence(output_vector, model, k=1):
    """
    Convert a 2D or 3D numpy array of token vectors to text.
    
    If output_vector is 2D, it's treated as (num_tokens, vector_size) for one sentence.
    If it's 3D, it's treated as (num_sentences, num_tokens, vector_size).
    
    Returns:
        A string with tokens joined by spaces, and sentences separated by newlines.
    """
    # If it's a single sentence
    if output_vector.ndim == 2:
        tokens = [vector_to_text(token_vec, model, k=k) for token_vec in output_vector]
        return " ".join(tokens)
    # If it's multiple sentences
    elif output_vector.ndim == 3:
        sentences = []
        for sentence in output_vector:
            tokens = [vector_to_text(token_vec, model, k=k) for token_vec in sentence]
            sentences.append(" ".join(tokens))
        return "\n".join(sentences)
    else:
        raise ValueError("Output vector must be 2D or 3D.")

# Example usage:
# Assume fasttext_model is loaded with:
#   import fasttext
#   fasttext_model = fasttext.load_model("cc.en.300.bin")
# And output_vector is your 3D numpy array.
translated_text = vector_to_text_sequence(output_vector, fasttext_model)
print("Translated output:")
print(translated_text)

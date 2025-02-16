nNormalize = {
    "name": "normalize",
    "mean": 0.0,      # Default mean; can be updated by the user.
    "std": 1.0,       # Default standard deviation; can be updated.
    "term": lambda self, input: (input - self.mean) / self.std
}

nTokenize = {
    "name": "tokenize",
    "term": lambda self, input: input.split()  # Simple whitespace tokenization.
}

nMapping = {
    "name": "mapping",
    "mapping": {"how": 1, "are": 2, "you": 3, "this": 4, "morning": 5},
    "default": 0,  # Use 0 for tokens not found in the mapping.
    "term": lambda self, tokens: [self.mapping.get(token, self.default) for token in tokens]
}

"""
nIO = Base(name="i/o node", term=lambda self, input: input)

nIO.connect(Base, nTokenize) \
   .connect(Base, nMapping) \
   .connect(Base, nNormalize)  # Optionally, if you need to normalize numerical features later.
"""

"""
# fastText using gensim
from gensim.models import KeyedVectors

# Load the embeddings (this may take a few minutes for large files)
fasttext_model = KeyedVectors.load_word2vec_format('cc.en.300.vec', binary=False)

nFastTextEmbedding = {
    "name": "fastText embedding",
    "embedding_model": fasttext_model,
    "vector_size": 300,  # for the 300-dimensional embeddings
    "default": [0.0] * 300,  # a default vector for out-of-vocabulary tokens
    "term": lambda self, tokens: [
         self.embedding_model.get_vector(token) if token in self.embedding_model else self.default 
         for token in tokens
    ]
}

\"""# Example pipeline:
nIO = Base(name="i/o node", term=lambda self, input: input)

# Tokenization node (simple whitespace split)
nTokenize = {
    "name": "tokenize",
    "term": lambda self, input: input.lower().split()
}

# Connect nodes: input -> tokenization -> fastText embedding
nIO.connect(Base, nTokenize) \
   .connect(Base, nFastTextEmbedding)
\"""
"""

import fasttext

# Load the pre-trained model (ensure the path is correct)
fasttext_model = fasttext.load_model("cc.en.300.bin")

# Define a fastText embedding node using dictionary-based customization
nFastTextEmbedding = {
    "name": "fastText embedding",
    "embedding_model": fasttext_model,
    "vector_size": 300,
    # The term function receives a list of tokens and returns a list of vectors.
    "term": lambda self, tokens: [self.embedding_model.get_word_vector(token) for token in tokens]
}

"""
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
"""
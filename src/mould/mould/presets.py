"""
Presets for common neural network architectures and components.
"""

import random
import math
from typing import List, Optional, Callable, Any

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

from .mould import Mould
from .activations import sigmoid, tanh, relu

nNormalize = {
    "name": "normalize",
    "description": "Normalizes input values to a range of [0, 1]",
    "parameters": {
        "min_val": 0.0,
        "max_val": 1.0
    }
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

# Preset for word embeddings using FastText
if FASTTEXT_AVAILABLE:
    try:
        fasttext_model = fasttext.load_model("cc.en.300.bin")
    except:
        print("Warning: Could not load FastText model. Word embedding features will be disabled.")
        fasttext_model = None
else:
    fasttext_model = None

def get_word_embedding(word: str) -> List[float]:
    """Get word embedding vector using FastText"""
    if not FASTTEXT_AVAILABLE or fasttext_model is None:
        return [random.uniform(-0.1, 0.1) for _ in range(300)]  # Fallback to random embeddings
    return fasttext_model.get_word_vector(word).tolist()

# Common activation function presets
activation_presets = {
    "sigmoid": {
        "func": sigmoid,
        "description": "Sigmoid activation function"
    },
    "tanh": {
        "func": tanh,
        "description": "Hyperbolic tangent activation function"
    },
    "relu": {
        "func": relu,
        "description": "Rectified Linear Unit activation function"
    }
}
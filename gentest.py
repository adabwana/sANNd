from activate import *
from pynn import *

from PIL import Image
import numpy as np


def tokenize_and_normalize_image(image, patch_size=(16, 16)):
    """
    Normalize an image and split it into flattened patches.
    
    Parameters:
        image (np.ndarray): Input image array with shape (H, W, C).
        patch_size (tuple): Desired (height, width) of each patch.
    
    Returns:
        np.ndarray: Array of flattened patches with shape (num_patches, patch_area * C).
    """
    # Normalize the image to [0, 1] if it isn't already in float32.
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    H, W, C = image.shape
    ph, pw = patch_size
    patches = []

    # Iterate over the image to extract non-overlapping patches.
    for i in range(0, H, ph):
        for j in range(0, W, pw):
            patch = image[i:i+ph, j:j+pw, :]
            # Only include patches that match the full patch size.
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(patch.flatten())
    
    patches = np.array(patches)

    #!!!HACK!!!
    #encoder_node.weight = np.random.randn(patches.shape[1], 2 * latent_dim).astype(np.float32) * 0.01
    #encoder_node.weight = np.random.randn(input_dim, 2 * latent_dim).astype(np.float32) * 0.01
    #encoder_node.bias = np.zeros((2 * latent_dim,), dtype=np.float32)
    
    #decoder_node.weight = np.random.randn(patches.shape[1], latent_dim).astype(np.float32) * 0.01
    ##decoder_node.weight = np.random.randn(input_dim, latent_dim).astype(np.float32) * 0.01
    #decoder_node.bias = np.zeros((latent_dim,), dtype=np.float32)

    return patches


def custom_encoder(input, node):
    """
    Encoder function for a variational autoencoder.
    
    This function performs a linear transformation on the input using
    node.weight and node.bias, then splits the result into two halves:
    one for the mean (μ) and one for the log-variance (logσ²) of the latent distribution.
    
    Assumptions:
      - 'input' is a NumPy array of shape (batch_size, input_dim).
      - node.weight is a NumPy array of shape (input_dim, 2 * latent_dim).
      - node.bias is a NumPy array of shape (2 * latent_dim,).
    
    Returns:
      A tuple (mean, logvar), where both are NumPy arrays of shape (batch_size, latent_dim).
    """
    # Linear transformation: (batch_size, 2*latent_dim)
    linear_output = np.dot(input, node.weight) + node.bias
    #latent_dim = node.weight.shape[1] // 2
    #latent_dim = nImageTokenizer.get("patch_size")[0]
    mean = linear_output[:, :latent_dim]
    logvar = linear_output[:, latent_dim:]
    return (mean, logvar)

def reparameterize(encoder_out):
    """
    Reparameterization trick: Given (mean, logvar) from the encoder,
    sample a latent vector z = mean + exp(0.5 * logvar) * epsilon,
    where epsilon is drawn from a standard normal distribution.
    
    Parameters:
      - encoder_out: a tuple (mean, logvar) with shapes (batch_size, latent_dim)
    
    Returns:
      A NumPy array z of shape (batch_size, latent_dim) representing the sampled latent vectors.
    """
    mean, logvar = encoder_out
    epsilon = np.random.randn(*mean.shape)  # sample from standard normal
    z = mean + np.exp(0.5 * logvar) * epsilon
    return z

def custom_decoder(latent, node):
    """
    Decoder function for a variational autoencoder.
    
    This function reconstructs the input data from the latent vector using
    a linear transformation defined by node.weight and node.bias. Optionally,
    an activation function (e.g., sigmoid) can be applied if defined as node.activate.
    
    Assumptions:
      - 'latent' is a NumPy array of shape (batch_size, latent_dim).
      - node.weight is a NumPy array of shape (latent_dim, output_dim).
      - node.bias is a NumPy array of shape (output_dim,).
    
    Returns:
      A NumPy array of shape (batch_size, output_dim) representing the reconstructed output.
    """
    linear_output = np.dot(latent, node.weight) + node.bias
    if hasattr(node, 'activate') and callable(node.activate):
        output = node.activate(linear_output)
    else:
        output = linear_output
    return output

def process_image(input_image, width, height, channels):
    # Convert image to NumPy array, normalize, etc.
    import numpy as np
    arr = np.array(input_image).astype(np.float32) / 255.0
    # Optionally reshape if needed (e.g., flatten or add batch dimension)
    return arr

def reconstruct_image(patches, original_height, original_width, patch_size):
    """
    Reassemble patches into an image.
    
    Parameters:
        patches (np.ndarray): Array of patches with shape (num_patches, patch_size, patch_size, channels)
        original_height (int): Original image height.
        original_width (int): Original image width.
        patch_size (int): Size of each patch (assumed square).
        
    Returns:
        np.ndarray: Reconstructed image with shape (rows*patch_size, cols*patch_size, channels)
    """
    # Compute number of patches along each dimension
    rows = original_height // patch_size
    cols = original_width // patch_size
    
    # Ensure the number of patches is as expected
    assert patches.shape[0] == rows * cols, "Mismatch in number of patches"
    
    # Reshape patches into (rows, cols, patch_size, patch_size, channels)
    patches = patches.reshape(cols, rows, patch_size, patch_size, -1)
    
    # Swap axes to bring patch dimensions together: (rows, patch_size, cols, patch_size, channels)
    patches = patches.transpose(0, 2, 1, 3, 4)
    
    # Finally, reshape to get the full image: (rows*patch_size, cols*patch_size, channels)
    reconstructed = patches.reshape(rows * patch_size, cols * patch_size, -1)
    return reconstructed


#Open an image file
img = Image.open("DISK1CRP.jpg")

# Get image dimensions.
#width, height = img.size  # Returns (width, height)

# Determine the number of channels from the mode.
# Common modes: "RGB" (3 channels), "L" (grayscale, 1 channel)
#channels = 3 if img.mode == "RGB" else 1

# Convert the image to a NumPy array with shape (H, W, C)
in_img = np.array(img)
print("Image shape:", in_img.shape)

input_dim = in_img.shape[0] * in_img.shape[1] * in_img.shape[2]
latent_dim = 32

# Define the image tokenizer/normalizer node as a dictionary.
nImageTokenizer = {
    "name": "image tokenizer/normalizer",
    "patch_size": (16, 16),  # you can adjust the patch size as needed
    "term": lambda self, input: tokenize_and_normalize_image(input, self.patch_size)
}

nLatent = {
    "name": "latent",
    "term": lambda self, encoder_out: reparameterize(encoder_out)  # reparameterize is your function that returns z
}

"""nEncoder = {
    "name": "encoder",
    # Example: a linear transformation plus activation that outputs a concatenated [mean, log_var] vector.
    "weight": 0.5,  # placeholder for initial weight or use a custom initializer
    "bias": 0.0,
    "term": lambda self, input: custom_encoder(input, self)  # custom_encoder is a function you define
}"""


"""nImageNode = {
    "name": "image node",
    "width": width,
    "height": height,
    "channels": channels,
    "term": lambda self, input: process_image(input, self.width, self.height, self.channels)
}"""

# Initialize encoder node parameters properly:
nEncoder = {
    "name": "encoder",
    "weight": 0.5,  # Initial parameter or use an initializer
    "bias": 0.0,
    "term": lambda self, input: custom_encoder(input, self)
}

nDecoder = {
    "name": "decoder",
    "weight": 0.5,  # again, initial parameter or use an initializer
    "bias": 0.0,
    "term": lambda self, z: custom_decoder(z, self)  # custom_decoder is a function you define
}



# Input node that feeds your data (ensure it normalizes or passes data as needed)
nIO = Base(
    name="i/o node",
    term=lambda self, input: input  # identity for now; preprocessed data should be provided
)

token_node = nIO.connect(Base, nImageTokenizer)

# Connect the encoder to the input
encoder_node = token_node.connect(Base, nEncoder)

# Feed the encoder output into the latent node
latent_node = encoder_node.connect(Base, nLatent)

# Feed the latent vector into the decoder
decoder_node = latent_node.connect(Base, nDecoder)

# Optionally, you could add a reconstruction activation node (e.g., softplus or sigmoid) to bound your outputs
nReconAct = {
    "name": "reconstruction activation",
    "term": lambda self, input: np_sigmoid(input)  # define or import a sigmoid function
}
decoder_node.connect(Base, nReconAct)

reconstructed = nIO(in_img)
#reconstructed = reconstruct_image(nIO(in_img), in_img.shape[0], in_img.shape[1], nImageTokenizer.get("patch_size")[0])

# Assume `reconstructed` is your output NumPy array from the reconstruction activation node.
# Scale values to 0-255 if necessary.
reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

# Create an image from the array.
img = Image.fromarray(reconstructed_uint8)

# Save the image to disk.
img.save("reconstructed_image.png")


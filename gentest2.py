from activate import *  # Your activation functions (including np_sigmoid)
from pynn import *      # Your pyNN framework
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
    # Extract only full patches.
    for i in range(0, H - H % ph, ph):
        for j in range(0, W - W % pw, pw):
            patch = image[i:i+ph, j:j+pw, :]
            patches.append(patch.flatten())
    return np.array(patches)

def custom_encoder(input, node):
    """
    Encoder function for a variational autoencoder.
    Expects:
      - input: (batch_size, input_dim) where input_dim = patch_area * channels.
      - node.weight: shape (input_dim, 2 * latent_dim)
      - node.bias: shape (2 * latent_dim,)
    Returns:
      A tuple (mean, logvar), each of shape (batch_size, latent_dim).
    """
    linear_output = np.dot(input, node.weight) + node.bias
    # latent_dim must be defined globally (or stored on the node)
    mean = linear_output[:, :latent_dim]
    logvar = linear_output[:, latent_dim:]
    return (mean, logvar)

def reparameterize(encoder_out):
    """
    Reparameterization trick: sample z = mean + exp(0.5 * logvar) * epsilon.
    """
    mean, logvar = encoder_out
    epsilon = np.random.randn(*mean.shape)
    z = mean + np.exp(0.5 * logvar) * epsilon
    return z

def custom_decoder(latent, node):
    """
    Decoder function for a variational autoencoder.
    Expects:
      - latent: (batch_size, latent_dim)
      - node.weight: shape (latent_dim, output_dim)
      - node.bias: shape (output_dim,)
    Returns:
      A reconstructed vector of shape (batch_size, output_dim).
    """
    linear_output = np.dot(latent, node.weight) + node.bias
    if hasattr(node, 'activate') and callable(node.activate):
        return node.activate(linear_output)
    else:
        return linear_output

def process_image(input_image, width, height, channels):
    """
    Convert input image to a normalized NumPy array.
    """
    arr = np.array(input_image).astype(np.float32) / 255.0
    return arr

def reconstruct_image(flat_patches, original_height, original_width, patch_size, channels):
    """
    Reassemble flattened patches into an image.
    
    Parameters:
      - flat_patches: np.ndarray of shape (num_patches, patch_area * channels)
      - original_height, original_width: dimensions of the original image.
      - patch_size: integer (assumes square patches).
      - channels: number of color channels.
    
    Returns:
      A reconstructed image as a NumPy array of shape (rows * patch_size, cols * patch_size, channels).
    """
    ph = patch_size
    # Compute how many full patches fit into the original dimensions.
    rows = original_height // ph
    cols = original_width // ph
    num_expected = rows * cols
    assert flat_patches.shape[0] == num_expected, "Mismatch in number of patches"
    
    # Reshape each flattened patch back to (patch_size, patch_size, channels)
    patches = flat_patches.reshape(num_expected, ph, ph, channels)
    # Reshape into a grid: (rows, cols, patch_size, patch_size, channels)
    patches = patches.reshape(rows, cols, ph, ph, channels)
    # Rearrange axes to interleave patch pixels: (rows, patch_size, cols, patch_size, channels)
    patches = patches.transpose(0, 2, 1, 3, 4)
    # Combine into full image: (rows * patch_size, cols * patch_size, channels)
    reconstructed = patches.reshape(rows * ph, cols * ph, channels)
    return reconstructed

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between predicted values and true labels.

    Args:
        y_true: Ground truth values, shape=(n_samples,)
        y_pred: Predicted values, shape=(n_samples,)

    Returns:
        Loss value, shape=()
    """
    return np.mean((y_true - y_pred) ** 2)

def kl_divergence(p: np.ndarray, q: np.ndarray):
    """
    Calculate the KL divergence between two probability distributions.

    Args:
        p: Ground truth probabilities, shape=(n_classes,)
        q: Predicted probabilities, shape=(n_classes,)

    Returns:
        Loss value, shape=()
    """
    return -np.sum(p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q)))

def generate(image):
    H, W, C = image.shape

    # Process the image through the pipeline
    output_flat_patches = nIO(image)
    # output_flat_patches shape should be (num_patches, patch_area * C)
    print("Output patches shape:", output_flat_patches.shape)

    # Reconstruct the image from patches
    reconstructed_img = reconstruct_image(output_flat_patches, H, W, patch_size, C)
    print("Reconstructed image shape:", reconstructed_img.shape)

    # Convert to uint8 and save the image
    return (reconstructed_img * 255).astype(np.uint8)

def init_coders(image):
    H, W, C = image.shape

    input_dim_per_patch = patch_size * patch_size * C

    encoder_node.weight = np.random.randn(input_dim_per_patch, 2 * latent_dim).astype(np.float32) * 0.01
    encoder_node.bias = np.zeros((2 * latent_dim,), dtype=np.float32)

    decoder_node.weight = np.random.randn(latent_dim, input_dim_per_patch).astype(np.float32) * 0.01
    decoder_node.bias = np.zeros((input_dim_per_patch,), dtype=np.float32)

def open_img(file_name):
    
    return np.array(Image.open(file_name))


def save_img(image, file_name):
    img_out = Image.fromarray(image)
    img_out.save(file_name)
    print("Reconstructed image saved - "+file_name)

"""
To train the generative model you’ll need to define a loss function that typically includes:

Reconstruction Loss: Measures how well the decoder reconstructs the input (e.g., mean squared error or binary cross‑entropy).
KL Divergence: Regularizes the latent space to be close to a standard normal distribution.
You would then write a training loop that:

Feeds batches of data into the input node.
Computes the output from the reconstruction node.
Evaluates the loss.
Computes gradients (you might integrate with an autodiff library or implement a rudimentary gradient update if your framework supports it).
Updates the parameters of each node accordingly.
"""

def train(self, input_img, desired_img=None, epochs=1000, learning_rate=0.01):
    if desired_img is None: desired_img = input_img

    for epoch in range(epochs):
        new_img = generate(input_img)
        
        #Sample
        if epoch == 1 or epoch % 100 == 0:
            save_img(new_img, f"sample.{epoch}.png")

        recon_loss = mean_squared_error(desired_img, new_img) #error
        latent_div = kl_divergence(desired_img, new_img)

            # Update the weights and biases using backpropagation
            self.layer.neurons[0].backward(error, learning_rate)

            # Calculate loss
            #total_loss += self.loss_function.compute_loss(np.array([expected_value]), np.array([float(prediction.split(" ")[-1][:-1])]))

        # Print the loss every 100 epochs
        #if epoch % 100 == 0:
        #    print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(training_data)}")

# --- Main Execution ---

# Load the image using PIL
#img = Image.open("DISK1CRP.jpg")
img = open_img("DSC05222.JPG")

print("Input image shape:", img.shape)  # e.g., (707, 800, 3)
#H, W, C = in_img.shape

# Global dimensions
patch_size = 64
#input_dim_per_patch = patch_size * patch_size * C
latent_dim = 96 # choose based on your application

# Create pyNN nodes

# Image tokenizer/normalizer node
nImageTokenizer = {
    "name": "image tokenizer/normalizer",
    "patch_size": (patch_size, patch_size),
    "term": lambda self, input: tokenize_and_normalize_image(input, self.patch_size)
}

# Latent sampling node
nLatent = {
    "name": "latent",
    "term": lambda self, encoder_out: reparameterize(encoder_out)
}

# Encoder node: initialize weight and bias with proper dimensions
nEncoder = {
    "name": "encoder",
    #"weight": np.random.randn(input_dim_per_patch, 2 * latent_dim).astype(np.float32) * 0.01,
    #"bias": np.zeros((2 * latent_dim,), dtype=np.float32),
    "term": lambda self, input: custom_encoder(input, self)
}

# Decoder node: output dimension equals input_dim_per_patch
nDecoder = {
    "name": "decoder",
    #"weight": np.random.randn(latent_dim, input_dim_per_patch).astype(np.float32) * 0.01,
    #"bias": np.zeros((input_dim_per_patch,), dtype=np.float32),
    "term": lambda self, z: custom_decoder(z, self)
}

# Reconstruction activation node (using a sigmoid for pixel output in [0, 1])
nReconAct = {
    "name": "reconstruction activation",
    "term": lambda self, input: np_sigmoid(input)
}

# Build the pyNN pipeline
nIO = Base(name="i/o node")

token_node = nIO.connect(Base, nImageTokenizer)
encoder_node = token_node.connect(Base, nEncoder)
latent_node = encoder_node.connect(Base, nLatent)
decoder_node = latent_node.connect(Base, nDecoder)
decoder_node.connect(Base, nReconAct)  # chain reconstruction activation



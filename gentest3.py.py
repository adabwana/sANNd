from activate import *  # Your activation functions (including np_sigmoid)
from pynn import *      # Your pyNN framework
from PIL import Image
import numpy as np
from optimizer import Adam  # Import your Adam optimizer class

# ----- Data Preprocessing Functions -----

def tokenize_and_normalize_image(image, patch_size=(16, 16)):
    """
    Normalize an image and split it into flattened patches.
    
    Returns an array of flattened patches of shape 
    (num_patches, patch_area * channels).
    """
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

def process_image(input_image, width, height, channels):
    arr = np.array(input_image).astype(np.float32) / 255.0
    return arr

def reconstruct_image(flat_patches, original_height, original_width, patch_size, channels):
    """
    Reassemble flattened patches into an image.
    Reconstructs an image by arranging patches in a grid. If the original dimensions 
    aren’t multiples of patch_size, the output will be cropped.
    """
    ph = patch_size
    rows = original_height // ph
    cols = original_width // ph
    num_expected = rows * cols
    assert flat_patches.shape[0] == num_expected, "Mismatch in number of patches"
    # Reshape each flattened patch back to (patch_size, patch_size, channels)
    patches = flat_patches.reshape(num_expected, ph, ph, channels)
    # Reshape into grid: (rows, cols, ph, ph, channels)
    patches = patches.reshape(rows, cols, ph, ph, channels)
    # Rearrange axes: (rows, ph, cols, ph, channels)
    patches = patches.transpose(0, 2, 1, 3, 4)
    # Combine into full image: (rows * ph, cols * ph, channels)
    reconstructed = patches.reshape(rows * ph, cols * ph, channels)
    return reconstructed

# ----- VAE Functions -----

def custom_encoder(input, node):
    """
    Performs a linear transformation on input (shape: (batch_size, input_dim))
    using node.weight and node.bias, then splits into mean and logvar.
    """
    linear_output = np.dot(input, node.weight) + node.bias
    mean = linear_output[:, :latent_dim]
    logvar = linear_output[:, latent_dim:]
    return (mean, logvar)

def reparameterize(encoder_out):
    mean, logvar = encoder_out
    epsilon = np.random.randn(*mean.shape)
    z = mean + np.exp(0.5 * logvar) * epsilon
    return z

def custom_decoder(latent, node):
    """
    Reconstructs the input from the latent vector.
    """
    linear_output = np.dot(latent, node.weight) + node.bias
    if hasattr(node, 'activate') and callable(node.activate):
        return node.activate(linear_output)
    else:
        return linear_output

# ----- Loss Functions -----

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def kl_divergence(p, q):
    # Placeholder KL divergence function.
    return -np.sum(p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q)))

# ----- Placeholder Gradient Function -----
# Replace this with your actual gradient computation mechanism.
def compute_gradients(loss, weight, bias):
    # For illustration purposes, return zeros of the same shape.
    return np.zeros_like(weight), np.zeros_like(bias)

# ----- Image I/O Functions -----

def open_img(file_name):
    return np.array(Image.open(file_name))

def save_img(image, file_name):
    img_out = Image.fromarray(image)
    img_out.save(file_name)
    print("Reconstructed image saved as", file_name)

# ----- Training and Generation Functions -----

def generate(image):
    H, W, C = image.shape
    output_flat_patches = nIO(image)
    print("Output patches shape:", output_flat_patches.shape)
    reconstructed_img = reconstruct_image(output_flat_patches, H, W, patch_size, C)
    print("Reconstructed image shape:", reconstructed_img.shape)
    return (reconstructed_img * 255).astype(np.uint8)

def init_coders(image):
    H, W, C = image.shape
    input_dim_per_patch = patch_size * patch_size * C
    # Initialize encoder parameters: shape (input_dim_per_patch, 2 * latent_dim)
    encoder_node.weight = np.random.randn(input_dim_per_patch, 2 * latent_dim).astype(np.float32) * 0.01
    encoder_node.bias = np.zeros((2 * latent_dim,), dtype=np.float32)
    # Initialize decoder parameters: shape (latent_dim, input_dim_per_patch)
    decoder_node.weight = np.random.randn(latent_dim, input_dim_per_patch).astype(np.float32) * 0.01
    decoder_node.bias = np.zeros((input_dim_per_patch,), dtype=np.float32)

def train(input_img, desired_img=None, epochs=1000, lr=0.01):
    if desired_img is None:
        desired_img = input_img
    init_coders(input_img)
    
    # Create Adam optimizer instances for encoder and decoder parameters.
    enc_optimizer = Adam(learning_rate=lr)
    dec_optimizer = Adam(learning_rate=lr)
    
    for epoch in range(epochs):
        new_img = generate(input_img)
        if epoch == 1 or epoch % 100 == 0:
            save_img(new_img, f"sample.{epoch}.png")
        
        # Compute losses (flatten images for MSE)
        recon_loss = mean_squared_error(desired_img.flatten(), new_img.flatten())
        latent_div = kl_divergence(desired_img.flatten() * 0.001 + 0.5,
                                   new_img.flatten() * 0.001 + 0.5)
        total_loss = recon_loss + latent_div
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Recon Loss: {recon_loss:.4f}, KL Div: {latent_div:.4f}, Total Loss: {total_loss:.4f}")
        
        # Compute gradients for encoder and decoder parameters.
        grad_enc_w, grad_enc_b = compute_gradients(total_loss, encoder_node.weight, encoder_node.bias)
        grad_dec_w, grad_dec_b = compute_gradients(total_loss, decoder_node.weight, decoder_node.bias)
        
        # Update parameters using Adam's update method.
        enc_optimizer.update(encoder_node.weight, grad_enc_w)
        enc_optimizer.update(encoder_node.bias, grad_enc_b)
        dec_optimizer.update(decoder_node.weight, grad_dec_w)
        dec_optimizer.update(decoder_node.bias, grad_dec_b)
        
    print("Training complete!")

# ----- Global Hyperparameters -----

patch_size = 64      # Adjust as needed.
latent_dim = 96      # Adjust latent dimension.

# ----- Build the pyNN Pipeline -----

nImageTokenizer = {
    "name": "image tokenizer/normalizer",
    "patch_size": (patch_size, patch_size),
    "term": lambda self, input: tokenize_and_normalize_image(input, self.patch_size)
}

nLatent = {
    "name": "latent",
    "term": lambda self, encoder_out: reparameterize(encoder_out)
}

nEncoder = {
    "name": "encoder",
    "term": lambda self, input: custom_encoder(input, self)
}

nDecoder = {
    "name": "decoder",
    "term": lambda self, z: custom_decoder(z, self)
}

nReconAct = {
    "name": "reconstruction activation",
    "term": lambda self, input: np_sigmoid(input)
}

nIO = Base(name="i/o node", term=lambda self, input: input)
token_node = nIO.connect(Base, nImageTokenizer)
encoder_node = token_node.connect(Base, nEncoder)
latent_node = encoder_node.connect(Base, nLatent)
decoder_node = latent_node.connect(Base, nDecoder)
decoder_node.connect(Base, nReconAct)  # Chain reconstruction activation

# ----- Main Execution -----

if __name__ == "__main__":
    img = open_img("DSC05222.JPG")
    print("Input image shape:", img.shape)
    
    # Begin training using Adam's gradient update mechanism.
    train(img, epochs=1000, lr=0.01)
    
    # Optionally, generate and save a final reconstructed image.
    final_img = generate(img)
    save_img(final_img, "final_reconstructed.png")

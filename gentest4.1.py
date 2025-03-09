from activate import *  # activation functions (including np_sigmoid)
from sANNd import *      # sANNd framework
from PIL import Image
import numpy as np
from optimizer import Adam  # Import your Adam optimizer class

# ----- Data Preprocessing Functions -----

def preprocess_img(parent, image):
    """
    Normalize an image and split it into flattened patches.
    
    Returns an array of flattened patches of shape 
    (num_patches, patch_area * channels).
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    parent.H, parent.W, parent.C = image.shape

    ph, pw = parent.patch_size
    patches = []

    # Extract only full patches.
    for i in range(0, parent.H - parent.H % ph, ph):
        for j in range(0, parent.W - parent.W % pw, pw):
            patch = image[i:i+ph, j:j+pw, :]
            patches.append(patch.flatten())
    
    return np.array(patches)

"""def process_image(input_image, width, height, channels):
    arr = np.array(input_image).astype(np.float32) / 255.0
    return arr"""

def reconstruct_image(flat_patches, original_height, original_width, patch_size, channels):
    """
    Reassemble flattened patches into an image.
    Reconstructs an image by arranging patches in a grid. If the original dimensions 
    aren’t multiples of patch_size, the output will be cropped.
    """
    ph, pw = patch_size

    rows = original_height // ph
    cols = original_width // pw

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

    return (reconstructed * 255).astype(np.uint8)

# ----- VAE Functions -----

def encoder(parent, input):
    """
    Performs a linear transformation on input (shape: (batch_size, input_dim))
    using node.weight and node.bias, then splits into mean and logvar.
    """
    linear_output = np.dot(input, parent.weight) + parent.bias
    mean = linear_output[:, :latent_dim]
    logvar = linear_output[:, latent_dim:]
    return (mean, logvar)

def reparameterize(encoder_out):
    mean, logvar = encoder_out
    epsilon = np.random.randn(*mean.shape)
    z = mean + np.exp(0.5 * logvar) * epsilon
    return z

def decoder(parent, latent):
    """
    Reconstructs the input from the latent vector.
    """
    linear_output = np.dot(latent, parent.weight) + parent.bias
    if hasattr(parent, 'activate') and callable(parent.activate):
        return parent.activate(linear_output)
    else:
        return linear_output

# ----- Loss Functions -----

def np_mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def kl_divergence(p, q):
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

def init_coders(image):
    H, W, C = image.shape
    input_dim_per_patch = patch_size * patch_size * C
    # Initialize encoder parameters: shape (input_dim_per_patch, 2 * latent_dim)
    encoder_node.weight = np.random.randn(input_dim_per_patch, 2 * latent_dim).astype(np.float32) * 0.5
    encoder_node.bias = np.zeros((2 * latent_dim,), dtype=np.float32)
    # Initialize decoder parameters: shape (latent_dim, input_dim_per_patch)
    decoder_node.weight = np.random.randn(latent_dim, input_dim_per_patch).astype(np.float32) * 0.5
    decoder_node.bias = np.zeros((input_dim_per_patch,), dtype=np.float32)

def train(input_img, desired_img=None, epochs=1000, lr=0.01):
    H, W, C = input_img.shape

    input_shaped = reconstruct_image(preprocess_img(io_node, input_img),  H, W, io_node.patch_size, C)
    save_img(input_shaped, f"sample.input.png")
    if desired_img is None:
        desired_img = input_shaped
    
    save_img(desired_img, f"sample.desired.png")

    init_coders(input_img)
    
    # Create Adam optimizer instances for encoder and decoder parameters.
    enc_optimizer = Adam(learning_rate=lr)
    dec_optimizer = Adam(learning_rate=lr)
    
    for epoch in range(epochs):
        new_img = reconstruct_image(io_node(input_img),  H, W, io_node.patch_size, C)
        if epoch == 1 or epoch % 100 == 0:
            save_img(new_img, f"sample.{epoch}.png")
        
        # Compute losses (flatten images for MSE)
        recon_loss = np_mse_loss(desired_img.flatten(), new_img.flatten())
        latent_div = kl_divergence(desired_img.flatten() * 0.0005 + 0.6,
                                   new_img.flatten() * 0.0005 + 0.6)
        total_loss = recon_loss + latent_div
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Recon Loss: {recon_loss:.4f}, KL Div: {latent_div:.4f}, Total Loss: {total_loss:.4f}")
        
        # Compute gradients for encoder and decoder parameters.
        grad_enc_w, grad_enc_b = compute_gradients(total_loss, encoder_node.weight, encoder_node.bias)
        grad_dec_w, grad_dec_b = compute_gradients(total_loss, decoder_node.weight, decoder_node.bias)
        
        # Update parameters using Adam's update method.
        enc_optimizer.update(grad_enc_w, grad_enc_b)
        dec_optimizer.update(grad_dec_w, grad_dec_b)
        
    print("Training complete!")


# ----- Global Hyperparameters -----

patch_size = 16      # Adjust as needed.
latent_dim = 96      # Adjust latent dimension.

# ----- Build the sANNd Pipeline -----

#nImageTokenizer = {
nIO = {
    "name": "image reshaper",
    "H": None,
    "W": None,
    "C": None,
    "patch_size": (patch_size, patch_size),
    "input_term": preprocess_img
}

nLatent = {
    "name": "latent",
    "input_term": lambda self, encoder_out: reparameterize(encoder_out)
}

nEncoder = {
    "name": "encoder",
    "input_term": encoder
}

nDecoder = {
    "name": "decoder",
    "input_term": decoder
}

nReconAct = {
    "name": "reconstruction",
    "input_term": lambda self, input: np_sigmoid(input)
}

io_node = Base(nIO, train=False)
encoder_node = io_node.connect(Base, nEncoder)
latent_node = encoder_node.connect(Base, nLatent)
decoder_node = latent_node.connect(Base, nDecoder)
decoder_node.connect(Base, nReconAct)  # Chain reconstruction activation

# ----- Main Execution -----

if __name__ == "__main__":
    #img = open_img("DSC05222.JPG")
    img = open_img("DISK1CRP.jpg")

    print("Input image shape:", img.shape)
    
    # Begin training using Adam's gradient update mechanism.
    train(img, epochs=1000, lr=0.01)
    
    # Optionally, generate and save a final reconstructed image.
    final_img = reconstruct_image(io_node(img),  io_node.H, io_node.W, io_node.patch_size, io_node.C)
    save_img(final_img, "final_reconstructed.png")

from activate import *  # activation functions (including np_sigmoid)
from sANNd import *      # sANNd framework
from PIL import Image
import numpy as np
from optimizer import Adam  # Import your Adam optimizer class

# ----- Data Preprocessing Functions -----

def preprocess_img(parent, image):
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    parent.H, parent.W, parent.C = image.shape
    ph, pw = parent.patch_size
    patches = [image[i:i+ph, j:j+pw, :].flatten()
               for i in range(0, parent.H - parent.H % ph, ph)
               for j in range(0, parent.W - parent.W % pw, pw)]
    
    return np.array(patches)

def reconstruct_image(flat_patches, original_height, original_width, patch_size, channels):
    ph, pw = patch_size
    rows, cols = original_height // ph, original_width // pw
    patches = flat_patches.reshape(rows, cols, ph, pw, channels).transpose(0, 2, 1, 3, 4)
    return (patches.reshape(rows * ph, cols * pw, channels) * 255).astype(np.uint8)

# ----- VAE Functions -----

def encoder(parent, input):
    linear_output = np.dot(input, parent.weight) + parent.bias
    linear_output = linear_output[:, :latent_dim], linear_output[:, latent_dim:]

    parent.kl_loss = kl_divergence(linear_output[0], linear_output[1])

    return linear_output

def reparameterize(parent, encoder_out):
    mean, logvar = encoder_out
    parent.output = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape)

    return parent.output

def decoder(parent, latent):
    linear_output = np.dot(latent, parent.weight) + parent.bias
    return parent.activate(linear_output) if hasattr(parent, 'activate') else linear_output

# ----- Loss Functions -----

def np_mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def kl_divergence(mean, logvar):
    return -0.5 * np.sum(1 + logvar - np.square(mean) - np.exp(logvar))

# ----- Compute Gradients -----

def compute_gradients(loss, inputs, weights):
    d_loss = mse_loss_derivative(inputs, loss)  # Gradient of loss w.r.t input
    grad_w = np.dot(inputs.T, d_loss) / inputs.shape[0]  # Gradient w.r.t weights
    grad_b = np.mean(d_loss, axis=0)
    return grad_w, grad_b

# ----- Image I/O Functions -----

def open_img(file_name):
    return np.array(Image.open(file_name))

def save_img(image, file_name):
    Image.fromarray(image).save(file_name)
    print(f"Reconstructed image saved as {file_name}")

# ----- Training and Generation Functions -----

def init_coders(image):
    input_dim = patch_size * patch_size * image.shape[2]
    encoder_node.weight = np.random.randn(input_dim, 2 * latent_dim).astype(np.float32) * 0.5
    encoder_node.bias = np.zeros((2 * latent_dim,), dtype=np.float32)
    decoder_node.weight = np.random.randn(latent_dim, input_dim).astype(np.float32) * 0.5
    decoder_node.bias = np.zeros((input_dim,), dtype=np.float32)

def train(input_img, desired_img=None, epochs=1000, lr=0.01):
    H, W, C = input_img.shape

    input_shaped = reconstruct_image(preprocess_img(io_node, input_img), H, W, io_node.patch_size, C)
    save_img(input_shaped, "sample.input.png")

    desired_img = desired_img if desired_img is not None else input_shaped
    save_img(desired_img, "sample.desired.png")

    init_coders(input_img)
    enc_optimizer, dec_optimizer = Adam(lr), Adam(lr)

    for epoch in range(epochs):
        """encoded = encoder(encoder_node, preprocess_img(io_node, input_img))
        latent = reparameterize(encoded)
        decoded = decoder(decoder_node, latent)
        new_img = reconstruct_image(decoded, H, W, io_node.patch_size, C)"""

        #io_node.tap(encoder=lambda self, output: (output[0], output[1]))
        #io_node.tap(encoded, "encoder")
        #io_node.tap(self.ref(latent), "latent")
        new_img = reconstruct_image(io_node(input_img),  H, W, io_node.patch_size, C)

        if epoch % 100 == 0:
            save_img(new_img, f"sample.{epoch}.png")

        recon_loss = np_mse_loss(desired_img.flatten(), new_img.flatten())
        #kl_loss = kl_divergence(encoded[0], encoded[1])
        total_loss = recon_loss + encoder_node.kl_loss

        print(f"Epoch {epoch}: Recon Loss: {recon_loss:.4f}, KL Div: {encoder_node.kl_loss:.4f}, Total Loss: {total_loss:.4f}")

        grad_enc_w, grad_enc_b = compute_gradients(total_loss, preprocess_img(io_node, input_img), encoder_node.weight)
        grad_dec_w, grad_dec_b = compute_gradients(total_loss, latent_node.output, decoder_node.weight)

        enc_optimizer.update(encoder_node.weight, grad_enc_w)
        enc_optimizer.update(encoder_node.bias, grad_enc_b)
        dec_optimizer.update(decoder_node.weight, grad_dec_w)
        dec_optimizer.update(decoder_node.bias, grad_dec_b)

    print("Training complete!")

# ----- Global Hyperparameters -----
patch_size = 16
latent_dim = 96

# ----- Build the sANNd Pipeline -----
io_node = Base({"name": "image reshaper", "patch_size": (patch_size, patch_size), "input_term": preprocess_img}, train=False)
encoder_node = io_node.connect(Base, {"name": "encoder", "input_term": encoder})
latent_node = encoder_node.connect(Base, {"name": "latent", "input_term": reparameterize})
decoder_node = latent_node.connect(Base, {"name": "decoder", "input_term": decoder})
decoder_node.connect(Base, {"name": "reconstruction", "input_term": lambda self, input: np_sigmoid(input)})

if __name__ == "__main__":
    img = open_img("DISK1CRP.jpg")
    print("Input image shape:", img.shape)
    train(img, epochs=1000, lr=0.01)
    final_img = reconstruct_image(io_node(img), io_node.H, io_node.W, io_node.patch_size, io_node.C)
    save_img(final_img, "final_reconstructed.png")

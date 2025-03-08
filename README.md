# 🏗️ sANNd: The Sandbox for Neural Networks 🏖️  

### **A novel, iterator-based approach to deep learning**  
sANNd (**sandbox artificial neural network design**) is an **open-source machine learning framework** that takes a radically different approach:  
- **No computational graphs**  
- **No tensors**  
- **No rigid layer structures**  
Instead, sANNd is built **entirely on trainable iterators** that **flow data and gradients** through the network, just like sand through a child's hands in a sandbox.  

🔥 **sANNd lets you build and train deep learning models with simple, composable `Mould` units.**  

---

## **🚀 Why sANNd?**
✅ **Iterator-Based Learning** → No static graphs, just **flowing iterables**.  
✅ **Flexible & Modular** → Networks **compose like LEGO**, making **residuals, LSTMs, and CNNs trivial**.  
✅ **Efficient Backpropagation** → Uses **parent-linked `Moulds`** to propagate gradients **automatically**.  
✅ **Lightweight & Fast** → No deep dependency trees, just **pure Python and NumPy (soon JAX/CUDA support!)**.  

> _Imagine neural networks built like a child's sandbox:  
> The **buckets, sifters, and dump trucks** are your transformations,  
> The **sand grains** are your iterables,  
> The **pivoting hourglass** is your training pipeline._  

💡 **sANNd is an experimental playground for AI research.**  

---

## **🛠️ Install sANNd**
```sh
git clone https://github.com/GuruMoore/sANNd.git
cd sANNd
pip install -r requirements.txt
```

---

## **🎨 Example: A Simple Neural Network**
A **basic neural network** in sANNd is just a **chain of `Moulds`** that **modulate data flow**.  
Here’s how you can **define and train a simple model**:

```python
import random
import math
from sANNd import Mould

# Define activation functions
def scale(x, y):
    return x * y

def add(x, y):
    return x + y

def softplus(x):
    return math.log1p(math.exp(min(x, 50)))  # Prevent overflow

# Gradient functions
def compute_gradient(output, target):
    return [(o - t) * 0.01 for o, t in zip(output, target)]  # Simple derivative

def apply_gradient(grad, param, lr):
    return param - lr * grad  # Learning rate-based update

# Initialize Moulds
input_layer = [0.5]
hw = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
hb = Mould([0.0], func=lambda x: x, train_func=apply_gradient)
ow = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
ob = Mould([0.0], func=lambda x: x, train_func=apply_gradient)

target_output = [1.0348316875442132]

for epoch in range(2000):
    # Forward pass
    ha = Mould(hw, input_layer, func=scale, parent=hw)
    ha = Mould(hb, ha, func=add, parent=hb)
    ha = Mould(ha, func=softplus, parent=ha)
    ha_output = list(ha)

    final_output = list(Mould(ob, Mould(ow, ha_output, func=scale, parent=ow), func=add, parent=ob))

    # Compute loss and gradients
    loss = sum((o - t) ** 2 for o, t in zip(final_output, target_output)) / len(final_output)
    gradients = compute_gradient(final_output, target_output)

    # Apply gradients
    ha.train(gradients)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Output: {final_output}, Loss: {loss}")
```
🔥 **This is a fully functional trainable model—built without traditional layers!**  

---

## **📌 Residual Networks (ResNet) in One Line**
In sANNd, residual connections **are just `Moulds` with additive identity connections**:
```python
residual = Mould(prev_layer, transformed_layer, func=lambda h, f_h: h + f_h)
```
💡 **Residuals, skip connections, and complex architectures are now trivial!**

---

## **📌 LSTMs Are Just Recurrent `Moulds`**
Unlike other ML frameworks, **LSTMs don’t require special treatment**.  
Just define your **memory cell as a `Mould`**, and everything **flows naturally**.
```python
hidden, cell = Mould(input_data, prev_h, prev_c, w_h, w_x, w_c, b, func=lstm_cell)
```
💡 **Recurrent models are as simple as stacking Moulds!**

---

## **🤝 Contributing**
We **welcome contributions** to improve sANNd!  
1️⃣ **Fork the repo**  
2️⃣ **Create a new branch** (`feature-xyz`)  
3️⃣ **Submit a pull request** 🚀  

🔥 **Join the discussion!** Open an issue or start a GitHub Discussion.  

---

## **📚 Roadmap**
🛠 **Upcoming Features**:
- ✅ **Multi-Layer Architectures** (MLPs, CNNs)  
- ✅ **Gradient Clipping & Adaptive Learning Rates**  
- ✅ **JAX/CUDA Acceleration**  
- 🚀 **Transformer Support**  
- 🚀 **Meta-Learning with Differentiable Programming**  

> **🌍 Let's push AI research forward—together.**  
> If **information isn't free, then neither are we.**  

---

## **📜 License**
**MIT License** – Free to use, modify, and share.  

---

## **🌍 Join the sANNd Community**
📢 **Share your experiments, insights, and ideas!**  
💬 Twitter, Reddit, Hacker News, Dev.to, Medium  
🚀 **Let’s build something amazing together!**  

---
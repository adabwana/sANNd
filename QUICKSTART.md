# 🚀 sANNd Quickstart Guide  

Welcome to **sANNd (sandbox artificial neural network design)**!  
This guide will show you how to **quickly build and train models** using `Mould`.  

---

## **1️⃣ Install sANNd**
```sh
git clone https://github.com/GuruMoore/sANNd.git
cd sANNd
pip install -r requirements.txt
```

---

## **2️⃣ The Basics: Moulds are the Building Blocks**  

In sANNd, networks are built from **Moulds**—trainable, iterable units that flow data and gradients.  
### **🔹 Example: Simple Forward Pass**
```python
from sANNd import Mould

# Define a transformation function
def scale(x, y):
    return x * y

# Create Moulds
input_data = [0.5]
weight = Mould([2.0])  # Initial weight
output = Mould(weight, input_data, func=scale)  # Multiply weight * input

print(list(output))  # Output: [1.0]
```
🔥 **Each Mould transforms data as it flows!**

---

## **3️⃣ Training a Simple Model**
Let’s train a **basic neural network** to match a target output.

```python
import random
import math
from sANNd import Mould

# Activation & Gradient Functions
def scale(x, y): return x * y
def add(x, y): return x + y
def softplus(x): return math.log1p(math.exp(min(x, 50)))
def compute_gradient(output, target): return [(o - t) * 0.01 for o, t in zip(output, target)]
def apply_gradient(grad, param, lr): return param - lr * grad

# Initialize Moulds
input_layer = [0.5]
hw = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
hb = Mould([0.0], func=lambda x: x, train_func=apply_gradient)
ow = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
ob = Mould([0.0], func=lambda x: x, train_func=apply_gradient)

target_output = [1.0348]

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
🔥 **A fully trainable network—no static layers, just flowing iterators!**

---

## **4️⃣ Residual Connections**
sANNd makes **skip connections trivial**:
```python
residual = Mould(prev_layer, transformed_layer, func=lambda h, f_h: h + f_h)
```
💡 **ResNets work naturally in sANNd!**

---

## **5️⃣ Building an LSTM**
LSTMs **flow naturally** with `Mould`—just chain cell states together.

```python
hidden, cell = Mould(input_data, prev_h, prev_c, w_h, w_x, w_c, b, func=lstm_cell)
```
🔥 **Recurrent models are just different `Mould` chains!**

---

## **6️⃣ Next Steps**
📚 **Explore More**:
- 🔗 **[Full sANNd Documentation](https://github.com/GuruMoore/sANNd)**
- 🚀 **[GitHub Issues: Contribute Ideas & Features](https://github.com/GuruMoore/sANNd/issues)**
- 💬 **Join the Discussion: [sANNd GitHub Discussions](https://github.com/GuruMoore/sANNd/discussions)**  

💡 **Start experimenting and shape the future of ML with sANNd!**

---

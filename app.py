import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from model import CVAE  # Guarda tu modelo en model.py

# Load model
device = torch.device('cpu')
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pt", map_location=device))
model.eval()

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit to generate", list(range(10)))

# One-hot encoding
y = torch.zeros(1, 10).repeat(5, 1)
y[range(5), digit] = 1.0

# Sample latent space
z = torch.randn(5, model.z_dim)
with torch.no_grad():
    samples = model.decode(z, y).view(-1, 1, 28, 28)

# Show images
grid = make_grid(samples, nrow=5)
npimg = grid.numpy().transpose(1, 2, 0)

st.image(npimg, caption=f"Generated samples for digit {digit}", width=500)

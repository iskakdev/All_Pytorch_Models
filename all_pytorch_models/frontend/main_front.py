import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import requests
from all_pytorch_models.frontend.mnist_front import check_number
from all_pytorch_models.frontend.fashion_mnist_front import check_fashion
from all_pytorch_models.frontend.cifar10_front import check_cifar


with st.sidebar:
    name = st.radio(label='All Models', options=['MNIST Project', 'Fashion MNIST Project', 'CIFAR10 Project'])

if name == 'MNIST Project':
    check_number()
elif name == 'Fashion MNIST Project':
    check_fashion()
elif name == 'CIFAR10 Project':
    check_cifar()

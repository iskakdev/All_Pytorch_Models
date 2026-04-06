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
from all_pytorch_models.frontend.flowers_front import check_flowers
from all_pytorch_models.frontend.food_and_coffee_front import check_food_and_coffee
from all_pytorch_models.frontend.cifar_100_front import check_cifar_100
from all_pytorch_models.frontend.image_scene_front import check_image_scene
from all_pytorch_models.frontend.trash_front import check_trash_image


with st.sidebar:
    name = st.radio(label='All Models', options=['MNIST Project', 'Fashion MNIST Project', 'CIFAR10 Project',
                                                 'Flowers Project', 'Food and Coffee Project', 'CIFAR100 Project',
                                                 'ImageScene Project', 'Trash Project'])

if name == 'MNIST Project':
    check_number()
elif name == 'Fashion MNIST Project':
    check_fashion()
elif name == 'CIFAR10 Project':
    check_cifar()
elif name == 'Flowers Project':
    check_flowers()
elif name == 'Food and Coffee Project':
    check_food_and_coffee()
elif name == 'CIFAR100 Project':
    check_cifar_100()
elif name == 'ImageScene Project':
    check_image_scene()
elif name == 'Trash Project':
    check_trash_image()

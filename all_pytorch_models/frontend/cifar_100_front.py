import streamlit as st
import requests
from PIL import Image
import io

def check_cifar_100():
    api_url = 'http://127.0.0.1:8001/cifar_100_predict/predict/'

    st.title('CIFAR100 Project')
    st.write('Загрузите изображение')

    uploaded_file = st.file_uploader('Загрузите изображение: ', type=['png', 'jpg', 'jpeg', 'jpd', 'webp'])
    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
               'boy', 'bridge', 'bus', 'butterfly',
               'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
               'couch', 'crab', 'crocodile', 'cup',
               'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
               'keyboard', 'lamp', 'lawn_mower',
               'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid',
               'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road',
               'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
               'squirrel', 'streetcar', 'sunflower',
               'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale',
               'willow_tree', 'wolf', 'woman', 'worm']

    if uploaded_file is not None:
         image = Image.open(uploaded_file)
         st.image(image, caption='Загруженное изображение')

         if st.button('Определить изображение'):
             try:
                uploaded_file.seek(0)
                file = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                answer = requests.post(api_url, files=file)

                if answer.status_code == 200:
                    result = answer.json()
                    st.success(f'Модель думает, что это: {result["Модель думает, что это: "]}')
                else:
                    st.error(f'Ошибка {answer.status_code}')
             except requests.exceptions.RequestException:
                 st.error('Ошибка, не удалось подключиться к API')

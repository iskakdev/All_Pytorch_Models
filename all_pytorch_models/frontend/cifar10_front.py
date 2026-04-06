import streamlit as st
import requests
from PIL import Image
import io

def check_cifar():
    api_url = 'http://127.0.0.1:8001/cifar_predict/cifar10_predicted/'

    st.title('CIFAR10 Project')
    st.write('Загрузите изображение')

    uploaded_file = st.file_uploader('Выберите изображение: ', type=['png', 'jpg', 'jpeg', 'jpd', 'webp'])
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
                    st.success(f'Модель думает, что это: {result["Predict"]}')
                else:
                    st.error(f'Ошибка: {answer.status_code}')
            except requests.exceptions.RequestException:
                st.error('Ошибка, не удалось подключиться к API')

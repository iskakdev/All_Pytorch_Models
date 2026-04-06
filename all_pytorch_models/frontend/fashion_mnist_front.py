import streamlit as st
import requests
from PIL import Image
import io

def check_fashion():
    api_url = 'http://127.0.0.1:8001/fashion_mnist_predict/fashion_predicted/'

    st.title('Fashion MNIST Project')
    st.write('Загрузите изображение')

    uploaded_file = st.file_uploader('Выберите изображение:', type=['png', 'jpg', 'jpeg', 'jpd'])
    words = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
             7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

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

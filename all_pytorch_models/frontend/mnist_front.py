import streamlit as st
import requests
from PIL import Image
import io
import os

def check_number():
    API_BASE = os.getenv('API_BASE_URL', 'http://127.0.0.1:8001')
    api_url = f'{API_BASE}/mnist_predict/predict/'

    st.title('MNIST Project')
    st.write('Загрузите изображение с цифрой')

    uploaded_file = st.file_uploader('Choice Image:', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение')

        if st.button('Send'):
            try:
                uploaded_file.seek(0)
                files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f'Predict: {result["Answer"]}')
                else:
                    st.error(f'Error: {response.status_code}')
            except requests.exceptions.RequestException:
                st.error(f'Fail to connect')

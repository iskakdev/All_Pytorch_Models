import streamlit as st
import requests
from PIL import Image
import io
import os

def check_image_scene():
    API_BASE = os.getenv('API_BASE_URL', 'http://127.0.0.1:8001')
    api_url = f'{API_BASE}/image_scene_predict/predict/'
    
    st.title('ImageScene Project')
    st.write('Загрузите изображение')

    uploaded_file = st.file_uploader('Загрузите изображение: ', type=['png', 'jpg', 'jpeg', 'jpd', 'webp'])
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

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

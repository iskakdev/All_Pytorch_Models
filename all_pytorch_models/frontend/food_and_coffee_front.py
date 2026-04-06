import streamlit as st
import requests
from PIL import Image
import io

def check_food_and_coffee():
    api_url = 'http://127.0.0.1:8001/food_and_coffee_predict/predicted/'

    st.title('Food and Coffee Project')
    st.write('Для того, чтобы модель определила загрузите изображение')

    uploaded_file = st.file_uploader('Загрузите изображение: ', type=['png', 'jpg', 'jpeg', 'jpd', 'webp'])
    foods = ['burger', 'coffee', 'dessert', 'fruit', 'pasta', 'pizza', 'salad', 'soup', 'steak', 'sushi']

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
                    st.success(f'Модель думает, что это: {result["Модель думает, что это"]}')
                else:
                    st.error(f'Ошибка: {answer.status_code}')
            except requests.exceptions.RequestException:
                st.error('Ошибка, не удалось подключиться к API')

from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os

fashion_mnist_predicted = APIRouter(prefix='/fashion_mnist_predict', tags=['Fashion MNIST Project'])

class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fashion_model = CheckImage()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'torch_ml_models', 'fashion_model.pth')
fashion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
fashion_model.to(device)
fashion_model.eval()

words = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

@fashion_mnist_predicted.post('/fashion_predicted/')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не добавлен')
        img = Image.open(io.BytesIO(image_data))
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = fashion_model(image_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {'Predict': words[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

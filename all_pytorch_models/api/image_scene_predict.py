from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from PIL import Image
import os

class ImageScenceVgg(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 8 * 8, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 6)
    )

  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image

transforms_data = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

image_scene_predicted = APIRouter(prefix='/image_scene_predict', tags=['ImageScene Project'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_scene_model = ImageScenceVgg()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'torch_ml_models', 'image_scence_model.pth')
image_scene_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
image_scene_model.eval()

@image_scene_predicted.post('/predict/')
async def check_image_scene(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не загружен')
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transforms_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = image_scene_model(image_tensor)
            predicted = y_pred.argmax(dim=1).item()
            return {'Модель думает, что это: ': classes[predicted]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

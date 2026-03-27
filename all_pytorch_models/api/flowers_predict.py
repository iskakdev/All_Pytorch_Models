from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

class FlowersVGG16(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
        nn.BatchNorm2d(2048),
        nn.ReLU(),
        nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
        nn.BatchNorm2d(2048),
        nn.ReLU(),
        nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
        nn.BatchNorm2d(2048),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048 * 4 * 4, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 5)
    )
  def forward(self, flower):
    flower = self.first(flower)
    flower = self.second(flower)
    return flower

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

flowers = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
flower_predicted = APIRouter(prefix='/flower_predict', tags=['Flowers Project'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flower_model = FlowersVGG16()
flower_model.load_state_dict(torch.load('torch_ml_models/flower_model.pth', map_location=device, weights_only=True))
flower_model.to(device)
flower_model.eval()


@flower_predicted.post('/flowers_predicted/')
async def check_flowers(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не добавлен')
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = flower_model(image_tensor)
            pred = y_pred.argmax(dim=1).item()
        return {'Модель думает, что это': flowers[pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

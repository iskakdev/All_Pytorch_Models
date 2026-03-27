from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

class CheckFood(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
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
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * 4 * 4, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 10)
    )

  def forward(self, food):
    food = self.first(food)
    food = self.second(food)
    return food

transform_data = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

foods = ['burger', 'coffee', 'dessert', 'fruit', 'pasta', 'pizza', 'salad', 'soup', 'steak', 'sushi']
food_and_coffee_predicted = APIRouter(prefix='/food_and_coffee_predict', tags=['Food and Coffee Project'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
food_and_coffee_model = CheckFood()
food_and_coffee_model.load_state_dict(torch.load('torch_ml_models/foods_model.pth', map_location=device, weights_only=True))
food_and_coffee_model.to(device)
food_and_coffee_model.eval()


@food_and_coffee_predicted.post('/predicted/')
async def check_food(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не загружен')
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = food_and_coffee_model(image_tensor)
            pred = y_pred.argmax(dim=1).item()

            return {'Модель думает, что это': foods[pred]}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

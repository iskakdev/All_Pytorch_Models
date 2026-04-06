from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from PIL import Image

class CheckTrashModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((32, 32))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transform_data = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

trash = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

trash_predicted = APIRouter(prefix='/trash_predict', tags=['Trash Project'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trash_model = CheckTrashModel()
trash_model.load_state_dict(torch.load('torch_ml_models/trash_model.pth', map_location=device, weights_only=True))
trash_model.to(device)
trash_model.eval()


@trash_predicted.post('/predict/')
async def cifar_100_check(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не загружен')
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = trash_model(image_tensor)
            predicted = y_pred.argmax(dim=1).item()
            return {'Модель думает, что это: ': trash[predicted]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import  APIRouter, HTTPException, UploadFile, File
import io
import uvicorn
import torch
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from PIL import Image
import os

class CIFAR100(nn.Module):
  def __init__(self):
    super().__init__()
    base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    self.features = base.features
    self.avgpool = base.avgpool

    for param in self.features.parameters():
      param.requires_grad = False

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 100)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier(x)
    return x

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly',
 'camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup',
 'dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
 'leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
 'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road',
 'rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower',
 'sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale',
 'willow_tree','wolf','woman','worm']

cifar_100_predicted = APIRouter(prefix='/cifar_100_predict', tags=['CIFAR100 Project'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cifar_100_model = CIFAR100()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'torch_ml_models', 'cifar_100_model.pth')
cifar_100_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
cifar_100_model.to(device)
cifar_100_model.eval()


@cifar_100_predicted.post('/predict/')
async def cifar_100_check(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не загружен')
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = cifar_100_model(image_tensor)
            predicted = y_pred.argmax(dim=1).item()
            return {'Модель думает, что это: ': classes[predicted]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

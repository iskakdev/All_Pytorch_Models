from fastapi import FastAPI
import uvicorn
from all_pytorch_models.api import (mnist_predict, fashion_mnist_predict, cifar10_predict)

all_pytorch_models_app = FastAPI(title='PyTorch Models')
all_pytorch_models_app.include_router(mnist_predict.mnist_predicted)
all_pytorch_models_app.include_router(fashion_mnist_predict.fashion_mnist_predicted)
all_pytorch_models_app.include_router(cifar10_predict.cifar_predicted)

if __name__ == '__main__':
    uvicorn.run(all_pytorch_models_app, host='127.0.0.1', port=8000)


# swa-tutorials-pytorch
[Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) Tutorials using pytorch. Based on [PyTorch 1.6 Official Features (Stochastic Weight Averaging)](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/), implement classification codebase using custom dataset. 

- author: hoya012  
- last update: 2020.10.20

## 0. Experimental Setup 
### 0-1. Prepare Library
- Need to install PyTorch and Captum

```python
pip install -r requirements.txt
```

### 0-2. Download dataset (Kaggle Intel Image Classification)

- [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification/)

This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }

- Make `data` folder and move dataset into `data` folder.

<p align="center">
  <img width="1200" src="/assets/data_folder.PNG">
</p>

### 1. Baseline Training 
- ImageNet Pretrained ResNet-18 from torchvision.models
- Batch Size 256 / Epochs 120 / Initial Learning Rate 0.0001
- Training Augmentation: Resize((256, 256)), RandomHorizontalFlip()
- Adam + Cosine Learning rate scheduling with warmup
- I tried NVIDIA Pascal GPU - GTX 1080 Ti 1 GPU

```python
python main.py --checkpoint_name baseline;
```

### 2. Stochastic Weight Averaging Training 

In PyTorch 1.6, Stochastic Weight Averaging is very easy to use! Thanks to PyTorch..

- PyTorch's official tutorial's guide
```python
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

loader, optimizer, model, loss_fn = ...
swa_model = AveragedModel(model)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
swa_start = 5
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(100):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()
      if epoch > swa_start:
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data 
preds = swa_model(test_input)
```

- My own implementations
```python
# in main.py
""" define model and learning rate scheduler for stochastic weight averaging """
swa_model = torch.optim.swa_utils.AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

...  

# in learning/trainer.py
if epoch > args.swa_start and args.decay_type == 'swa':
  self.swa_model.update_parameters(self.model)
  self.swa_scheduler.step()
else:
  self.scheduler.step()

...

# in main.py
swa_model = swa_model.cpu()
torch.optim.swa_utils.update_bn(train_loader, swa_model)
swa_model = swa_model.cuda() 
```

#### Run Script (Command Line)
```python
python main.py --checkpoint_name swa --decay_type swa --swa_start 90 --swa_lr 5e-5;
```

### 3. Performance Table
- B : Baseline
- SWA : Stochastic Weight Averaging
    - SWA_{swa_start}_{swa_lr}

|   Algorithm  | Test Accuracy |  
|:------------:|:-------------:|  
|      B       |      94.10    |  
|  SWA_90_0.05 |      80.53    |  
|  SWA_90_1e-4 |      94.20    |  
|  SWA_90_5e-4 |      93.87    |  
|  SWA_90_1e-5 |      94.23    |  
|  SWA_90_5e-5 |    **94.57**  |  
|  SWA_75_5e-5 |     Running   |  
|  SWA_60_5e-5 |     Running   |  

### 4. Code Reference
- Baseline Code: https://github.com/hoya012/carrier-of-tricks-for-classification-pytorch
- Gradual Warmup Scheduler: https://github.com/ildoonet/pytorch-gradual-warmup-lr
- PyTorch Stochastic Weight Averaging: https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging
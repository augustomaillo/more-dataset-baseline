from utils.metric_learning_generator_hard_sampling import MetricLearningGeneratorHardSampling
from utils.faster_generator import ClassificationGenerator
from utils.dataset import Dataset
from torch_model.metric_learning_model import Resnet50MetricLearning
from torch_utils.lr_scheduler import WarmupLR

from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from torch_utils.quadruplet_loss import QuadrupletLoss, MyQuadrupletLoss
from torch_utils.center_loss import CenterLoss
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import logging
import psutil

def train(
    dataset : Dataset,
    num_epochs,
    batch_size,
    weights_output=None,
    starting_weights=None,
    device='cpu',
    only_classification=False,
    quad_weight=0.5,
    center_weight=1e-2
    ):

    # metric learning train
    if weights_output is None:
        weights_output = f'weights_{datetime.now()}.pt'
        
    ident_num = dataset.ident_num('train')
    metric_learning_model = Resnet50MetricLearning(identities_num=ident_num, inference=False)
    if starting_weights is not None:
        print('Loading pre-trained weights')
        non_matched_keys = metric_learning_model.load_state_dict(torch.load(starting_weights))
        print(f'Non matched keys: {non_matched_keys}')
    metric_learning_model = metric_learning_model.to(device).train()
    

    steps = int(ident_num/(batch_size/2))
    print('Creating loader.')
    if only_classification:
        gen = ClassificationGenerator(
            dataset=dataset, 
            steps=steps*num_epochs,
            batch_size = batch_size,
            partition = 'train',
            aug = True,
            img_size = (256, 256),
            label_smoothing = True
        )
    else:
        gen = MetricLearningGeneratorHardSampling(
            dataset=dataset, 
            steps=steps*num_epochs,
            batch_size = batch_size,
            partition = 'train',
            aug = True,
            img_size = (256, 256),
            label_smoothing = True
        )


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    )
    
    ce_loss = nn.CrossEntropyLoss()
    quad = QuadrupletLoss(margin1=0.3, margin2=0.3)
    center = CenterLoss(num_classes=ident_num,feat_dim=2048,use_gpu=True)
    
    optimizer = torch.optim.Adam(list(metric_learning_model.parameters()) + list(center.parameters()), lr=1) # warmup learning rate will multiply this by an factor (the desired learning rate)
    lr_scheduler = WarmupLR(optimizer)
    
    bestloss = np.inf
    best_acc = 0
    print(f"Initial lr: {optimizer.param_groups[-1]['lr']}")
    
    chard_size = int(batch_size//4)
    
    epochs_with_no_improvement = 0
    for epoch in range(num_epochs):
        current_loss = 0
        current_quad = 0
        current_center = 0
        current_ce = 0
        seem = 0
        correct=0
        tqdm_bar = tqdm(range(steps), desc=f'Epoch [1/{num_epochs}], Loss: {bestloss:.4f} Acc: {best_acc:.4f}' , leave=True)
        loader = DataLoader(gen, num_workers=1, batch_size=1, pin_memory=True)
        loader_iter = iter(loader)
        for _  in tqdm_bar:
            optimizer.zero_grad()
        
            (x, y, indexes) = next(loader_iter)
            images = normalize(x[0]).to(device)
            labels = y[0].to(device)

            # Forward pass
            features, outputs = metric_learning_model(images)
            # Losses
            cls_loss = ce_loss(outputs, labels)
            torch.cuda.synchronize()
            if only_classification:
                quad_loss = torch.tensor(0) 
                center_loss = torch.tensor(0)
            else:
                quad_loss = quad(
                    anchor=features[:chard_size].squeeze(), # cam A
                    positive=features[chard_size: 2 * chard_size].squeeze(), # cam B
                    negative1=features[2*chard_size: 3*chard_size].squeeze(), # cam B neg 
                    negative2=features[3*chard_size: 4*chard_size].squeeze(), # cam A neg
                )
                
                center_loss = center(features.squeeze(), labels.squeeze().argmax(dim=1))
             
            total_loss = quad_loss*quad_weight + cls_loss + center_loss*center_weight


            gen.update_features(
                features.cpu().detach().numpy()[:, :, 0,0],
                indexes=indexes
            )
            
            
            torch.cuda.synchronize()
            total_loss.backward()
            for param in center.parameters():
                param.grad.data *= 1/ (1+center_weight)
            optimizer.step()
            torch.cuda.synchronize()
            
            # Imprima as métricas de treinamento
            seem += batch_size
            correct+= sum(labels.argmax(1) == outputs.argmax(1)).item()
            current_acc = correct/seem
            current_loss += total_loss.item()
            current_quad += quad_loss.item()
            current_center += center_loss.item()
            current_ce += cls_loss.item()
            
            step = seem//batch_size
            tqdm_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss/step:.4f} (CE: {current_ce/step:.4f} QuadLoss: {current_quad/step:.4f} CenterLoss: {current_center/step:.4f}) Acc: {current_acc:.4f}')
        lr_scheduler.step()
        
        current_loss = current_loss/steps
        if only_classification:
            if current_acc > best_acc:
                print(f'Saving! Improved from {best_acc} to {current_acc}')
                torch.save(metric_learning_model.state_dict(), weights_output)
                bestloss = current_loss
                best_acc = current_acc
        else:
            if current_loss < bestloss:
                print(f'Saving! Improved from {bestloss} to {current_loss}')
                torch.save(metric_learning_model.state_dict(), weights_output)
                bestloss = current_loss
                best_acc = current_acc
                epochs_with_no_improvement = 0
            else:
                epochs_with_no_improvement +=1
                if epochs_with_no_improvement >= 20:
                    print('Loss not improved for the last 20 epochs. Early stopping.')
                    break
        
        gen.update_distances()
        gen.evaluate_on_train()
        print('Distances updated.')
        
        torch.cuda.empty_cache()
        if only_classification and current_acc >= 1.0:
            return
from model.Quadruplet import quadruplet_net
from model.Triplet import triplet_net
from model.AuxLayers import embedding, classification_layer, classification_net

from utils.dataset import Dataset

# from model.moto_ResNet50 import feat_net as std_feat_net
from model.modified_resnet50 import ResNet50 as modified_feat_net

models_folder = 'saved_models'

import numpy as np
import os 

np.random.seed(2301)

TrainData = Dataset('datasets/train_23_04.hdf5')
# TestData = Dataset('test.hdf5')

feat_model = modified_feat_net((224,320,3), TrainData.ident_num()) # BASELINE
feat_model.load_weights(os.path.join(models_folder,'pesos_RESNET50_BASELINE.h5'), by_name=True) 
feat_model.summary()
identity_model = classification_layer(TrainData.ident_num(), bias = False)


# quadrupletNet = quadruplet_net(feat_model, identity_model, (224,320))

clsNet  = classification_net(feat_model, identity_model, (224,320))

tripletNet = triplet_net(feat_model, identity_model, (224,320))

from utils.Trainer import train_classifier, train_trinet, train_quadnet

train_classifier(
    feat_model,
    clsNet,
    'datasets/train_23_04.hdf5',
    'datasets/val_23_04.hdf5',
    'cls_newdiv_23_04.hdf5',
    train_flag = False,
    epochs = 120
)

train_trinet(
    feat_model, 
    tripletNet, 
    'datasets/train_23_04.hdf5',
    'datasets/val_23_04.hdf5',
    'trinet_newdiv_23_04_modified.hdf5',  
    epochs = 120,
    batch_size = 6
)



from model.AuxLayers import classification_layer_baseline, classification_net
from utils.dataset import Dataset
from model.Resnet50_LastStride1 import ResNet50 as ResNet50_LastStride
from utils.Trainer import train_classifier, train_trinet, train_quadnet, train_msml

from model.general_model import general_net_center 
from model.general_model import centernet

import numpy as np
import os 
np.random.seed(2301)

models_folder = ''

MoRe_path = "MoRe.hdf5"
MoRe = Dataset(MoRe_path, to_bgr = True) # Images were used in BGR format

INPUT_SHAPE = (256, 256)

feat_model = ResNet50_LastStride(input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], 3)) # BASELINE STRIDE = 1

# you should instantie a Resnet50 keras model and save its weights to 'RESNET50_ORIGINAL_WEIGHTS.h5'
# feat_model.load_weights(os.path.join(models_folder,'RESNET50_ORIGINAL_WEIGHTS.h5'), by_name=True)
feat_model._make_predict_function()

identity_model = classification_layer_baseline(MoRe.ident_num('train'), BN = True) # Loading BN
clsNet  = classification_net(feat_model, identity_model, img_shape = INPUT_SHAPE)

train_classifier(
    feat_model = feat_model,
    model = clsNet,
    train_dataset = MoRe,
    modelpath =  'tsting_cls.hdf5',
    models_folder = models_folder,
    epochs = 1,
    batch_size = 32, 
    img_size = INPUT_SHAPE,
    label_smoothing = True,
    wlr = True,
    BN = True
)

# after training cls...
# clsNet.load_weights(os.path.join(models_folder, 'tsting_cls.hdf5'))

centernet = centernet(MoRe.ident_num('train'), 2048)

gen_model = general_net_center(feat_model, identity_model, centernet, INPUT_SHAPE)

train_trinet(
    feat_model = feat_model,
    model = gen_model,
    train_dataset = "MoRe.hdf5",
    modelpath = 'tsting_ML.hdf5',
    models_folder = models_folder,
    epochs = 1,
    batch_size = 6, 
    img_size = INPUT_SHAPE,
    aug = True,
    label_smoothing = True,
    wlr = True,
    BN = True, 
    CenterLoss = True
)

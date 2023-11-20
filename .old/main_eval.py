from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from model.AuxLayers import classification_layer_baseline, classification_net
from model.general_model import general_net_center, centernet
from utils.dataset import Dataset
from model.Resnet50_LastStride1 import ResNet50 as ResNet50_LastStride
from utils.Evaluator import EvalModel
import tensorflow as tf
import numpy as np
import os
np.random.seed(2301)

models_folder = ''

MoRe_path = "MoRe.hdf5"
MoRe = Dataset(MoRe_path, to_bgr = True) # Images were used in BGR format

INPUT_SHAPE = (256, 256)

feat_model = ResNet50_LastStride(input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], 3)) # BASELINE STRIDE = 1
identity_model = classification_layer_baseline(MoRe.ident_num('train'), BN = True) # Loading BN
centernet = centernet(MoRe.ident_num('train'), 2048)

gen_model = general_net_center(feat_model, identity_model, centernet, INPUT_SHAPE)

gen_model.load_weights('sanity.hdf5')

eval_model = EvalModel(MoRe, partition='test', image_shape=(256,256))
result = eval_model.compute(feat_model=feat_model)






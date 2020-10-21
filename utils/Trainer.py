from utils.dataset import Dataset
# from utils.generator import classification_generator, triplet_generator, quadruplet_generator
from utils.generator import general_generator, general_generator_center, classification_generator
from utils.test_metrics import generate_cmc_curve
from tensorflow.keras.applications.resnet import preprocess_input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import Model
from keras import optimizers
import tensorflow as tf
import os
from time import time
from shutil import copyfile
import numpy as np 
from PIL import Image
from multiprocessing.pool import ThreadPool
from keras.callbacks import Callback
from model.Quadruplet import quadruplet_loss

class TrainProgress(tf.keras.callbacks.Callback):
  def __init__(self, feat_model, folder, prefix, dataset, image_shape, bn):
    print('Loading train data on RAM.')
    self._folder = folder
    self._prefix = prefix
    self._dataset = dataset
    self._fm = feat_model
    self._image_shape = image_shape
    self._bn = bn

    test_dataA = self._dataset.content_array('camA')[0]
    test_dataB = self._dataset.content_array('camB')[0]
    test_identA = self._dataset.content_array('camA')[1]
    test_identB = self._dataset.content_array('camB')[1]
    self._camA_set = []
    self._camB_set = []
    
    def process_image(img_name):
      imgA = dataset.get_image(img_name)
      imgA = Image.fromarray(imgA)
      imgA = imgA.resize((image_shape[1], image_shape[0]))

      imgA = np.array(imgA)
      imgA = preprocess_input(imgA)
      imgA /= 255
      return imgA

    processA = ThreadPool(4).map(process_image, test_dataA)
    for r in processA:
      self._camA_set.append(r)
      print(len(self._camA_set),len(test_dataA), end='\r')
    self._camA_set = np.array(self._camA_set)

    processB = ThreadPool(4).map(process_image, test_dataB)
    for r in processB:
      self._camB_set.append(r)
      print(len(self._camB_set),len(test_dataB), end='\r')
    self._camB_set = np.array(self._camB_set)

    self.ids_camA = np.array(test_identA)
    self.ids_camB = np.array(test_identB)

  def on_epoch_end(self, epoch, logs=None):
    if (epoch+1)%20 ==0:
      if(not os.path.exists(self._folder)):
          os.mkdir(self_folder)
          
      
      new_generate_cmc_curve(
        self._fm,
        self._camA_set,
        self._camB_set,
        self.ids_camA,
        self.ids_camB,
        self._dataset, 
        name = os.path.join(self._folder, self._prefix+'_epoch_%d'%epoch),
        image_size = self._image_shape,
        BN = self._bn
        )

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


def step_decay(epoch):
  0.000035
  lrate = 0.00035
  if epoch > 40:
    lrate = 0.000035

  if epoch > 70:
    lrate = 0.0000035

  return lrate

def warmup_lr(epoch):
  epoch = epoch+1
  if epoch <= 10:
    lrate = 3.5e-5*(epoch)
  elif epoch <= 40:
    lrate = 3.5e-4
  elif epoch <= 70:
    lrate = 3.5e-5
  elif epoch >70:
    lrate = 3.5e-6
  return lrate

def standard_optimizer():
  return optimizers.Adam(lr=0.00035)

def copy_dataset(dataset):
  dataset_folder = 'datasets'
  out_name = os.path.join(dataset_folder, 'temp_%s.hdf5'%str(time())[-4:-1])
  copyfile(dataset, out_name)
  return out_name

def del_temp_dataset(dataset):
  os.remove(dataset)

def train_classifier(
      feat_model,
      model, 
      dataset,
      modelpath,  
      models_folder='saved_models', 
      epochs = 120, 
      batch_size = 32, 
      img_size = (224,320), 
      label_smoothing = False,
      wlr = False,
      BN = False
    ):
    
  model.compile(
    optimizer = standard_optimizer(), 
    loss=['categorical_crossentropy'], 
    metrics = ['acc']
    )

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Loading Classifier')
    model.load_weights(modelpath)

  checkpoint = ModelCheckpoint(modelpath, monitor='acc', verbose=1, save_best_only=True, mode='max')

  if wlr:
    print('Using Warmup learning rate.')
    lrate = LearningRateScheduler(warmup_lr)
  else:
    lrate = LearningRateScheduler(step_decay)

  callbacks_list = [lrate, checkpoint, TerminateOnBaseline(monitor='acc', baseline=1.0)]

  cls_gen = classification_generator(
      dataset, 
      batch_size = batch_size,
      partition = 'train',
      aug = True,
      img_size = img_size,
      label_smoothing = label_smoothing
    )

  H = model.fit_generator(
                          cls_gen,
                          steps_per_epoch = int(train_dataset.ident_num('train')/(batch_size/2)), 
                          epochs = epochs,
                          callbacks = callbacks_list,
                          use_multiprocessing = True,
                          workers =  8 # need to be adjusted
                          )




def train_quadnet(feat_model, 
  model, 
  train_dataset, # object
  modelpath, 
  train_flag = True, 
  val_dataset = None,
   aug = True, 
   models_folder='saved_models', 
   epochs = 120, 
   batch_size = 12, 
   img_size = (224,320), 
   label_smoothing = False, 
   wlr = False,
   BN = False,
   CenterLoss = False
   ):

  losses_list = [quadruplet_loss(),'categorical_crossentropy']
  if CenterLoss:
    print('Using CenterLoss')
    losses_list.append(lambda y_true, y_pred: y_pred)
    losses_list.append(lambda y_true, y_pred: y_pred)

  model.compile(
    optimizer=standard_optimizer(),
    loss= losses_list
    )

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Quadnet carregada.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    if wlr:
      print('Using warmup lr')
      lrate = LearningRateScheduler(warmup_lr)
    else:
      lrate = LearningRateScheduler(step_decay)

    # TrainProgress(feat_model, 'curvas', 'quadruplet', train_dataset, img_size, BN)
    callbacks_list = [lrate, checkpoint]



    if not CenterLoss:
      quad_gen = general_generator(
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )
    else:
        quad_gen = general_generator_center( 
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )

    H = model.fit_generator(
                              quad_gen,
                              steps_per_epoch=int(train_dataset.ident_num()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list,
                              use_multiprocessing=True,
                              workers = 8
                            )

from model.Triplet import triplet_loss

def train_trinet(
  feat_model, 
  model, 
  train_dataset, 
  modelpath,
  val_dataset = None, 
  train_flag = True, 
  aug = True, 
  models_folder='saved_models', 
  epochs = 120, 
  batch_size = 4, 
  img_size = (224,320), 
  label_smoothing = False, 
  wlr = False, 
  BN = False,
  CenterLoss = False
  ):

  losses_list = [triplet_loss(),'categorical_crossentropy']
  if CenterLoss:
    print('Using CenterLoss')
    losses_list.append(lambda y_true, y_pred: y_pred)
    losses_list.append(lambda y_true, y_pred: y_pred)

  model.compile(
    optimizer=standard_optimizer(), 
    loss=losses_list
    )

  train_dataset = Dataset(train_dataset, to_rgb = False)

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Trinet carregada.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    if wlr:
      print('Using warmup lr')
      lrate = LearningRateScheduler(warmup_lr)
    else:
      lrate = LearningRateScheduler(step_decay)

    callbacks_list = [lrate, TrainProgress(feat_model, 'curvas', 'triplet', train_dataset, img_size, bn=BN), checkpoint]


    if not CenterLoss:
      tri_gen = general_generator(
          feat_model, 
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )
    else:
        tri_gen = general_generator_center(
          feat_model, 
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )

    H = model.fit_generator(
                              tri_gen,
                              steps_per_epoch=int(train_dataset.ident_num()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list
                            )


from model.MSML import msml 
def train_msml(
  feat_model, 
  model, 
  train_dataset, 
  modelpath,
  val_dataset = None, 
  train_flag = True, 
  aug = True, 
  models_folder='saved_models', 
  epochs = 120, 
  batch_size = 4, 
  img_size = (224,320), 
  label_smoothing = False, 
  wlr = False,
  BN = False,
  CenterLoss = False
  ):

  losses_list = [msml(),'categorical_crossentropy']
  if CenterLoss:
    print('Using CenterLoss')
    losses_list.append(lambda y_true, y_pred: y_pred)
    losses_list.append(lambda y_true, y_pred: y_pred)

  model.compile(
    optimizer=standard_optimizer(),
    loss=losses_list, 
    )

  train_dataset = Dataset(train_dataset, to_rgb = False)

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('MSML carregada.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    if wlr:
      print('Using warmup lr')
      lrate = LearningRateScheduler(warmup_lr)
    else:
      lrate = LearningRateScheduler(step_decay)

     
    callbacks_list = [lrate, TrainProgress(feat_model, 'curvas', 'MSML', train_dataset, img_size, BN), checkpoint]

    if not CenterLoss:
      msml_gen = general_generator(
          feat_model, 
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )
    else:
        msml_gen = general_generator_center(
          feat_model, 
          train_dataset, 
          batch_size = batch_size, 
          img_size = img_size,
          aug = aug,
          label_smoothing = label_smoothing
        )

    H = model.fit_generator(
                              msml_gen,
                              steps_per_epoch=int(train_dataset.ident_num()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list,
                              workers = 1
                            )

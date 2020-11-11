from utils.dataset import Dataset
from utils.generator import general_generator, general_generator_center, classification_generator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import Model
from keras import optimizers
import tensorflow as tf
import os
import numpy as np 
from PIL import Image
from keras.callbacks import Callback
from model.Quadruplet import quadruplet_loss

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

    callbacks_list = [lrate, checkpoint]


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

     
    callbacks_list = [lrate, checkpoint]

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

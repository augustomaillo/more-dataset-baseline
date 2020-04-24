from utils.dataset import Dataset
from utils.generator import classification_generator, triplet_generator, quadruplet_generator
from model.Quadruplet import quadruplet_accuracy, quadruplet_loss
from model.Triplet import triplet_loss
from utils.test_metrics import generate_cmc_curve

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import Model
from keras import optimizers
import tensorflow as tf
import os
from time import time
from shutil import copyfile


class MyCustomCallback(tf.keras.callbacks.Callback):
  def __init__(self, feat_model, folder, prefix, dataset):
    self._folder = folder
    self._prefix = prefix
    self._dataset = dataset
    self._fm = feat_model

  def on_epoch_end(self, epoch, logs=None):
    
    if(not os.path.exists(self._folder)):
        os.mkdir(self_folder)
        
    
    generate_cmc_curve(self._fm,
                       self._dataset, 
                       name = os.path.join(self._folder, self._prefix+'_epoch_%d'%epoch) )


def step_decay(epoch):
  
  lrate = 0.00035
  if epoch > 40:
    lrate = 0.000035

  if epoch > 70:
    lrate = 0.0000035

  return lrate

def standard_optimizer():
  # return AdamOptimizer
  return optimizers.Adam(lr=0.00035)

def copy_dataset(dataset):
  dataset_folder = 'datasets'
  out_name = os.path.join(dataset_folder, 'temp_%s.hdf5'%str(time())[-4:-1])
  copyfile(dataset, out_name)
  return out_name

def del_temp_dataset(dataset):
  os.remove(dataset)


def train_classifier(feat_model, model, train_dataset, val_dataset, modelpath, train_flag = True, models_folder='saved_models', epochs = 120, batch_size = 32, img_size = (224,320)):
  model.compile(
    optimizer = standard_optimizer(), 
    loss=['categorical_crossentropy'], 
    metrics = ['acc']
    )

  train_data_name = copy_dataset(train_dataset)
  train_dataset = Dataset(train_data_name)

  val_data_name = copy_dataset(val_dataset)
  val_dataset = Dataset(val_data_name)


  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Classificador carregado.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    lrate = LearningRateScheduler(step_decay)

     
    callbacks_list = [lrate, MyCustomCallback(feat_model, 'curvas', 'cls', val_dataset), checkpoint]

    cls_gen = classification_generator(
        train_dataset, 
        batch_size = batch_size,
        aug = True,
        img_size = img_size,
        label_smoothing = True
      )

    H = model.fit_generator(
                              cls_gen,
                              steps_per_epoch=int(train_dataset.images_amount()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list
                            )

    del_temp_dataset(train_data_name)
    del_temp_dataset(val_data_name)


def train_quadnet(feat_model, model, train_dataset, val_dataset, modelpath, train_flag = True, models_folder='saved_models', epochs = 120, batch_size = 12, img_size = (224,320)):
  model.compile(
    optimizer=standard_optimizer(),
    loss=[quadruplet_loss(),'categorical_crossentropy'], 
    metrics = [quadruplet_accuracy]
    )

  train_data_name = copy_dataset(train_dataset)
  train_dataset = Dataset(train_data_name)

  val_data_name = copy_dataset(val_dataset)
  val_dataset = Dataset(val_data_name)

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Quadnet carregada.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='val_stacked_dists_quadruplet_accuracy', verbose=1, save_best_only=True, mode='max')
    lrate = LearningRateScheduler(step_decay)

     
    callbacks_list = [lrate, MyCustomCallback(feat_model, 'curvas', 'quadruplet', val_dataset), checkpoint]

    quad_gen = quadruplet_generator(
        feat_model, 
        train_dataset, 
        batch_size = batch_size, 
        img_size = img_size
      )

    val_quad_gen = quadruplet_generator(
        feat_model, 
        val_dataset, 
        batch_size = batch_size, 
        img_size = img_size,
        validation = True,
        val_ident_num = train_dataset.ident_num(),
        aug = False
      )

    H = model.fit_generator(
                              quad_gen,
                              steps_per_epoch=int(train_dataset.ident_num()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list,
                              validation_data = val_quad_gen,
                              validation_steps = int(val_dataset.ident_num()/batch_size)
                            )
    del_temp_dataset(train_data_name)
    del_temp_dataset(val_data_name)

def train_trinet(feat_model, model, train_dataset, val_dataset, modelpath, train_flag = True, models_folder='saved_models', epochs = 120, batch_size = 4, img_size = (224,320)):
  model.compile(
    optimizer=standard_optimizer(), 
    loss=[triplet_loss(),'categorical_crossentropy']
    )

  train_data_name = copy_dataset(train_dataset)
  train_dataset = Dataset(train_data_name)

  val_data_name = copy_dataset(val_dataset)
  val_dataset = Dataset(val_data_name)

  modelpath = os.path.join(models_folder, modelpath)
  if( os.path.exists(modelpath)):
    print('Trinet carregada.')
    model.load_weights(modelpath)

  if train_flag:
    checkpoint = ModelCheckpoint(modelpath, monitor='val_stacked_feats_loss', verbose=1, save_best_only=True, mode='min')
    lrate = LearningRateScheduler(step_decay)

     
    callbacks_list = [lrate, MyCustomCallback(feat_model, 'curvas', 'triplet', val_dataset), checkpoint]

    tri_gen = triplet_generator(
        feat_model,
        train_dataset, 
        batch_size = batch_size,
        img_size = img_size
      )

    val_tri_gen = triplet_generator(
        feat_model,
        val_dataset, 
        batch_size = batch_size,
        img_size = img_size,
        validation = True,
        val_ident_num = train_dataset.ident_num(),
        aug = False
      )

    H = model.fit_generator(
                              tri_gen,
                              steps_per_epoch=int(train_dataset.ident_num()/batch_size), 
                              epochs=epochs,
                              callbacks=callbacks_list,
                              validation_data = val_tri_gen,
                              validation_steps = int(val_dataset.ident_num()/batch_size)
                            )
    del_temp_dataset(train_data_name)
    del_temp_dataset(val_data_name)


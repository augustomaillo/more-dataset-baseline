import cv2
import copy
import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
from utils import data_augmentation
from keras.callbacks import ModelCheckpoint
import scipy.spatial.distance
from keras.models import Model
import random
from PIL import Image

def data_generator_classification(img_data, labels, ident_num, batch_size = 32, aug=1, img_size = (224,320)):
  
  c = list(zip(img_data, labels))
  np.random.shuffle(c)
  img_data, labels = zip(*c)
  
  while True:
    
    # images 
    img_batch = []
    labels_batch = []
    
    for img_file,lb in zip(img_data,labels):
        
        img = cv2.imread(img_file)
        img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
        if aug == 1:
          img = data_augmentation.augment(img,['zooming','translate','rotate','horizontal_flip','brightness','cutout'])
        


        # conjunto de imagens a serem consideradas
        img_batch.append(np.array(img))
        label_ = np.array(np_utils.to_categorical(lb,ident_num))
        
        ## LABEL SMOOTHING
        epsilon = 0.1
        label_[np.where(label_ == 0)] = epsilon/ident_num
        label_[np.where(label_ == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
        ####
        
        labels_batch.append(label_)
        # quando o conjunto de imagens é formado -> encontre os pares com dist. absolutas menores (negativas) e maiores (positivas)
        if len(img_batch) == batch_size:
          
          yield(np.array(img_batch),np.array(labels_batch)) 

          img_batch = []
          labels_batch = []


def data_generator(feat_model, data_camA, idA, data_camB, idB, ident_num, batch_size = 32, aug = 1, img_size = (224,320) ):
  
  # images 
  img_batch = []
  # ids
  p1 = []
  p2 = []
  n1 = []
  n2 = []
  
  while True:
    
    camA_set = []
    camB_set = []
    idA_set = []
    idB_set = []

    # embaralhando os dados para gerar conjuntos diferentes
    uIDs = np.unique(idA)
    random.shuffle(uIDs)

    posA = []
    posB = []

    for f in uIDs[0:5]:
      posA.extend(np.where(idA == f)[0])
      posB.extend(np.where(idB == f)[0])

    posA = np.array(posA)
    posB = np.array(posB)
    
    pidA = np.array(idA)[posA]
    pidB = np.array(idB)[posB]
    dataA = np.array(data_camA)[posA]
    dataB = np.array(data_camB)[posB]


    for pA, iA in zip(dataA, pidA):
      imgA = np.asarray(Image.open(pA)).copy()
      imgA = Image.fromarray(imgA)
      if aug ==1:
        imgA = data_augmentation.augment(imgA,['brightness', 'cutout', 'zooming', 'horizontal_flip', 'rotate', 'translate'])
      
#       print(type(imgA))
      imgA = imgA.resize((img_size[1], img_size[0]))
      # conjunto de imagens a serem consideradas
      camA_set.append( np.array(imgA)  )
      idA_set.append(iA)  
     

    for pB, iB in zip(dataB, pidB):
      imgB = np.asarray(Image.open(pB)).copy()
      imgB = Image.fromarray(imgB)
      if aug ==1:
        imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
      
      imgB = imgB.resize((img_size[1], img_size[0]))
      
      # conjunto de imagens a serem consideradas
      camB_set.append(np.array(imgB))
      idB_set.append(iB)  
    
 
    featA = feat_model.predict(np.array(camA_set))
    featB = feat_model.predict(np.array(camB_set))
    
    

    dist_m = scipy.spatial.distance.cdist(featA, featB)

    # apenas dist. de imagens das mesmas pessoas.
    out = np.zeros(dist_m.shape,dtype=bool)

    for cA,xA in enumerate(idA_set):
      for cB,xB in enumerate(idB_set):
        if xA == xB:
          out[cA,cB] = 1
   
    # positive elements with maximum distance
    
    hard_pos_id = np.where(dist_m == np.amax(dist_m[out==1]))
    hard_neg_id = np.where(dist_m == np.amin(dist_m[out==0]))

    # par de imagens positivo
    

    if(np.shape(hard_pos_id)[1] != 0 and np.shape(hard_neg_id)[1] != 0):
        pos1 = camA_set[int(hard_pos_id[0][0])]
        pos2 = camB_set[int(hard_pos_id[1][0])]
        pos1_id = idA_set[int(hard_pos_id[0][0])]
        pos2_id = idB_set[int(hard_pos_id[1][0])]


        neg1 = camA_set[int(hard_neg_id[0][0])]  
        neg2 = camB_set[int(hard_neg_id[1][0])]
        neg1_id = idA_set[int(hard_neg_id[0][0])]  
        neg2_id = idB_set[int(hard_neg_id[1][0])]

        # list of images
        img_batch.append([np.array(pos1), np.array(neg1), np.array(pos2), np.array(neg2)])

        #list of ids
        p1.append(np.array(np_utils.to_categorical(pos1_id,ident_num)))
        n1.append(np.array(np_utils.to_categorical(neg1_id,ident_num))) 
        p2.append(np.array(np_utils.to_categorical(pos2_id,ident_num)))
        n2.append(np.array(np_utils.to_categorical(neg2_id,ident_num))) 

        camA_set = []
        camB_set = []
        idA_set = []
        idB_set = []

        if len(img_batch) == batch_size:
          y_true = np.zeros(batch_size) # não tem significado, apenas para não dar erro

          img_batch = np.array(img_batch)

          pos_img1 = img_batch[:,0,:,:]
          neg_img1 = img_batch[:,1,:,:]
          pos_img2 = img_batch[:,2,:,:]
          neg_img2 = img_batch[:,3,:,:]

          yield([pos_img1, neg_img1, pos_img2, neg_img2],
                [y_true,np.stack([p1,n1,p2,n2],axis=1)])   

          img_batch = []
          p1 = []
          p2 = []
          n1 = []
          n2 = []

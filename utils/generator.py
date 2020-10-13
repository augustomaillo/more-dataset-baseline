import h5py
import numpy as np
from keras.utils import np_utils
from utils import data_augmentation
import random
from PIL import Image
import cv2
from utils.dataset import Dataset
import scipy.spatial.distance
from tensorflow.keras.applications.resnet import preprocess_input


def general_generator( 
    dataset, 
    batch_size = 4, 
    validation = False, 
    val_ident_num = 0, 
    aug = True, 
    img_size = (224,320), 
    label_smoothing = True
    ):

    data_camA = dataset.content_array('camA')[0]
    data_camB = dataset.content_array('camB')[0]
    idA = dataset.content_array('camA')[1]
    idB = dataset.content_array('camB')[1]
    ident_num = dataset.ident_num()

    if validation:
        idA = idA%ident_num
        idB = idB%ident_num
        label_smoothing = False

    while True:
    
        camA_set = []
        camB_set = []
        idA_set = []
        idB_set = []

        uIDs = np.unique(idA)
        random.shuffle(uIDs)
        posA = []
        posB = []

        for f in uIDs[0:batch_size]:
          posA.extend(
            np.random.permutation(
            np.where(idA == f)[0]
            )[:2]
            )
          posB.extend(
            np.random.permutation(
            np.where(idB == f)[0]
            )[:2]
            )

        posA = np.array(posA)
        posB = np.array(posB)

        pidA = np.array(idA)[posA]
        pidB = np.array(idB)[posB]
        dataA = np.array(data_camA)[posA]
        dataB = np.array(data_camB)[posB]
        
        

        for pA, iA in zip(dataA, pidA):
            imgA = dataset.get_image(pA)
            imgA = Image.fromarray(imgA)
            
            imgA = imgA.resize((img_size[1], img_size[0]))

            if aug:
                imgA = data_augmentation.augment(imgA,['brightness', 'cutout', 'zooming', 'horizontal_flip', 'rotate', 'translate'])

    #       print(type(imgA))
            # conjunto de imagens a serem consideradas
            imgA = np.array(imgA)
            imgA = preprocess_input(imgA)
            imgA /= 255
            camA_set.append(imgA)
            idA_set.append(np_utils.to_categorical(iA-1,ident_num))  



        for pB, iB in zip(dataB, pidB):
            
            imgB = dataset.get_image(pB)
            imgB = Image.fromarray(imgB)
            imgB = imgB.resize((img_size[1], img_size[0]))

            if aug:
                imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
           
            # conjunto de imagens a serem consideradas
            imgB = np.array(imgB)
            imgB = preprocess_input(imgB)
            imgB /= 255
            camB_set.append(imgB)
            idB_set.append(np_utils.to_categorical(iB-1,ident_num))  


        y_true = np.zeros(batch_size) # não tem significado, apenas para não dar erro
        
        if(len(camA_set) < len(camB_set)):
            for i in range(len(camB_set) - len(camA_set)):
                camA_set.append(camA_set[-1])
                idA_set.append(idA_set[-1])

        elif(len(camB_set) < len(camA_set)):
            for i in range(len(camA_set) - len(camB_set)):
                camB_set.append(camB_set[-1])
                idB_set.append(idB_set[-1])


        if len(np.shape(idA_set)) == 3:
            idA_set = np.squeeze(np.array(idA_set), axis=1)
            idB_set = np.squeeze(np.array(idB_set), axis=1)

        cls_idA_set = np.array(idA_set)
        cls_idB_set = np.array(idB_set)

        if(label_smoothing):
           ## LABEL SMOOTHING
            epsilon = 0.1
            cls_idA_set[np.where(cls_idA_set == 0)] = epsilon/ident_num
            cls_idA_set[np.where(cls_idA_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            
            cls_idB_set[np.where(cls_idB_set == 0)] = epsilon/ident_num
            cls_idB_set[np.where(cls_idB_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            ####
        
        if validation:
            cls_idA_set = np.zeros((cls_idA_set.shape[0], val_ident_num))
            cls_idB_set = np.zeros((cls_idB_set.shape[0], val_ident_num))

        
        yield([
                np.array(camA_set),
                np.array(camB_set)
            ],
            [
                np.stack(
                    [np.array(idA_set),
                    np.array(idB_set)],
                    axis=1),
                np.stack(
                    [cls_idA_set,
                    cls_idB_set],
                    axis=1)
            ])   

def general_generator_center(
    dataset, 
    batch_size = 4, 
    validation = False, 
    val_ident_num = 0, 
    aug = True, 
    img_size = (224,320), 
    label_smoothing = True
    ):
    data_camA = dataset.content_array('camA')[0]
    data_camB = dataset.content_array('camB')[0]
    idA = dataset.content_array('camA')[1]
    idB = dataset.content_array('camB')[1]
    ident_num = dataset.ident_num()


    if validation:
        idA = idA%ident_num
        idB = idB%ident_num
        label_smoothing = False


    while True:
    
        camA_set = []
        camB_set = []
        idA_set = []
        idB_set = []

        # idA_num = []
        # idB_num = []
        # embaralhando os dados para gerar conjuntos diferentes
        uIDs = np.unique(idA)
        random.shuffle(uIDs)
        posA = []
        posB = []

        for f in uIDs[0:batch_size]:
          posA.extend(
            np.random.permutation(
            np.where(idA == f)[0]
            )[:2]
            )
          posB.extend(
            np.random.permutation(
            np.where(idB == f)[0]
            )[:2]
            )

        posA = np.array(posA)
        posB = np.array(posB)

        pidA = np.array(idA)[posA]
        pidB = np.array(idB)[posB]
        dataA = np.array(data_camA)[posA]
        dataB = np.array(data_camB)[posB]
        
        for pA, iA in zip(dataA, pidA):
            imgA = dataset.get_image(pA)
            imgA = Image.fromarray(imgA)
            
            imgA = imgA.resize((img_size[1], img_size[0]))

            if aug:
                imgA = data_augmentation.augment(imgA,['brightness', 'cutout', 'zooming', 'horizontal_flip', 'rotate', 'translate'])

    #       print(type(imgA))
            # conjunto de imagens a serem consideradas
            imgA = np.array(imgA)
            imgA = preprocess_input(imgA)
            imgA /= 255
            camA_set.append(imgA)
            idA_set.append(np_utils.to_categorical(iA-1,ident_num))  
            # idA_num.append(iA-1)



        for pB, iB in zip(dataB, pidB):
            imgB = dataset.get_image(pB)
            imgB = Image.fromarray(imgB)
            imgB = imgB.resize((img_size[1], img_size[0]))

            if aug:
                imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
           
            # conjunto de imagens a serem consideradas
            imgB = np.array(imgB)
            imgB = preprocess_input(imgB)
            imgB /= 255
            camB_set.append(imgB)
            idB_set.append(np_utils.to_categorical(iB-1,ident_num))  
            # idB_num.append(iB-1)

        y_true = np.zeros(batch_size) # não tem significado, apenas para não dar erro
        
        if(len(camA_set) < len(camB_set)):
            for i in range(len(camB_set) - len(camA_set)):
                camA_set.append(camA_set[-1])
                idA_set.append(idA_set[-1])

        elif(len(camB_set) < len(camA_set)):
            for i in range(len(camA_set) - len(camB_set)):
                camB_set.append(camB_set[-1])
                idB_set.append(idB_set[-1])
                
        if len(np.shape(idA_set)) == 3:
            idA_set = np.squeeze(np.array(idA_set), axis=1)
            idB_set = np.squeeze(np.array(idB_set), axis=1)

        cls_idA_set = np.array(idA_set)
        cls_idB_set = np.array(idB_set)

        if(label_smoothing):
           ## LABEL SMOOTHING
            epsilon = 0.1
            cls_idA_set[np.where(cls_idA_set == 0)] = epsilon/ident_num
            cls_idA_set[np.where(cls_idA_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            
            cls_idB_set[np.where(cls_idB_set == 0)] = epsilon/ident_num
            cls_idB_set[np.where(cls_idB_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            ####
        
        if validation:
            cls_idA_set = np.zeros((cls_idA_set.shape[0], val_ident_num))
            cls_idB_set = np.zeros((cls_idB_set.shape[0], val_ident_num))

        
        yield([
                np.array(camA_set),
                np.array(camB_set),
                np.expand_dims(np.argmax(idA_set, axis=1), axis=1),
                np.expand_dims(np.argmax(idB_set, axis=1), axis=1)
            ],
            [
                np.stack(
                    [np.array(idA_set),
                    np.array(idB_set)],
                    axis=1),
                np.stack(
                    [cls_idA_set,
                    cls_idB_set],
                    axis=1),
                np.zeros(len(cls_idA_set)),
                np.zeros(len(cls_idA_set))
            ])   

from itertools import cycle
def classification_generator(
 dataset,
 batch_size = 32, 
 aug = True, 
 img_size = (224,320), 
 label_smoothing = True
 ):
    """Returns a generator that yields batchs for Classication Train

    Args:
            dataset_file: path to dataset hdf5 file
            batch_size: an integer specifying batch size
            aug: Boolean specifying data augmentation use
            img_size: tuple that contains image size to train the network, in the form (H, W)
            label_smoothing: Boolean specifying label smoothing use
    Returns:
            a generator object
    """  

    # def get_image(dataset_file, img_name):
    #     with h5py.File(dataset_file, 'r') as f:
    #         return f['images'][img_name].value, f['images'][img_name].attrs['ID']

    # with h5py.File(dataset_file, 'r') as f:
    #     all_train_ids = f['metadata']['train_ids'].value.tolist()

    data_camA = dataset.content_array('camA')[0]
    data_camB = dataset.content_array('camB')[0]
    idA = dataset.content_array('camA')[1]
    idB = dataset.content_array('camB')[1]
    all_images_names = np.concatenate([data_camA, data_camB])
    all_ids_list = np.concatenate([idA,idB])
    ident_num = dataset.ident_num()

    all_train_ids = np.unique(idA)

    np.random.shuffle(all_train_ids)
    pool = cycle(all_train_ids)
    
    current_ids = []
    while True:
        current_ids = []
        for i_, k in enumerate(pool):
            current_ids.append(k)
            if(i_ == (batch_size/2) - 1):
                break

        # images 
        img_batch = []
        labels_batch = []
        

        # yield( None)
        img_data = []
        labels_list = []
        for c_ in current_ids:
            s = np.random.choice(all_images_names[all_ids_list==c_],2)
            for s_ in s:
                    img_data.append(s_)
                    labels_list.append(c_)

        # print(img_data, labels_list)




        for img_file, lb in zip(img_data, labels_list):
                img = dataset.get_image(img_file)
                img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
                if aug:
                    img = data_augmentation.augment(img,['zooming','translate','rotate','horizontal_flip','brightness','cutout'])
            
                img = np.array(img)
                img = preprocess_input(img)
                img /= 255
                # conjunto de imagens a serem consideradas
                img_batch.append(img)
                label_ = np.array(np_utils.to_categorical(lb-1,ident_num))
                
                ## LABEL SMOOTHING

                if label_smoothing:
                    epsilon = 0.1
                    label_[np.where(label_ == 0)] = epsilon/ident_num
                    label_[np.where(label_ == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
                ####
                

                labels_batch.append(label_)
                
                if len(img_batch) == batch_size:

                    if len(np.shape(labels_batch))==3:

                        out = np.squeeze(np.array(labels_batch), axis=1)
                    else:
                        out = np.array(labels_batch)
                    yield(np.array(img_batch), out)

                    img_batch = []
                    labels_batch = []
import h5py
import numpy as np
# from keras.utils import np_utils
from utils import data_augmentation
import random
from PIL import Image
import cv2
from utils.dataset import Dataset
from tensorflow.keras.applications.resnet import preprocess_input
from itertools import cycle

def general_generator( 
    dataset, 
    batch_size = 4, 
    partition = 'train',  
    aug = True, 
    img_size = (224,320), 
    label_smoothing = True
    ):

    data_camA, idA = dataset.content_array(partition =partition, cam_name ='camA')
    data_camB, idB = dataset.content_array(partition =partition, cam_name ='camB')
    ident_num = dataset.ident_num(partition)

    if partition != 'train':
        idA = idA%ident_num
        idB = idB%ident_num
        label_smoothing = False
        aug = False

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

            imgA = np.array(imgA)
            imgA = preprocess_input(imgA)
            imgA /= 255
            camA_set.append(imgA)
            idA_set.append(np.eye(ident_num)[iA-1]) # (np_utils.to_categorical(iA-1,ident_num))  



        for pB, iB in zip(dataB, pidB):
            imgB = dataset.get_image(pB)
            imgB = Image.fromarray(imgB)
            imgB = imgB.resize((img_size[1], img_size[0]))

            if aug:
                imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
           
            imgB = np.array(imgB)
            imgB = preprocess_input(imgB)
            imgB /= 255
            camB_set.append(imgB)
            idB_set.append(np.eye(ident_num)[iB-1]) #np_utils.to_categorical(iB-1,ident_num))  


        y_true = np.zeros(batch_size) # no meaning, just to fit batch shape
        
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
            epsilon = 0.1
            cls_idA_set[np.where(cls_idA_set == 0)] = epsilon/ident_num
            cls_idA_set[np.where(cls_idA_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            
            cls_idB_set[np.where(cls_idB_set == 0)] = epsilon/ident_num
            cls_idB_set[np.where(cls_idB_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
        
        if partition != 'train':
            cls_idA_set = np.zeros((cls_idA_set.shape[0], itent_num))
            cls_idB_set = np.zeros((cls_idB_set.shape[0], itent_num))

        
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
    partition = 'train',  
    aug = True, 
    img_size = (224,320), 
    label_smoothing = True
    ):

    data_camA, idA = dataset.content_array(partition =partition, cam_name ='camA')
    data_camB, idB = dataset.content_array(partition =partition, cam_name ='camB')
    ident_num = dataset.ident_num(partition)

    ids_map = {i : j for j, i in enumerate(set(idA))}

    
    idA = [ids_map[id_] for id_ in idA]
    idB = [ids_map[id_] for id_ in idB]
    
    if partition != 'train':
        idA = idA%ident_num
        idB = idB%ident_num
        label_smoothing = False
        aug = False

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

            imgA = np.array(imgA)
            imgA = preprocess_input(imgA)
            imgA /= 255
            camA_set.append(imgA)
            idA_set.append(np.eye(ident_num)[iA-1]) #np_utils.to_categorical(iA-1,ident_num))  

        for pB, iB in zip(dataB, pidB):
            imgB = dataset.get_image(pB)
            imgB = Image.fromarray(imgB)
            imgB = imgB.resize((img_size[1], img_size[0]))

            if aug:
                imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
           
            imgB = np.array(imgB)
            imgB = preprocess_input(imgB)
            imgB /= 255
            camB_set.append(imgB)
            idB_set.append(np.eye(ident_num)[iB-1]) # (np_utils.to_categorical(iB-1,ident_num))  

        y_true = np.zeros(batch_size)
        
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
            epsilon = 0.1
            cls_idA_set[np.where(cls_idA_set == 0)] = epsilon/ident_num
            cls_idA_set[np.where(cls_idA_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
            
            cls_idB_set[np.where(cls_idB_set == 0)] = epsilon/ident_num
            cls_idB_set[np.where(cls_idB_set == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
        
        if partition != 'train':
            cls_idA_set = np.zeros((cls_idA_set.shape[0], itent_num))
            cls_idB_set = np.zeros((cls_idB_set.shape[0], itent_num))

        
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

def classification_generator(
 dataset,
 batch_size = 32,
 partition = 'train',
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
    data_camA, idA = dataset.content_array(partition =partition, cam_name ='camA')
    data_camB, idB = dataset.content_array(partition =partition, cam_name ='camB')

    all_images_names = np.concatenate([data_camA, data_camB])
    all_ids_list = np.concatenate([idA,idB])
    ident_num = dataset.ident_num(partition)
    all_train_ids = np.unique(idA)
    ids_map = {i : j for j, i in enumerate(all_train_ids)}

    np.random.shuffle(all_train_ids)
    pool = cycle(all_train_ids)

    if partition != 'train':
        idA = idA%ident_num
        idB = idB%ident_num
        label_smoothing = False
        aug = False

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
            s = np.random.choice(
                all_images_names[all_ids_list==c_],
                2)
            for s_ in s:
                    img_data.append(s_)
                    labels_list.append(c_)

        for img_file, lb in zip(img_data, labels_list):
            img = dataset.get_image(img_file)
            img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
            if aug:
                # ""As data augmentation techniques [29], we used random erasing, brightness
                # transformation, horizontal flipping, rotation, translation and
                # zooming, each of them with 50% of probability"
                img = data_augmentation.augment(img,['zooming','translate','rotate','horizontal_flip','brightness','cutout'])
        
            img = np.array(img)
            img = preprocess_input(img)
            img /= 255
            img_batch.append(img)
            label_ = np.array(np.eye(ident_num)[ids_map[lb]])#(np_utils.to_categorical(ids_map[lb],ident_num))
            
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
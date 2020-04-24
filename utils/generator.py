import h5py
import numpy as np
from keras.utils import np_utils
from utils import data_augmentation
import random
from PIL import Image
import cv2
from utils.dataset import Dataset
import scipy.spatial.distance


def classification_generator(dataset, batch_size = 32, aug = True, img_size = (224,320), label_smoothing = True):
    """Returns a generator that yields batchs for Classication Train

    Args:
            dataset: a Dataset object containing interesting images
            batch_size: an integer specifying batch size
            aug: Boolean specifying data augmentation use
            img_size: tuple that contains image size to train the network, in the form (H, W)
            label_smoothing: Boolean specifying label smoothing use
    Returns:
            a generator object
    """  
    c = list(zip(
        dataset.content_array('camA')[0] + dataset.content_array('camB')[0],
        np.concatenate((
                dataset.content_array('camA')[1],
                dataset.content_array('camB')[1]
            ))
        ))

    np.random.shuffle(c)
    img_data, labels = zip(*c)
    
    ident_num = dataset.ident_num()

    while True:
        
        # images 
        img_batch = []
        labels_batch = []
        
        for img_file,lb in zip(img_data,labels):
                
                img = dataset.get_image(img_file)
                img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if aug:
                    img = data_augmentation.augment(img,['zooming','translate','rotate','horizontal_flip','brightness','cutout'])
            

                # conjunto de imagens a serem consideradas
                img_batch.append(np.array(img))
                label_ = np.array(np_utils.to_categorical(lb,ident_num))
                
                ## LABEL SMOOTHING

                if label_smoothing:
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

def quadruplet_generator(feat_model, dataset, batch_size = 12, validation = False, val_ident_num = 0, aug = True, img_size = (224,320), label_smoothing = True):
    data_camA = dataset.content_array('camA')[0]
    data_camB = dataset.content_array('camB')[0]
    idA = dataset.content_array('camA')[1]
    idB = dataset.content_array('camB')[1]
    ident_num = dataset.ident_num()

    if validation:
        ident_num = val_ident_num
        label_smoothing = False

    # images 
    img_batch = []
    # ids
    p1 = []
    p2 = []
    n1 = []
    n2 = []
    

    def apply_ls(label_):
        epsilon = 0.1
        label_[np.where(label_ == 0)] = epsilon/ident_num
        label_[np.where(label_ == 1)] = 1 - ( (ident_num-1)/ident_num)*0.1
        return label_

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
            imgA = dataset.get_image(pA)
            imgA = Image.fromarray(imgA)
            if aug:
                imgA = data_augmentation.augment(imgA,['brightness', 'cutout', 'zooming', 'horizontal_flip', 'rotate', 'translate'])
            
            imgA = imgA.resize((img_size[1], img_size[0]))
            # conjunto de imagens a serem consideradas
            camA_set.append( np.array(imgA)  )
            idA_set.append(iA)  
         

        for pB, iB in zip(dataB, pidB):
            imgB = dataset.get_image(pB)
            imgB = Image.fromarray(imgB)
            if aug:
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
                if label_smoothing:
                    p1.append(apply_ls(np.array(np_utils.to_categorical(pos1_id,ident_num))))
                    n1.append(apply_ls(np.array(np_utils.to_categorical(neg1_id,ident_num))))
                    p2.append(apply_ls(np.array(np_utils.to_categorical(pos2_id,ident_num))))
                    n2.append(apply_ls(np.array(np_utils.to_categorical(neg2_id,ident_num))))
                else:
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
                    if not validation:
                        yield([pos_img1, neg_img1, pos_img2, neg_img2],
                                [y_true,np.stack([p1,n1,p2,n2],axis=1)]) 
                    else:
                        yield([pos_img1, neg_img1, pos_img2, neg_img2],
                                [y_true,np.zeros(np.stack([p1,n1,p2,n2],axis=1).shape)])  

                    img_batch = []
                    p1 = []
                    p2 = []
                    n1 = []
                    n2 = []


def triplet_generator(feat_model, dataset, batch_size = 4, validation = False, val_ident_num = 0, aug = True, img_size = (224,320), label_smoothing = True ):
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

        # embaralhando os dados para gerar conjuntos diferentes
        uIDs = np.unique(idA)
        random.shuffle(uIDs)
        
        posA = []
        posB = []
        for f in uIDs[0:batch_size]:
          posA.extend(np.where(idA == f)[0])
          posB.extend(np.where(idB == f)[0])

        posA = np.array(posA)
        posB = np.array(posB)

        pidA = np.array(idA)[posA]
        pidB = np.array(idB)[posB]
        dataA = np.array(data_camA)[posA]
        dataB = np.array(data_camB)[posB]
        
        for pA, iA in zip(dataA, pidA):
            imgA = dataset.get_image(pA)
            imgA = Image.fromarray(imgA)
            if aug:
                imgA = data_augmentation.augment(imgA,['brightness', 'cutout', 'zooming', 'horizontal_flip', 'rotate', 'translate'])

    #       print(type(imgA))
            imgA = imgA.resize((img_size[1], img_size[0]))
            # conjunto de imagens a serem consideradas
            camA_set.append( np.array(imgA)  )
            idA_set.append(np_utils.to_categorical(iA,ident_num))  



        for pB, iB in zip(dataB, pidB):
            imgB = dataset.get_image(pB)
            if aug:
                imgB = data_augmentation.augment(imgB,['brightness','cutout','zooming', 'horizontal_flip', 'rotate', 'translate'])
           
            if(type(imgB) is np.ndarray):
                imgB = Image.fromarray(imgB)

            imgB = imgB.resize((img_size[1], img_size[0]))
            
            # conjunto de imagens a serem consideradas
            
            camB_set.append(np.array(imgB))
            idB_set.append(np_utils.to_categorical(iB,ident_num))  


        y_true = np.zeros(batch_size) # não tem significado, apenas para não dar erro
        
        if(len(camA_set) < len(camB_set)):
            for i in range(len(camB_set) - len(camA_set)):
                camA_set.append(camA_set[-1])
                idA_set.append(idA_set[-1])

        elif(len(camB_set) < len(camA_set)):
            for i in range(len(camA_set) - len(camB_set)):
                camB_set.append(camB_set[-1])
                idB_set.append(idB_set[-1])
        
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


import numpy as np
from itertools import cycle
import cv2
from utils import data_augmentation

# np.random.seed(42)

class ClassificationGenerator:
    def __init__(self,
                dataset,
                steps,
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
        self.dataset = dataset
        self.batch_size = batch_size
        self.aug = aug
        self.img_size = img_size
        self.label_smoothing = label_smoothing
        self.steps = 0
        self.max_steps = steps
        
        self.data_camA, self.idA = dataset.content_array(partition =partition, cam_name ='camA')
        self.data_camB, self.idB = dataset.content_array(partition =partition, cam_name ='camB')

        self.all_images_names = np.concatenate([self.data_camA, self.data_camB])
        self.all_ids_list = np.concatenate([self.idA,self.idB])
        self.ident_num = dataset.ident_num(partition)
        self.all_train_ids = np.unique(self.idA)
        self.ids_map = {i : j for j, i in enumerate(self.all_train_ids)}

        np.random.shuffle(self.all_train_ids)
        self.pool = cycle(self.all_train_ids)

        if partition != 'train':
            self.idA = self.idA%self.ident_num
            self.idB = self.idB%self.ident_num
            self.label_smoothing = False
            self.aug = False
            
    def __next__(self):
        self.steps+=1
        if self.steps > self.max_steps:
            self.steps = 0
            raise StopIteration 
        return self.get_batch()
    
    def __len__(self):
        return self.max_steps
    
    def __getitem__(self, _):
        return self.__next__()
    
    def process_image(self, img_file):
        img = self.dataset.get_image(img_file)
        img = cv2.resize(img, dsize=(self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
        if self.aug:
            img = data_augmentation.augment(img,['zooming','translate','rotate','horizontal_flip','brightness','cutout'])
        img = np.array(img)/255
        return img

    def get_batch(self):
        current_ids = np.random.choice(self.all_train_ids, int(self.batch_size/2),replace=False)

        # images 
        img_batch = []
        labels_batch = []
        
        # yield( None)
        img_data = []
        labels_list = []
        for c_ in current_ids:
            s = np.random.choice(self.all_images_names[self.all_ids_list==c_],2, replace=False)
            for s_ in s:
                img_data.append(s_)
                labels_list.append(c_)

        for img_file, lb in zip(img_data, labels_list):
            img_batch.append(self.process_image(img_file))
            label_ = np.array(np.eye(self.ident_num)[self.ids_map[lb]])#(np_utils.to_categorical(ids_map[lb],ident_num))
            
            ## LABEL SMOOTHING
            if self.label_smoothing:
                epsilon = 0.1
                label_[np.where(label_ == 0)] = epsilon/self.ident_num
                label_[np.where(label_ == 1)] = 1 - ( (self.ident_num-1)/self.ident_num)*0.1

            labels_batch.append(label_)
            
            # if len(img_batch) == self.batch_size:

                # if len(np.shape(labels_batch))==3:

                    # out = np.squeeze(np.array(labels_batch), axis=1)
                # else:
            out = np.array(labels_batch)
        return (np.moveaxis(np.array(img_batch, dtype=np.float32), -1, 1), out)
    
    def __iter__(self):
        
        return self
    

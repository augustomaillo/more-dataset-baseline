import numpy as np
from itertools import cycle
import cv2
# from tensorflow.keras.applications.resnet import preprocess_input
from utils import data_augmentation

# np.random.seed(42)

class MetricLearningGenerator:
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
        
        self.data_camA = np.array(self.data_camA)
        self.data_camB = np.array(self.data_camB)
        self.idA = np.array(self.idA)
        self.idB = np.array(self.idB)

        self.all_images_names = np.concatenate([self.data_camA, self.data_camB])
        self.all_ids_list = np.concatenate([self.idA,self.idB])
        self.ident_num = dataset.ident_num(partition)
        self.all_train_ids = np.unique(self.idA)
        self.ids_map = {i : j for j, i in enumerate(self.all_train_ids)}

        np.random.shuffle(self.all_train_ids)
        # self.pool = cycle(self.all_train_ids)

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
        n_anchors = self.batch_size//2
        
        current_ids = np.random.choice(self.all_train_ids, int(n_anchors),replace=False)
        # first half anchors from cam A
        img_batchA = []
        img_batchB = []
        labels_batchA = []
        labels_batchB = []
        for id_ in current_ids:
            img_file = np.random.choice(self.data_camA[self.idA == id_], 1)[0] # anchor
            img_batchA.append(self.process_image(img_file))
            
            img_file = np.random.choice(self.data_camB[self.idB == id_], 1)[0] # positive
            img_batchB.append(self.process_image(img_file))
            
            label_ = np.array(np.eye(self.ident_num)[self.ids_map[id_]])
            
            ## LABEL SMOOTHING
            if self.label_smoothing:
                epsilon = 0.1
                label_[np.where(label_ == 0)] = epsilon/self.ident_num
                label_[np.where(label_ == 1)] = 1 - ( (self.ident_num-1)/self.ident_num)*0.1
            ####
            
            labels_batchA.append(label_)
            labels_batchB.append(label_)

        img_batch = img_batchA + img_batchB
        labels_batch = labels_batchA + labels_batchB
    
        return np.moveaxis(np.array(img_batch, dtype=np.float32), -1, 1) , np.array(labels_batch, dtype=np.float32)

    def get_batch_my_quad_loss(self):
        img_batchA = []
        img_batchB = []
        labels_batchA = []
        labels_batchB = []
        while len(img_batchA) < self.batch_size//2:
            id_ = np.random.choice(self.all_train_ids,1)[0]
            
            img_files_A = np.random.permutation(np.where(self.idA == id_))
            img_files_A = img_files_A[:2]
            
            img_files_B = np.random.permutation(np.where(self.idB == id_))
            img_files_B = img_files_B[:2]
            
            selected_samples_for_id = min(len(img_files_A[0]), len(img_files_B[0]))
            
            for img_file in img_files_A[0][:selected_samples_for_id]:
                img_batchA.append(self.process_image(self.data_camA[img_file]))
            
            for img_file in img_files_B[0][:selected_samples_for_id]:
                img_batchB.append(self.process_image(self.data_camB[img_file]))


            label_ = np.array(np.eye(self.ident_num)[self.ids_map[id_]])#(np_utils.to_categorical(ids_map[lb],ident_num))
            ## LABEL SMOOTHING
            if self.label_smoothing:
                epsilon = 0.1
                label_[np.where(label_ == 0)] = epsilon/self.ident_num
                label_[np.where(label_ == 1)] = 1 - ( (self.ident_num-1)/self.ident_num)*0.1
            ####
            for _ in range(selected_samples_for_id):
                labels_batchA.append(label_)
                labels_batchB.append(label_)
        
        img_batch = img_batchA[:int(self.batch_size//2)] + img_batchB[:int(self.batch_size//2)]
        labels_batch = labels_batchA[:int(self.batch_size//2)] + labels_batchB[:int(self.batch_size//2)]

        return np.moveaxis(np.array(img_batch, dtype=np.float32), -1, 1) , np.array(labels_batch, dtype=np.float32)

    
    def __iter__(self):
        
        return self
    

    
    
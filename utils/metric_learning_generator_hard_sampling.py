import numpy as np
from utils.metric_learning_generator import MetricLearningGenerator
from sklearn.metrics.pairwise import cosine_distances
from torch_utils.eval_metric_model import _cmc_curve

class MetricLearningGeneratorHardSampling(MetricLearningGenerator):
    POSITVE_CANDIDATES = 3
    NEGATIVE_CANDIDATES = 50
    
    def __init__(self,
                dataset,
                steps,
                batch_size = 32,
                partition = 'train',
                aug = True, 
                img_size = (224,320), 
                label_smoothing = True,
                feature_dims = 2048
                ):
        
        super().__init__(dataset, steps, batch_size, partition, aug, img_size, label_smoothing)

        self._indexes_cam_a = np.arange(0, len(self.data_camA))
        self._indexes_cam_b = np.arange(0, len(self.data_camB)) + len(self.data_camA)
        
        self._all_indexes = np.concatenate([self._indexes_cam_a, self._indexes_cam_b])

        
        self._features =  np.random.rand(len(self._all_indexes),  feature_dims)
        
        self._distances =np.zeros((len(self._all_indexes), len(self._all_indexes)))
        
        
    def update_features(self, features, indexes):
        assert features.shape[1] == 2048
        self._features[indexes] = features

    def update_distances(self):        
        self._distances = cosine_distances(self._features, self._features)
        
    def evaluate_on_train(self):
        _cmc_curve(
            probe_features=self._features[self._indexes_cam_a],
            ids_camA=self.idA,
            gallery_features=self._features[self._indexes_cam_b],
            ids_camB=self.idB,
            output_path=None,
            metric='cosine'
        )
        
    def get_positive_for_anchor(self, anchor_index):
        
        camA = anchor_index < len(self.data_camA)

        anchor_pid = self.idA[anchor_index] if camA else self.idB[anchor_index - len(self.data_camA)]
        positives = self.idB == anchor_pid if camA else self.idA == anchor_pid
        positives_indexes = self._indexes_cam_b[positives] if camA else self._indexes_cam_a[positives]

        # FARTHEST NEGATIVES (LONG DISTANCES)
        positive_order = self._distances[anchor_index][positives_indexes].argsort()[::-1][:self.POSITVE_CANDIDATES]

        chosen = np.random.choice(positives_indexes[positive_order], 1, replace=False)[0]
        
        return chosen
    
    def get_negative_for_anchor(self, anchor_index, blacklist_id=[]):
        camA = anchor_index < len(self.data_camA)
        
        blacklist_pid = [
            self.idA[i] if i < len(self.data_camA) else self.idB[i - len(self.data_camA)] for i in blacklist_id
        ]
        
        anchor_pid = self.idA[anchor_index] if camA else self.idB[anchor_index - len(self.data_camA)]
        negative = (self.idB != anchor_pid) if camA else (self.idA != anchor_pid) 
        blacklisted = (~np.isin(self.idB, blacklist_pid)) if camA else (~np.isin(self.idA, blacklist_pid))
        
        negative_indexes = self._indexes_cam_b[negative & blacklisted] if camA else self._indexes_cam_a[negative & blacklisted]

        # CLOSEST NEGATIVES (SHORT DISTANCES)
        negative_order = self._distances[anchor_index][negative_indexes].argsort()[:self.NEGATIVE_CANDIDATES]

        chosen = np.random.choice(negative_indexes[negative_order], 1, replace=False)[0]
        return chosen
        

    def _load_by_index(self, index):
        camA = index < len(self.data_camA)
        img = self.process_image(
            self.data_camA[index] if camA else self.data_camB[index - len(self.data_camA)]
        )
        label = np.zeros(self.ident_num)
        id_ = self.ids_map[self.idA[index] if camA else self.idB[index-len(self.data_camA)]]
        if self.label_smoothing:
            epsilon = 0.1
            label = np.ones(self.ident_num) * epsilon/self.ident_num
            label[id_] = 1 - ( (self.ident_num-1)/self.ident_num)*0.1
        else:
            label = np.zeros(self.ident_num)
            label[id_] = 1.0
            
        return img, label
        
        
        """ O LOADER CRIA UMA COPIA DO GENERATOR """

    def get_batch(self):
        n_anchors = self.batch_size//4
        
        current_ids = np.random.choice(self._all_indexes, int(n_anchors),replace=False)
        
        anchors = [] # cam A
        positives = [] # cam B
        negative1 = [] # cam B
        negative2 = [] # cam A
        
        anchors_labels = []
        positives_labels = []
        negative1_labels = []
        negative2_labels = []
        
        anchors_indexes = []
        positives_indexes = []
        negative1_indexes = []
        negative2_indexes = []
        
        for anchor_index in current_ids:
            anchor_img, anchor_label = self._load_by_index(anchor_index)
            
            positive_index = self.get_positive_for_anchor(anchor_index)
            positive_img, positive_label = self._load_by_index(positive_index)
            
            negative1_index = self.get_negative_for_anchor(anchor_index)
            negative1_img, negative1_label = self._load_by_index(negative1_index)
            
            negative2_index = self.get_negative_for_anchor(negative1_index, blacklist_id=[anchor_index])
            negative2_img, negative2_label = self._load_by_index(negative2_index)
            
            
            anchors.append(anchor_img)
            anchors_labels.append(anchor_label)
            anchors_indexes.append(anchor_index)
            
            positives.append(positive_img)
            positives_labels.append(positive_label)
            positives_indexes.append(positive_index)
            
            negative1.append(negative1_img)
            negative1_labels.append(negative1_label)
            negative1_indexes.append(negative1_index)
            
            negative2.append(negative2_img)
            negative2_labels.append(negative2_label)
            negative2_indexes.append(negative2_index)

        img_batch = anchors + positives + negative1 + negative2
        labels_batch = anchors_labels + positives_labels + negative1_labels + negative2_labels
        indexes_batch = anchors_indexes + positives_indexes + negative1_indexes + negative2_indexes
        
        
        return np.moveaxis(np.array(img_batch, dtype=np.float32), -1, 1), \
                np.array(labels_batch, dtype=np.float32), \
                np.array(indexes_batch, dtype=np.int32)
                
    

    
    
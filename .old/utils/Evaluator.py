# from matplotlib import pyplot
import numpy as np 
from PIL import Image
# from keras.models import Model
import scipy.spatial.distance
from sklearn.metrics import average_precision_score
from tensorflow.keras.applications.resnet import preprocess_input
from multiprocessing.pool import ThreadPool
# from model.AuxLayers import BNNeckForInference
from utils.dataset import Dataset
class EvalModel():
        def __init__(self, dataset, partition, image_shape):
            print('Loading train data on RAM.')
            self.image_shape = image_shape
            test_dataA, self.ids_camA = dataset.content_array(partition = partition, cam_name ='camA')
            test_dataB, self.ids_camB = dataset.content_array(partition = partition, cam_name ='camB')
            self.camA_set = []
            self.camB_set = []
            
            def process_image(img_name):
                imgA = dataset.get_image(img_name)
                imgA = Image.fromarray(imgA)
                imgA = imgA.resize((self.image_shape[1], self.image_shape[0]))
                imgA = np.array(imgA)
                imgA = preprocess_input(imgA)
                imgA /= 255.0
                return imgA

            processA = ThreadPool(4).map(process_image, test_dataA)
            for r in processA:
                self.camA_set.append(r)
                print(len(self.camA_set), '/',len(test_dataA), end='\r')
            self.camA_set = np.array(self.camA_set)
            print()
            processB = ThreadPool(4).map(process_image, test_dataB)
            for r in processB:
                self.camB_set.append(r)
                print(len(self.camB_set), '/', len(test_dataB), end='\r')
            self.camB_set = np.array(self.camB_set)

        def compute(self, 
            feat_model,
            name='cmc_curve', 
            metrics = ['r1, mAP'], 
            curve_label = 'Evaluating', 
            curve_color='red', 
            BN = False
        ):                       
            return generate_cmc_curve(
                feat_model,
                self.camA_set,
                self.camB_set,
                self.ids_camA,
                self.ids_camB,
                name = name,
                image_size = self.image_shape,
                metrics = metrics,
                curve_label = curve_label,
                curve_color = curve_color,
                BN = BN,
                )
            
class TestGenerator:
    def __init__(self, dataset : Dataset, partition, cam, image_shape):
        """A generetor for test data

        Args:
            dataset (Dataset): dataset containing all more data
            partition (str): train/test
            cam (str): camA/camB
            image_shape (tuple): img target size
        """
        self.dataset = dataset
        self.partition = partition
        self.img_shape = image_shape
        self.cam = cam
    
        self.test_data, self.ids_cam = dataset.content_array(partition = partition, cam_name =cam)
        
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        img = self.process_image(self.test_data[idx])
        img = np.moveaxis(img, -1, 0)
        label = self.ids_cam[idx]
        return img, label
        
        
    def process_image(self, img_name):
        imgA = self.dataset.get_image(img_name)
        imgA = Image.fromarray(imgA)
        imgA = imgA.resize((self.img_shape[1], self.img_shape[0]))
        imgA = np.array(imgA, dtype=np.float32)
        imgA = np.divide(imgA, 255, casting='unsafe')
        return imgA
             

def generate_cmc_curve(
        feat_model, 
        camA_set,
        camB_set, 
        ids_camA,
        ids_camB,
        name='cmc_curve', 
        image_size = (224,320),
        metrics = ['r1, mAP'], 
        curve_label = 'On Train Data', 
        curve_color='red', 
        BN = False
        ):

        metric = 'euclidean'
        if not BN:
            probe_features = feat_model.predict(camA_set, verbose=1)
            gallery_features = feat_model.predict(camB_set, verbose=1)
        else:
            print('BN active for inference. Metric for inference changed to cosine.')
            bnneck = BNNeckForInference(feat_model, image_size)
            probe_features = bnneck.predict(camA_set, verbose=1)
            gallery_features = bnneck.predict(camB_set, verbose=1)
            metric = 'cosine'

        cmc_curve = np.zeros(gallery_features.shape[0])
        ap_array = []
        all_dist = scipy.spatial.distance.cdist(probe_features, gallery_features, metric=metric)

        for idx, p_dist in enumerate(all_dist):
            rank_p = np.argsort(p_dist,axis=None)
            ranked_ids = ids_camB[rank_p]
            pos = np.where(ranked_ids == ids_camA[idx])
            cmc_curve[pos[0][0]]+=1
            y_true = np.zeros(np.shape(ids_camB))
            y_true[np.where(ids_camB == ids_camA[idx])] = 1
            y_pred = 1/(p_dist + 1e-8) # remove /0
            y_pred = np.squeeze(y_pred)
            ap = average_precision_score(y_true, y_pred)
            ap_array.append(ap)
        cmc_curve = np.cumsum(cmc_curve)/probe_features.shape[0]

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.title('CMC Curve - R1: %.2f / mAP: %.2f'%(cmc_curve[0]*100, np.mean(ap_array)*100) )
        pyplot.ylabel('Recognition Rate (%)')
        pyplot.xlabel('Rank')
        pyplot.plot(cmc_curve,label = curve_label, color=curve_color)
        pyplot.legend(loc='upper left')
        pyplot.savefig(name+'.png')
        pyplot.close()
        return cmc_curve, cmc_curve[0], np.mean(ap_array)


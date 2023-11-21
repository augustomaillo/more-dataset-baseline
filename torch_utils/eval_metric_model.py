from utils.dataset import Dataset
from utils.test_generator import TestGenerator
from torch_model.metric_learning_model import Resnet50MetricLearning

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np 
import scipy
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from sklearn.metrics.pairwise import cosine_distances    
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                )

def process_generator(generator, model, batch_size, workers):
    loader = DataLoader(generator, batch_size=batch_size, shuffle=False, num_workers=workers)
    extracted_features = []
    ids_cam = []
    for x,y in tqdm(loader):
        with torch.no_grad():
            x = x.to('cuda')
            x = normalize(x)
            features, _ = model(x)
        extracted_features.append(features.detach())
        ids_cam.append(y)
    extracted_features = [tensor.cpu().numpy().copy() for tensor in extracted_features]
    extracted_features = np.concatenate(extracted_features).squeeze()    
    ids_cam = [id_list.cpu().numpy().tolist() for id_list in ids_cam]
    ids_cam = np.array(sum(ids_cam, []))
    return extracted_features, ids_cam

def _cmc_curve(probe_features, ids_camA, gallery_features, ids_camB, output_path, metric='euclidean'):
    # metric = 'euclidean'
    # if not BN:
    #     probe_features = feat_model.predict(camA_set, verbose=1)
    #     gallery_features = feat_model.predict(camB_set, verbose=1)
    # else:
    #     print('BN active for inference. Metric for inference changed to cosine.')
    #     bnneck = BNNeckForInference(feat_model, image_size)
    #     probe_features = bnneck.predict(camA_set, verbose=1)
    #     gallery_features = bnneck.predict(camB_set, verbose=1)
    #     metric = 'cosine'

    cmc_curve = np.zeros(gallery_features.shape[0])
    ap_array = []
    all_dist = cosine_distances(probe_features, gallery_features)
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
    print('CMC Curve - R1: %.2f / mAP: %.2f'%(cmc_curve[0]*100, np.mean(ap_array)*100) )
    if output_path is None:
        return
    print('Saving CMC Curve.')
    pyplot.title('CMC Curve - R1: %.2f / mAP: %.2f'%(cmc_curve[0]*100, np.mean(ap_array)*100) )
    pyplot.ylabel('Recognition Rate (%)')
    pyplot.xlabel('Rank')
    pyplot.plot(cmc_curve,label = 'cmc', color='red')
    pyplot.legend(loc='upper left')
    pyplot.savefig(output_path)
    pyplot.close()


def evaluate(
    metric_learning_model_weights : str,
    dataset : Dataset,
    train_target_ids_num: int,
    device='cpu',
    batch_size=32,
    workers=1,
    cmc_curve_plot_path:str = None,
    metric='euclidean'
    ):
    print('Creating model and loading weights...')
    model_metric = Resnet50MetricLearning(train_target_ids_num, inference=False).eval().to(device)
    incompatible_keys = model_metric.load_state_dict(torch.load(metric_learning_model_weights), strict=False, )
    print('Keys not loaded:', incompatible_keys)

    print('Processing cams images')
    test_gen_camA = TestGenerator(dataset, partition='test', cam='camA', image_shape=(256,256))
    test_gen_camB = TestGenerator(dataset, partition='test', cam='camB', image_shape=(256,256))
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    probe_features, ids_camA = process_generator(test_gen_camA, model_metric, batch_size=batch_size, workers=workers)
    print(f'Loaded {len(ids_camA)} samples for cam A')

    gallery_features, ids_camB = process_generator(test_gen_camB, model_metric, batch_size=batch_size, workers=workers)
    print(f'Loaded {len(ids_camB)} samples for cam B')
    
    print('Generating metrics...')
    _cmc_curve(probe_features, ids_camA, gallery_features, ids_camB, cmc_curve_plot_path, metric=metric)
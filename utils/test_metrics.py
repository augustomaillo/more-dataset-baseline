from matplotlib import pyplot
import numpy as np 
from PIL import Image
from keras.models import Model
import scipy.spatial.distance
from sklearn.metrics import average_precision_score
from model import resnet_b4_max_pool as new_resnet


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

'''
def generate_cmc_curve(feat_model, test_data_list, name='cmc_curve', image_size =(224,320),metrics = ['r1, mAP'], curve_label = 'Test', curve_color='red' ):
    test_dataA, test_identA, test_dataB, test_identB = test_data_list

    camA_set0 = []
    camB_set0 = []
      
    # feat_model = new_resnet.feat_net((224,320)) # prev layer
    # feat_model._make_predict_function()    # https://github.com/keras-team/keras/issues/6462
    # feat_model.load_weights(feat_path)

    for pA in test_dataA:
      imgA = np.asarray(Image.open(pA)).copy()
      imgA = Image.fromarray(imgA)
      imgA = imgA.resize((image_size[1], image_size[0]))
      camA_set0.append(np.array(imgA))

    camA_set = np.array(camA_set0)

    for pB in test_dataB:
      imgB = np.asarray(Image.open(pB)).copy()
      imgB = Image.fromarray(imgB)
      imgB = imgB.resize((image_size[1], image_size[0]))
      camB_set0.append(np.array(imgB)) 

    camB_set = np.array(camB_set0)

    ids_camA = np.array(test_identA)
    ids_camB = np.array(test_identB)

    probe_features = feat_model.predict(camA_set)
    gallery_features = feat_model.predict(camB_set)

    cmc_curve = np.zeros(gallery_features.shape[0])
    ap_array = []

    for idx, probe in enumerate(probe_features):

      rank_p = np.argsort(scipy.spatial.distance.cdist(np.expand_dims(probe,axis=0), gallery_features),axis=None)
      ranked_ids = ids_camB[rank_p]
      pos = np.where(ranked_ids == ids_camA[idx])
      cmc_curve[pos[0][0]]+=1

      y_true = np.zeros(np.shape(ids_camB))
      y_true[np.where(ids_camB == ids_camA[idx])] = 1
      y_pred = 1/(scipy.spatial.distance.cdist(np.expand_dims(probe,axis=0), gallery_features))
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
'''


def generate_cmc_curve(feat_model, dataset, name='cmc_curve', image_size =(224,320),metrics = ['r1, mAP'], curve_label = 'Test', curve_color='red' ):
    test_dataA = dataset.content_array('camA')[0]
    test_dataB = dataset.content_array('camB')[0]
    test_identA = dataset.content_array('camA')[1]
    test_identB = dataset.content_array('camB')[1]

    camA_set0 = []
    camB_set0 = []
      
    # feat_model = new_resnet.feat_net((224,320)) # prev layer
    # feat_model._make_predict_function()    # https://github.com/keras-team/keras/issues/6462
    # feat_model.load_weights(feat_path)

    for pA in test_dataA:
      imgA = dataset.get_image(pA)
      imgA = Image.fromarray(imgA)
      imgA = imgA.resize((image_size[1], image_size[0]))
      camA_set0.append(np.array(imgA))

    camA_set = np.array(camA_set0)

    for pB in test_dataB:
      imgB = dataset.get_image(pB)
      imgB = Image.fromarray(imgB)
      imgB = imgB.resize((image_size[1], image_size[0]))
      camB_set0.append(np.array(imgB)) 

    camB_set = np.array(camB_set0)

    ids_camA = np.array(test_identA)
    ids_camB = np.array(test_identB)

    probe_features = feat_model.predict(camA_set)
    gallery_features = feat_model.predict(camB_set)

    cmc_curve = np.zeros(gallery_features.shape[0])
    ap_array = []

    for idx, probe in enumerate(probe_features):

      rank_p = np.argsort(scipy.spatial.distance.cdist(np.expand_dims(probe,axis=0), gallery_features),axis=None)
      ranked_ids = ids_camB[rank_p]
      pos = np.where(ranked_ids == ids_camA[idx])
      cmc_curve[pos[0][0]]+=1

      y_true = np.zeros(np.shape(ids_camB))
      y_true[np.where(ids_camB == ids_camA[idx])] = 1
      y_pred = 1/(scipy.spatial.distance.cdist(np.expand_dims(probe,axis=0), gallery_features) + 1e-8)
      y_pred = np.squeeze(y_pred)

      # print('min: ',(scipy.spatial.distance.cdist(np.expand_dims(probe,axis=0), gallery_features)).min())

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

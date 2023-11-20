import torch
from utils.dataset import Dataset
from torch_utils.eval_metric_model import evaluate


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    more_path = "../MoRe---ReID-Codes/MoRe.hdf5"
    more = Dataset(more_path, to_bgr = True) # Images were used in BGR format

    _, idsa = more.content_array(partition='train', cam_name='camA')
    _, idsb = more.content_array(partition='train', cam_name='camB')

    total_images_train = len(idsa) + len(idsb)
    target_ident_num = more.ident_num(partition='train')


    evaluate(
        metric_learning_model_weights='weights/torch.myquad(1.0)center(5e-4).november.pt', 
        dataset=more,
        train_target_ids_num=more.ident_num('train'),
        device='cuda',
        batch_size=96,
        workers=1,
        cmc_curve_plot_path='cmc_curves/torch.myquad(1.0)center(5e-4).november.no_last_stride.png',
        metric='cosine'
    )
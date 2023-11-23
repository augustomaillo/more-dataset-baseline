import torch

from utils.dataset import Dataset
from torch_utils.train_metric_model_hard_sampling import train


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    more_path = "../MoRe---ReID-Codes/MoRe.hdf5"
    more = Dataset(more_path, to_bgr = True) # Images were used in BGR format

    train(
        more,
        num_epochs=120,
        batch_size=32,
        weights_output='weights/torch.quad(1.0)center(1e-3).last_stride.november.alpha_divide.HARD_SAMPLING.pt',
        starting_weights='weights/torch.only_cls.last_stride.pt',
        device=device,
        only_classification=False,
        quad_weight=1,
        center_weight=1e-3
    )
from utils.dataset import Dataset
import numpy as np
from PIL import Image

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
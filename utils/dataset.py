import h5py
import numpy as np
from PIL import Image

class Dataset():
    def __init__(self, h5_file, to_bgr):
        """Constructor for MoRe_Dataset class

        Args:
                h5_file: path to hdf5 file with images
                to_bgr: bool to convert data to bgr
        """

        self._h5File = h5_file
        self.to_bgr = to_bgr
        

        self._identNum = dict()
        with h5py.File(h5_file, 'r') as f:
            train_uids = np.unique(list(f['metadata']['train_idsA'].value))
            test_uids = np.unique(list(f['metadata']['test_idsA'].value))

        self._identNum['train'] = train_uids
        self._identNum['test'] = test_uids

        amount = []
        with h5py.File(h5_file, 'r') as f:
            for e in f['metadata'].keys():
                if 'ids' in e:
                    amount.append(len(f['metadata'][e].value))
        self._imgs_amount = sum(amount)


    def content_array(self, partition, cam_name):
        """Returns files names and its ident number

        Args:
                cam_name: a string specifying the camera

        Returns:
                a list with files names
                a numpy array with its ident number
        """  
        cam_name = cam_name[-1]
        with h5py.File(self._h5File, 'r') as f:
            return (list(f['metadata/%s_files%s'%(partition, cam_name)].value), np.asarray(f['metadata/%s_ids%s'%(partition, cam_name)].value, np.int))


    def get_image(self, img_path):
        """Returns image data

        Args:
                img_path: a string specifying image path

        Returns:
                a numpy array containing image data
        """  

        with h5py.File(self._h5File, 'r') as f:
            if self.to_bgr:
                return f['images'][img_path].value[:,:,::-1]
            else:
                return f['images'][img_path].value

    def ident_num(self, partition):
        """Returns the number of uniques ID on dataset
        """  
        return len(self._identNum[partition])

    def images_amount(self):
        """Returns the number of images on dataset
        """
        return self._imgs_amount

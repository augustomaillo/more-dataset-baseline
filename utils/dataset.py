import h5py
import numpy as np

class Dataset():

    def __init__(self, h5_file):
        self._infoDict = dict()

        self._h5File = h5_file
        self._IDs = dict()
        self._keys = dict()

        with h5py.File(self._h5File, 'r') as f:
            for cam in f.keys():
                self._IDs[cam] = []
                self._keys[cam] = list(f[cam].keys())

                for img in f[cam].keys():
                    self._IDs[cam].append(f[cam][img].attrs['ID'])

        self._identNum = len(
            np.unique(
                np.concatenate(
                    [self._IDs[f] for f in self._IDs.keys()]
                    )
                )
            )

        self._imgs_amount = sum([ len(self._keys[cam]) for cam in self._keys.keys() ])


    def content_array(self, cam_name):
        """Returns files names and its ident number

        Args:
                cam_name: a string specifying the camera

        Returns:
                a list with files names
                a numpy array with its ident number
        """  
        return self._keys[cam_name],np.array(self._IDs[cam_name])

    def get_image(self, filename):
        """Returns image data

        Args:
                filename: a string specifying image name

        Returns:
                a numpy array containing image data
        """  
        # print(filename)

        # print(type(self._keys['camA']))
        # print('filename: ', filename, ' keys type: ', type(self._keys['camA']))
        # print(filename in self._keys['camA'])
        # if( filename in self._keys['camA']):
        #     cam = 'camA'
        # else:
        #     cam = 'camB'

        # with h5py.File(self._h5File, 'r') as f:
        #     # print(type(f[cam][filename]))
        #     return f[cam][filename].value
        with h5py.File(self._h5File, 'r') as f:
            # print(type(f[cam][filename]))
            return f[str(filename[:4])][filename].value

    def ident_num(self):
        """Returns the number of uniques ID on dataset
        """  
        return self._identNum

    def images_amount(self):
        """Returns the number of images on dataset
        """
        return self._imgs_amount


import os, h5py
from PIL import Image
import argparse
import numpy as np

def build_hdf5(dataset_folder, name):
    """Convert data to hdf5 file

    Args:
        dataset_folder: a string path to data

        folder must contain only:
            cameras subfolder 
            Distractors subfolder
            train_filesA.txt -> path to each train file from cam A
            train_filesB.txt -> path to each train file from cam B
            test_filesA.txt -> path to each test file from cam A
            test_filesB.txt -> path to each test file from cam B
    """
    with h5py.File(name, 'a') as f:
        if not 'images' in f.keys():
            f.create_group('images')
        if not 'metadata' in f.keys():
            f.create_group('metadata')


    for folder in os.listdir(dataset_folder):
        
        if not os.path.isdir(os.path.join(dataset_folder, folder)):
            continue

        with h5py.File(name, 'a') as f:
            if not folder in f['images'].keys():
                f['images'].create_group(folder)

        for subfolder in os.listdir(os.path.join(dataset_folder, folder)):
            with h5py.File(name, 'a') as f:
                if not subfolder in f['images'][folder].keys():
                    f['images'][folder].create_group(subfolder)

            for image in os.listdir(os.path.join(dataset_folder,folder,subfolder)):
                img_array = np.array(Image.open(os.path.join(dataset_folder,folder,subfolder,image)))
                with h5py.File(name, 'a') as f:  
                    if image in f['images'][folder][subfolder].keys():
                        del f['images'][folder][subfolder][image]           
                    dset = f['images'][folder][subfolder].create_dataset(
                                image,
                                img_array.shape,
                                dtype='uint8'
                            )
                    dset[:,:,:] = img_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert to hdf5')

    parser.add_argument('-i', action="store", dest="i", type=str)
    parser.add_argument('-o', action="store", dest="o", type=str)

    args = parser.parse_args()
    build_hdf5(args.i, args.o)




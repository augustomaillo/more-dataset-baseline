import os, h5py
from PIL import Image
import argparse
import numpy as np
import unicodedata

def build_hdf5(dataset_folder, name):
    """Convert data to hdf5 file

    Args:
        dataset_folder: a string path to data

        folder must contain only:
            cameras subfolder 
            Distractors subfolder
            train_files.txt -> path to each train file 
            test_files.txt -> path to each test file
    """
    with h5py.File(name, 'w') as f:
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

    train_file = os.path.join(dataset_folder,'train_files.txt')
    test_file = os.path.join(dataset_folder,'test_files.txt')
    camA_train = []
    camB_train = []
    camA_test = []
    camB_test = []
    camA_files_train = []
    camB_files_train = []
    camA_files_test = []
    camB_files_test = []

    with open(train_file, 'r', encoding='ascii') as pointer:
        files = pointer.readlines()
    files = map(str.strip, files)

    for file in files:
        if 'camA' in file:
            camA_train.append(int(file.split('/')[2][8:13]))
            camA_files_train.append(file)
        else:
            camB_train.append(int(file.split('/')[2][8:13]))
            camB_files_train.append(file)
    with open(test_file, 'r', encoding='ascii') as pointer:
        files = pointer.readlines()
    files = map(str.strip, files)
    for file in files:
        if 'camA' in file:
            camA_test.append(int(file.split('/')[2][8:13]))
            camA_files_test.append(file)
        else:
            camB_test.append(int(file.split('/')[2][8:13]))
            camB_files_test.append(file)

    with h5py.File(name, 'a') as f: 
        f['metadata'].create_dataset('train_filesA', data=np.asarray(camA_files_train, dtype='S'))
        f['metadata'].create_dataset('train_filesB', data=np.asarray(camB_files_train, dtype='S'))
        f['metadata'].create_dataset('test_filesA', data=np.asarray(camA_files_test, dtype='S'))
        f['metadata'].create_dataset('test_filesB', data=np.asarray(camB_files_test, dtype='S'))
        f['metadata'].create_dataset('train_idsA', data=np.asarray(camA_train,))
        f['metadata'].create_dataset('train_idsB', data=np.asarray(camB_train,))
        f['metadata'].create_dataset('test_idsA', data=np.asarray(camA_test,))
        f['metadata'].create_dataset('test_idsB', data=np.asarray(camB_test,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert to hdf5')

    parser.add_argument('-i', action="store", dest="i", type=str)
    parser.add_argument('-o', action="store", dest="o", type=str)

    args = parser.parse_args()
    build_hdf5(args.i, args.o)
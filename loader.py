import os
import numpy as np
import SimpleITK as sitk
import cProfile
from sklearn.model_selection import KFold

def load_input_list(path, classes, channels=True):
    labels = {}
    images = {}
    t2_images = {}
    names = set([])
    for entry in os.scandir(path):
        name = entry.name.split('.', maxsplit=1)[0]
        names.add(name)
        if entry.is_file() and (entry.name.endswith(".label.mhd") or entry.name.endswith(".label.nrrd")):
            labels[name] = entry.path
        if entry.is_file() and (entry.name.endswith(".dce.image.mhd") or entry.name.endswith(".dce.image.nrrd")):
            images[name] = entry.path
        if entry.is_file() and (entry.name.endswith(".t2.image.mhd") or entry.name.endswith(".t2.image.nrrd")):
            t2_images[name] = entry.path
    #data = []
    imagelist = []
    t2imagelist = []
    labellist = []
    namelist = []
    dsolist = []
    for name in names:
        if name in images and name in labels and name in t2_images:
            realImage =sitk.ReadImage(images[name])
            dsolist.append((realImage.GetDirection(), realImage.GetSpacing(), realImage.GetOrigin()))
            image = sitk.GetArrayFromImage(realImage)
            no_z = image.shape[0] == 1
            if channels:
                if no_z:
                    image = image.reshape((image.shape[1], image.shape[2], image.shape[3], 1))
                else:
                    image = image.reshape((image.shape[0], image.shape[1], image.shape[2], image.shape[3], 1))
            else:
                if no_z:
                    image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
                else:
                    image = image.reshape((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
            
            label = sitk.GetArrayFromImage(sitk.ReadImage(labels[name]))
            if classes > 0:
                if no_z:
                    label = label.reshape((label.shape[0], label.shape[1], label.shape[2], 1))
                else:
                    label = label.reshape((1, label.shape[0], label.shape[1], label.shape[2],1))
            else:
                if no_z:
                    label = label.reshape((label.shape[0], label.shape[1], label.shape[2]))
                else:
                    label = label.reshape((1, label.shape[0], label.shape[1], label.shape[2]))
            #label = np.repeat(label, image.shape[0], axis=0)
            #data.append((image, label, name))
            imagelist.append(image)

            realT2image = sitk.ReadImage(t2_images[name])
            t2image = sitk.GetArrayFromImage(realT2image)
            if channels:
                t2image = t2image.reshape(t2image.shape + (1,))
            t2imagelist.append(t2image)

            if classes > 1:
                label = to_categorical(label)
            labellist.append(label)
            namelist.append(name.split()[-1])
            #print(name, image.shape, label.shape)
            #yield image, label, name
    imagelist = np.array(imagelist)
    t2imagelist = np.array(t2imagelist)
    labellist = np.array(labellist)
    namelist = np.array(namelist)
    dsolist = np.array(dsolist)

    sorted_index = np.argsort(namelist)
    return imagelist[sorted_index], t2imagelist[sorted_index], labellist[sorted_index], namelist[sorted_index], dsolist[sorted_index]


def load_input(path, classes, label_selector=None, channels=True, use_processed = True):
    labels = {}
    images = {}
    slice_images = {}
    slice_labels = {}
    t2_images = {}
    names = set([])
    if use_processed:
        scanpath = path + "/preprocessed"
    else:
        scanpath = path + "/unprocessed"
    for entry in os.scandir(scanpath):
    #for entry in os.scandir(path + "/unprocessed"):
        name = entry.name.split('.', maxsplit=1)[0]
        names.add(name)
        if entry.is_file() and (entry.name.endswith(".label.mhd") or entry.name.endswith(".label.nrrd")):
            labels[name] = entry.path
        if entry.is_file() and (entry.name.endswith(".dce.image.mhd") or entry.name.endswith(".dce.image.nrrd")):
            images[name] = entry.path
        if entry.is_file() and (entry.name.endswith(".t2.image.mhd") or entry.name.endswith(".t2.image.nrrd")):
            t2_images[name] = entry.path
    for entry in os.scandir(path+"/normalized_slice"):
        name = entry.name.split('.', maxsplit=1)[0]
        names.add(name)
        if entry.is_file() and (entry.name.endswith(".dce.image.mhd") or entry.name.endswith(".dce.image.nrrd")):
            slice_images[name] = entry.path
            #images[name] = entry.path
        if entry.is_file() and (entry.name.endswith(".label.mhd") or entry.name.endswith(".label.nrrd")):
            slice_labels[name] = entry.path
    #data = []
    usable_names = []
    for name in names:
        if name in images and name in slice_images and name in labels and name in t2_images:
            usable_names.append(name)
    names = np.sort(np.array(usable_names))

    #imagelist = np.array([])
    #t2imagelist = np.array([])
    #labellist = np.array([])
    #dsolist = []
    for i, name in enumerate(names):
        #if name != "42306110":
        #    continue
        realImage = sitk.ReadImage(images[name])
        dso = (realImage.GetDirection(), realImage.GetSpacing(), realImage.GetOrigin())
        #dsolist.append((realImage.GetDirection(), realImage.GetSpacing(), realImage.GetOrigin()))
        image = sitk.GetArrayFromImage(realImage)
        realImage = None
        no_z = image.shape[0] == 1 and False
        if channels:
            if no_z:
                image = image.reshape((image.shape[1], image.shape[2], image.shape[3], 1))
            else:
                image = image.reshape((image.shape[0], image.shape[1], image.shape[2], image.shape[3], 1))
        else:
            if no_z:
                image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
            else:
                image = image.reshape((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
        
        label = sitk.GetArrayFromImage(sitk.ReadImage(labels[name]))
        if classes > 0:
            if no_z:
                label = label.reshape((label.shape[0], label.shape[1], label.shape[2], 1))
            else:
                label = label.reshape((label.shape[0], label.shape[1], label.shape[2],label.shape[3],1))
                #label = label.reshape((1, label.shape[0], label.shape[1], label.shape[2],1))
        else:
            if no_z:
                label = label.reshape((label.shape[0], label.shape[1], label.shape[2]))
            else:

                label = label.reshape((label.shape[0], label.shape[1], label.shape[2],label.shape[3],1))
                #label = label.reshape((1, label.shape[0], label.shape[1], label.shape[2]))
        
        slice_image = sitk.GetArrayFromImage(sitk.ReadImage(slice_images[name]))
        slice_label = sitk.GetArrayFromImage(sitk.ReadImage(slice_labels[name]))
        if classes > 0:
            if no_z:
                slice_label = slice_label.reshape((slice_label.shape[0], slice_label.shape[1], slice_label.shape[2], 1))
            else:
                slice_label = slice_label.reshape((1, slice_label.shape[0], slice_label.shape[1], slice_label.shape[2],1))
        else:
            if no_z:
                slice_label = slice_label.reshape((slice_label.shape[0], slice_label.shape[1], slice_label.shape[2]))
            else:
                slice_label = slice_label.reshape((slice_label.shape[0], slice_label.shape[1], slice_label.shape[2], slice_label.shape[3],1))
                #label = label.reshape((1, label.shape[0], label.shape[1], label.shape[2]))
        slice_image = slice_image.reshape(slice_label.shape)
        
        realT2image = sitk.ReadImage(t2_images[name])
        t2image = sitk.GetArrayFromImage(realT2image)
        realT2image = None
        if channels:
            t2image = t2image.reshape(t2image.shape + (1,))
        t2image = t2image.reshape((1,) + t2image.shape)
        #if t2imagelist.shape[0] == 0:
        #    t2imagelist =np.zeros((len(names),) + t2image.shape)
        #t2imagelist[i] = t2image
        #t2image = None
        if label_selector:
            label = np.array(label==label_selector, dtype=label.dtype)
            slice_label = np.array(slice_label==label_selector, dtype=slice_label.dtype)

        if classes > 1:
            label = to_categorical(label, dtype=label.dtype)
            slice_label = to_categorical(slice_label, dtype=slice_label.dtype)
        #if labellist.shape[0] == 0:
        #    labellist = np.zeros((len(names),) + label.shape)
        #labellist[i] = label
        #label = None
        #namelist.append(name.split()[-1])
        #print(name, image.shape, label.shape)
        #yield image, label, name
        #print(image.shape, slice_image.shape, t2image.shape, label.shape)
        yield image, slice_image, t2image, label, slice_label, name.split()[-1], dso
    #dsolist = np.array(dsolist)

    #return imagelist[sorted_index], t2imagelist[sorted_index], labellist[sorted_index], namelist[sorted_index], dsolist[sorted_index]


def to_categorical(y, num_classes=None, dtype='int32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

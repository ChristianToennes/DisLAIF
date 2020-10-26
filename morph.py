import numpy as np
import scipy.ndimage.morphology as morph
import scipy.ndimage as ndimage
import loader
import SimpleITK as sitk
import os

def create_circle(radius=3):
    return create_ellipse(radius, radius)

def create_ellipse(a, b):
    #n1, n2 = a*2 + 1, b*2+1
    #y,x = np.ogrid[-a:n1-a, -b:n2-b]
    y,x = np.ogrid[-a:a, -b:b]
    mask = ((x*x) / (a*a) + (y*y)/(b*b)) <= 1
    return mask.astype(int)

def morph_segment(image, structure):
    labels = []
    for i in range(image.shape[0]):
        segmentation = image[i] > 0
        segmentation = morph.binary_opening(segmentation, structure=structure)
        segmentation = morph.binary_fill_holes(segmentation, structure=structure)
        labels.append(segmentation)
    return np.array(labels)

if __name__ == "__main__":
    input_path = "./Data/normalized/"
    output_path = "./Data/output/"
    classes = 0
    images, slice_image, t2images, labels, names, dsos = loader.load_input(input_path, classes, channels=False)
    print(names[0])
    for image4d, Name, dso in zip(images, names, dsos):
        for size in [1, 2, 3, 4, 5]:
            labels = []
            for image3d in image4d:
                segmentation = morph_segment(image3d, create_circle(size))
                label_mask, no_features = ndimage.label(segmentation, structure=ndimage.generate_binary_structure(3,2))
                max_mask = 0
                max_i = 0
                for i in range(1,no_features+1):
                    count = np.count_nonzero(label_mask==i)
                    if count > max_i:
                        max_i = count
                        max_mask = i
                label = np.zeros_like(label_mask)
                if max_mask > 0:
                    label[label_mask==max_mask] = 256.0
                #labels.append(label)
                #label = np.argmax(labels, axis=3)*256.
                #if no_features > 0:
                #    label = label_mask*256/no_features
                #else:
                #    label = label_mask*256.0
                label = label[:, :, :, np.newaxis]
                #print(label.shape, no_features, np.max(label))
                labels.append(label)
            Label = sitk.JoinSeries([sitk.GetImageFromArray(label) for label in labels])
            Label.SetDirection(dso[0])
            Label.SetSpacing(dso[1])
            Label.SetOrigin(dso[2])
            if True or np.max(label) > 0:
                if(not os.path.exists(output_path)):
                    os.mkdir(output_path)
                sitk.WriteImage(Label, os.path.join(output_path, Name + "-morph." + str(size) + ".label.mhd"))
                print("written output: " + os.path.join(output_path, Name + "-morph." + str(size) + ".label.mhd"))
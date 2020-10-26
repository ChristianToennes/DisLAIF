import numpy as np
import loader
import SimpleITK as sitk
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morph
from skimage import segmentation, color
from skimage.future import graph
from skimage import morphology
import sklearn
import sklearn.feature_extraction
import sklearn.cluster
import scipy.stats
from morph import create_ellipse
import matplotlib.pyplot as plt
import time
import label_evaluation
import os
import traceback

import logging

from DisLAIF import dante, dante_timing, DisLAIF, OldDisLAIF, DisLAIF_lower, DisLAIF_upper, DisLAIF_middle, DisLAIF_middlep, DisLAIF_middlem, DisLAIF_size, dante_lower, dante_upper, dante_middle, dante_middlep, dante_middlem, DisLAIF_1, DisLAIF_2, DisLAIF_3, DisLAIF_4, DisLAIF_5, DisLAIF_4_6, DisLAIF_4_6_8, DisLAIF_4_3_8, DisLAIF_4_8, DisLAIF_3_8_9, DisLAIF_3_9, DisLAIF_2_9, DisLAIF_gvt_pixel, step4, step4_sato, step4_frangi, step4_hessian, DisLAIF_frangi, DisLAIF_sato, DisLAIF_hessian, DisLAIF_frangi2, DisLAIF_sato2, DisLAIF4, DisLAIF_meijering

logger = logging.getLogger('perfusion.scope')
logger.setLevel(logging.INFO)

file_log_handler = logging.FileHandler('logfile.log')
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler()
logger.addHandler(stdout_log_handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stdout_log_handler.setFormatter(formatter)

def hybrid_aif(a,b,c1,c2,d,t0,t):
    s = a*(t-t0)**b * (np.exp(-c1*(t-t0)) + d*np.exp(-c2*(t-t0)))
    return s

def crop_images(images):
    x = images.shape[4]
    y = images.shape[3]
    left = 0
    #left = int(x*0.1)
    right = x
    #right = int(x*0.9)
    top = 0
    #top = int(y*0.2)
    bottom = y
    #bottom = int(y*0.7)

    return images[:, :, :, top:bottom, left:right, :], (top, bottom, left, right, images.shape)

def uncrop_image(image, crop_box):
    uncropped_image = np.zeros(crop_box[4][1:])
    uncropped_image[:, :, crop_box[0]:crop_box[1], crop_box[2]:crop_box[3], :] = image
    return uncropped_image

def smooth(images):
    smooth_images = ndimage.median_filter(images, size=(1, 1, 1, 3, 3, 1))
    smooth_images = ndimage.uniform_filter(smooth_images, size=(1,2,1,1,1,1))
    #smooth_images = ndimage.gaussian_filter(images, sigma=(0, 0, 0, 1, 1, 0))
    logger.info("{} {}".format(images.shape, smooth_images.shape))
    return smooth_images

def create_weight_graph(image, dist_mat, sigma_g = 1e-2):
    im = image.flatten()
    mat = np.zeros((im.shape[0], im.shape[0]))
    for i in range(im.shape[0]):
        for j in range(i, im.shape[0]):
            w = np.exp(-1.0*np.sum((im[i]-im[j])**2)/sigma_g) * dist_mat[i,j]
            mat[i,j] = w
            mat[j,i] = w
    return mat, im

def create_dist_mat(image, sigma_d = 1e2):
    im = image.flatten()
    dist_mat = np.zeros((im.shape[0], im.shape[0]))
    for i in range(im.shape[0]):
        for j in range(i, im.shape[0]):
            a = np.unravel_index([i], image.shape)
            b = np.unravel_index([j], image.shape)
            d = (a[0]-b[0])**2 + (a[1]-b[1])**2
            if d > 2:
                dist_mat[i,j] = np.inf
                dist_mat[j,i] = np.inf
            else:
                w = np.exp(-d/sigma_d)
                dist_mat[i,j] = w
                dist_mat[j,i] = w
    return dist_mat

def ncut(image):
    result4d = []
    logger.info(image.shape)
    im = image.flatten()
    dist_mat = False
    for t in range(image.shape[0]):
        result3d = []
        logger.info(t)
        for z in range(image.shape[1]):
            logger.info(z)
            if not dist_mat:
                dist_mat = create_dist_mat(image[t,z])
            mat, im = create_weight_graph(image[t,z], dist_mat)
            #U,sigma,V = np.linalg.svd(mat, full_matrices=True)
            #e_index = np.argsort(sigma)[1]
            #eigen = U[:, e_index]

            clustered = graph.cut_normalized(im, mat)
            clustered = clustered.reshape(image.shape)
            result3d.append(clustered)
        result4d.append(np.array(result3d))
    return np.array(result4d)

def ncut_sklearn(image):
    result4d = []
    im = image.reshape(image.shape[:4])
    dist_mat = False
    for t in range(im.shape[0]):
        result3d = []
        for z in range(im.shape[1]):
            #g = sklearn.feature_extraction.image.img_to_graph(im[t,z])
            if not dist_mat:
                dist_mat = create_dist_mat(image[t,z])
            g = create_weight_graph(image[t,z], dist_mat)
            labels = sklearn.cluster.spectral_clustering(g, 2)
            result3d.append(labels.reshape(labels.shape + (1,)))
        result4d.append(np.array(result3d))
    return np.array(result4d)

def HIER(image):
    return image

def mouridsen2006(image):
    
    img = image.reshape((image.shape[0], -1))
    
    c = img / img[0]
    c[np.isnan(c)] = 1
    c = -np.log(c)
    c[np.isnan(c)] = 0
    
    
    auc = np.sum(c, axis=0)
    
    img1 = c[:, auc>np.quantile(auc, 0.9)]
    
    cdt = img1[1:]-img1[:-1]
    
    cdtdt = cdt[1:]-cdt[:-1]
    
    r = np.sum(cdtdt*cdtdt, axis=0)
    
    img2 = img1[:, r>np.quantile(r, 0.75)]
    
    img3 = np.swapaxes(img2, 0, 1)
    
    (_centroid, label, _inertia) = sklearn.cluster.k_means(img3, 5)
    
    min_c = img3.shape[1]
    min_c_i = 0
    for i in range(5):
        if np.count_nonzero(label==i) == 0:
            continue
        print(img3[label==i].shape)
        avg = np.argmax(img3[label==i], axis=1)
        print(avg.shape)
        
        cur = np.average(avg)
        print(cur)
        if cur > 4 and cur < min_c:
            min_c = cur
            min_c_i = i
    
    img4 = img3[label==min_c_i]
    
    (_centroid, label1, _inertia) = sklearn.cluster.k_means(img4, 5)
    min_c = img4.shape[1]
    min_c_i1 = 0
    for i in range(5):
        if np.count_nonzero(label1==i) == 0:
            continue
        cur = np.argmax(np.average(img4[label1==i], axis=0))
        
        if cur > 4 and cur < min_c:
            min_c = cur
            min_c_i1 = i
    
    mask = np.zeros((img4.shape[1], img4.shape[0]), img4.dtype)
    mask[:,label1==min_c_i1] = 1
    mask1 = np.zeros_like(img2)
    mask1[:,label==min_c_i] = mask
    mask2 = np.zeros_like(img1)
    mask2[:,r>np.quantile(r, 0.75)] = mask1
    mask3 = np.zeros_like(c)
    mask3[:,auc>np.quantile(auc, 0.9)] = mask2

    mask4 = mask3.reshape(image.shape)
    return mask4, None

def shi2013(image):
    # calculate contrast agent concentration curve
    Sgd = np.swapaxes(image.reshape((image.shape[0], -1)), 0,1)
    S0 = np.average(Sgd[:,:3], axis=1)[:,np.newaxis]
    r1 = 1
    T10 = 1
    Hct = 0
    #C = 1/(r1*T10*(1-Hct))*(Sgd-S0)/S0
    C = Sgd
    # exclude background voxel
    Cmax_average = np.average(np.max(C, axis=1))
    Cmax_sd = np.std(np.max(C, axis=1))
    threshold = Cmax_average + 3.1415 * Cmax_sd
    mask_background = np.max(C, axis=1)>threshold
    C = C[mask_background]
    # calculate area under the curve
    Pauc = np.sum(C, axis=1)
    # exclude voxel with smallest area
    auc_threshold = np.quantile(Pauc, 0.2)
    mask_auc = Pauc>auc_threshold
    C = C[mask_auc]
    # calculate peak value
    peaks = np.max(C, axis=1)
    peak_time = np.argmax(C, axis=1)
    # exclude voxel with smallest peak
    peak_threshold = np.quantile(peaks, 0.2)
    peak_time_threshold = np.quantile(peak_time, 0.5)
    mask_peak = peaks>peak_threshold
    mask_peak = np.bitwise_and(peak_time<peak_time_threshold, mask_peak)
    C = C[mask_peak]
    # predifine number of clusters
    # determining multi grid search space
    # cluster with affinity propagation
    
    no_clusters = min(20, max(np.count_nonzero(C)//5, 1))
    logger.info("{} {}".format(no_clusters, np.count_nonzero(C)))
    _centroid, labels, _inertia = sklearn.cluster.k_means(C, no_clusters)
    
    label_image = np.zeros_like(Sgd)
    label_peak = np.zeros(mask_peak.shape, dtype=int)
    #label_peak[mask_peak] = np.repeat(labels[:,np.newaxis], Sgd.shape[1], axis=1)
    label_peak[mask_peak] = labels
    label_peak_time = np.zeros(label_peak.shape + (label_image.shape[1],), dtype=int)

    for i, slices in enumerate(ndimage.find_objects(label_peak), start=1):
        if not slices:
            continue
        object_mask = label_peak==i
        timepoint = int(np.median(peak_time[object_mask]))
        label_peak_time[object_mask, timepoint] = i

    label_auc = np.zeros(mask_auc.shape + (label_image.shape[1],), dtype=int)
    label_auc[mask_auc] = label_peak_time
    label_image[mask_background] = label_auc
    label_image = np.swapaxes(label_image, 0,1).reshape(image.shape)
    
    structure = np.ones((3,3,3,3,3))
    labels, _no_labels = ndimage.label(label_image, structure=structure)

    new_label_image = np.zeros_like(label_image)
    for i, slices in enumerate(ndimage.find_objects(labels), start=1):
        if not slices:
            continue
        object_mask = labels[slices]==i
        hist = np.count_nonzero(object_mask, axis=(1,2,3,4))
        zs = []
        for z,h in enumerate(hist):
            zs = zs + [z]*h
        timepoint = int(np.median(zs))
        object_mask = np.any(object_mask, axis=0)
        new_label_image[slices][timepoint, object_mask] = i

    new_label_image[new_label_image>0] = 1

    return new_label_image, None

def AP(C):
    r = np.zeros((C.shape[0],C.shape[0]))
    a = np.zeros((C.shape[0],C.shape[0]))
    s = -np.sqrt(np.array([[np.dot(C[i]-C[k], C[i]-C[k]) for k in range(C.shape[0])] for i in range(C.shape[0])]))
    #s = np.random.rand(C.shape[0], C.shape[0])
    c = np.zeros(C.shape[0], dtype=int)

    A = np.bitwise_and(np.ones((C.shape[0], C.shape[0]), dtype=bool), np.bitwise_not(np.identity(C.shape[0], dtype=bool)))
    #A = np.repeat(A, C.shape[0])
    Id = np.identity(r.shape[0], dtype=bool)

    for step in range(30):
        if step%10 == 0:
            logger.info(step)
        r = 0.9*r + 0.1*(s - np.max((a + s)*A, axis=1))
        mask = r>0
        mask = np.bitwise_and(mask, np.bitwise_not(Id))
        a_id_new = 0.9*a[Id] + 0.1*np.sum(r[mask], axis=0)
        a_new = r[Id] + np.sum(r[mask], axis=0)
        a_new[a_new<0] = 0
        a = 0.9*a + 0.1*a_new
        a[Id] = a_id_new
        c = np.argmax(r+a, axis=1)
    return c

def gamma_variate(image):

    image = scipy.ndimage.gaussian_filter1d(image, 0.7, axis=0)
    image = scipy.ndimage.gaussian_filter1d(image, 0.7, axis=1)
    image = scipy.ndimage.gaussian_filter1d(image, 0.7, axis=2)
    image = scipy.ndimage.gaussian_filter1d(image, 0.7, axis=3)

    full_im = np.swapaxes(image.reshape((image.shape[0], -1)), 0,1)

    full_im = full_im - np.min(full_im)
    full_tmax = np.argmax(full_im[:,1:-1], axis=1)
    m = np.bitwise_and(full_tmax>0, full_im[range(full_tmax.shape[0]),full_tmax]>0)
    indices = np.arange(0, full_im.shape[0])
    tmax = full_tmax[m]
    
    im = full_im[m, 1:]
    
    tl = im.shape[1]-1
    im_dt = np.zeros_like(im)
    im_dtdt = np.zeros_like(im)
    im_dt[:,:-1] = im[:,1:]-im[:,:-1]
    im_dtdt[:,:-1] = im_dt[:,1:]-im_dt[:,:-1]

    index = np.ravel_multi_index((16,66,160,0), image.shape[1:])
    index = np.argmax(indices[m]==index)

    for i, t in enumerate(tmax):
        im_dtdt[i][t:] = 0
    t0 = np.argmax(im_dtdt, axis=1)

    m1 = np.bitwise_and(
            np.bitwise_and(t0<tmax, im_dtdt[range(t0.shape[0]), t0]>0), 
            np.bitwise_and( (tmax-t0) < 4, t0<(full_im.shape[1]*0.3) )
        )

    im = im[m1]
    t0 = t0[m1]
    tmax = tmax[m1]
    
    k = im[:,tl] / (im[range(tmax.shape[0]),tmax])
    k = np.log(k)
    a_denom = (1 + np.log((tl-t0)/(tmax-t0)) - (tl-t0)/(tmax-t0) )
    lowerA = (k / a_denom)
    upperA = (tmax-t0)
    
    logIm = im / np.repeat(im[range(tmax.shape[0]),tmax,np.newaxis], im.shape[1], axis=1) 
    logIm = np.log(logIm+1e-10)
    alpha = logIm / np.repeat( a_denom[:,np.newaxis], im.shape[1], axis=1)
    alpha = np.array([np.average(alpha[t0:tmax]) for alpha,t0,tmax in zip(alpha,t0,tmax) ]) 
    mask1 = np.zeros(im_dt.shape[0], dtype=float)
    mask1[m1] = np.bitwise_and(alpha>lowerA, alpha<upperA, dtype=int)
    
    mask = np.zeros(full_im.shape[0], dtype=float)
    mask[m] = mask1
    mask = mask.reshape((1,) + image.shape[1:])
    return mask, None

def FastAP(image):
    result4d = []
    im = image.reshape(image.shape[:4])
    for t in range(im.shape[0]):
        logger.info(str(t) + "/" + str(im.shape[0]))
        result3d = []
        for z in range(im.shape[1]):
            logger.info(str(z) + "/" + str(im.shape[1]))
            g = sklearn.feature_extraction.image.img_to_graph(im[t,z], return_as=np.ndarray)
            _clusters, labels = sklearn.cluster.affinity_propagation(g)
            result3d.append(labels.reshape(labels.shape + (1,)))
        result4d.append(np.array(result3d))
    return np.array(result4d), None

def overlap(old_slices, slices):
    x_length = 0
    y_length = 0
    if old_slices[0] < slices[0]:
        if old_slices[1] > slices[1]:
            x_length = 1.0
        else:
            if (old_slices[1] > slices[0]):
                x_length = (old_slices[1]-slices[0]) / (slices[1] - slices[0])
            else:
                x_length = 0
    else:
        if old_slices[0] < slices[1]:
            if slices[1] > old_slices[0]:
                x_length = (slices[1]-old_slices[0]) / (slices[1] - slices[0])
            else:
                x_length = 0
        else:
            x_length = 1.0
    if old_slices[2] < slices[2]:
        if old_slices[3] > slices[3]:
            y_length = 1.0
        else:
            if old_slices[3] > slices[2]:
                y_length = (old_slices[3]-slices[2]) / (slices[3] - slices[2])
            else:
                y_length
    else:
        if old_slices[2] < slices[3]:
            if slices[3] > old_slices[2]:
                y_length = (slices[3] - old_slices[2]) / (slices[3] - slices[2])
            else:
                y_length = 0
        else:
            y_length = 1.0
    overlap = x_length*y_length
    return overlap

def chen2008(image):
    peak_times = np.argmax(image[1:], axis=0)+1
    peaks = np.max(image[1:], axis=0)
    sorted_index = np.argsort(peaks.flatten())
    sorted_image = peaks.flatten()[sorted_index]
    mean_value = np.mean(sorted_image)
    index = np.searchsorted(sorted_image, mean_value, 'right')
    mask = np.zeros_like(peaks)
    mask[np.unravel_index(sorted_index[index:], peaks.shape)] = 1
    
    slope = (peaks*mask)/peak_times

    slope[peak_times==0] = 0

    mean_value = np.mean(slope[slope!=0])
    quantile = np.quantile(slope[slope!=0], 0.9)
    
    mask = np.zeros_like(slope)
    mask[slope>quantile] = 1
    
    tmask = np.zeros_like(image)
    tmask[peak_times[mask==1], mask==1] = 1

    structure = np.array([
        [
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]],
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,1,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]],
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]]
        ],
        [
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]],
        [[[0,0,0], [0,1,0], [0,0,0]],[[0,0,0], [0,1,0], [0,0,0]],[[0,0,0], [0,1,0], [0,0,0]]],
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]]
        ],
        [
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]],
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,1,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]],
        [[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]],[[0,0,0], [0,0,0], [0,0,0]]]
        ]
    ])

    structure[1, 1, :, :, 1] = 1

    label_mask, _no_features = ndimage.label(tmask, structure=structure)
    
    czs = {}
    new_i = 1
    for i, slices in enumerate(ndimage.find_objects(label_mask), start=1):
        label_i = label_mask[slices[0].start, slices[1].start]==i
        count = np.count_nonzero(label_i)
        if count < 3:
            label_mask[slices[0].start, slices[1].start][label_i] = 0
        else:
            area = (slices[2].stop-slices[2].start)*(slices[3].stop-slices[3].start)
            ellipsis = create_ellipse((slices[2].stop-slices[2].start)/2, (slices[3].stop-slices[3].start)/2)
            ellipsis[ellipsis==0] = -1
            ellipsis = ellipsis
            cz = np.sum(label_i[slices[2].start:slices[2].stop,slices[3].start:slices[3].stop,0]*ellipsis) / area
            best_i = new_i
            best_overlap = 0
            if slices[1].start > 0:
                for old_i in czs:
                    current_overlap = overlap((slices[2].start, slices[2].stop, slices[3].start, slices[3].stop), czs[old_i][3][-1])
                    if current_overlap > best_overlap:
                        best_i = old_i
                        best_overlap = current_overlap
            
            if best_i in czs:
                czs[best_i][0].append(count)
                czs[best_i][1].append(area)
                czs[best_i][2].append(cz)
                czs[best_i][3].append((slices[2].start, slices[2].stop, slices[3].start, slices[3].stop))
                czs[best_i][4].append(slices[1].start)
            else:
                czs[best_i] = ([count], [area], [cz], [(slices[2].start, slices[2].stop, slices[3].start, slices[3].stop)], [slices[1].start])
                new_i = new_i + 1
            label_mask[slices[0].start, slices[1].start][label_i] = best_i
    label = np.array(label_mask>1, dtype=image.dtype)

    return label, None


def chen2008_(image):
    #peak_times = np.argmax(image[1:], axis=0)+1
    peak_time = 7
    t1 = 0.2
    t2 = 0.1
    peaks = np.max(image[1:], axis=0)
    sorted_index = np.argsort(peaks.flatten())
    sorted_image = peaks.flatten()[sorted_index]
    mean_value = np.mean(sorted_image)
    index = np.searchsorted(sorted_image, mean_value, 'right')
    mask = np.zeros_like(peaks)
    mask[np.unravel_index(sorted_index[index:], peaks.shape)] = 1
    
    #slope = (peaks*mask)/peak_times

    #slope[peak_times==0] = 0

    #mean_value = np.mean(slope[slope!=0])
    #quantile = np.quantile(slope[slope!=0], 0.9)
    
    #mask = np.zeros_like(slope)
    #mask[slope>quantile] = 1
    
    #tmask = np.zeros_like(image)
    #tmask[peak_times[mask==1], mask==1] = 1

    slope = (image[peak_time]/image[0]-1) / peak_time
    mask[slope<t1] = 0

    structure = np.zeros((3,3,3,3))

    structure[1, :, :, 1] = 1

    label_mask, _no_features = ndimage.label(mask, structure=structure)
    
    czs = []
    new_i = 1
    for i, slices in enumerate(ndimage.find_objects(label_mask), start=1):
        label_i = label_mask[slices[0].start]==i
        count = np.count_nonzero(label_i)
        #print(label_i.shape, count)
        if count < 10:
            label_mask[slices[0].start][label_i] = 0
        else:
            i_mask = np.zeros_like(image, dtype=bool)
            i_mask[:,slices[0].start] = label_i
            ps = np.argmax(image[:, i_mask[0]], axis=0)
            #print(i_mask.shape, image[:, i_mask[0]].shape, ps, np.quantile(ps, 0.2))
            tp = np.mean(ps[ps<=np.quantile(ps, 0.2)])

            if np.mean((image[int(tp)]/image[0]-1)/tp) < t1:
                label_mask[slices[0].start][label_i] = 0
            else:
                A = np.count_nonzero(label_i)
                P = 0
                for index, x in np.ndenumerate(label_i):
                    if x == 1:
                        if index[0] == 0:
                            P += 1
                        elif label_i[index[0]-1, index[1]] == 0:
                            P += 1
                        if index[0] == label_i.shape[0]-1:
                            P += 1
                        elif label_i[index[0]+1, index[1]] == 0:
                            P += 1

                        if index[1] == 0:
                            P += 1
                        elif label_i[index[0], index[1]-1] == 0:
                            P += 1
                        if index[1] == label_i.shape[1]-1:
                            P += 1
                        elif label_i[index[0], index[1]+1] == 0:
                            P += 1

                slice_t = ( slices[1].start, slices[1].stop, slices[2].start, slices[2].stop )
                czs.append([len(czs), A/(P*P), slices, slice_t, label_i])
                for cz in czs:
                    if overlap(cz[3], slice_t):
                        change = czs[-1][0]
                        for i, cz2 in enumerate(czs):
                            if cz2[0] == change:
                                czs[i][0] = cz[0]

    #print(len(czs))

    cyl = {}
    max_i = 0
    new_i = 0
    for cz in czs:
        if cz[0] not in cyl:
            cyl[cz[0]] = cz[1]
        else:
            cyl[cz[0]] += cz[1]
        if cyl[cz[0]] > max_i:
            max_i = cyl[cz[0]]
            new_i = cz[0]
    
    #print(cyl)
    
    slices = sorted([(cz[2], cz[4]) for cz in czs if cz[0] == new_i])
    if len(slices) > 0:
        #print(len(slices), slices[0], slices[-1])

        z_indexe = np.array([s[0].start for s, _ in slices])
        mid_index = np.argmin(np.abs(z_indexe-((np.max(z_indexe)-np.min(z_indexe))/2)))

        mid_slice = slices[mid_index][0]

        z_index = z_indexe[mid_index]

        i_mask = np.zeros_like(image, dtype=bool)
        print(i_mask.shape, z_index, mid_slice, slices[mid_index][1].shape)
        i_mask[:,z_index] = slices[mid_index][1]
        ps = np.argmax(image[:, i_mask[0]], axis=0)
        #print(i_mask.shape, image[:, i_mask[0]].shape, ps, np.quantile(ps, 0.2))
        tp = int(np.mean(ps[ps<=np.quantile(ps, 0.2)]))

        t = np.average(image[mid_slice]) - np.std(image[mid_slice])

        seed = (mid_slice[0].start+(mid_slice[0].stop-mid_slice[0].start)//2, mid_slice[1].start+(mid_slice[1].stop-mid_slice[1].start)//2, mid_slice[2].start+(mid_slice[2].stop-mid_slice[2].start)//2)

        print(image[tp].shape, seed)

        #print(mid_slice, seed, t, image.shape)

        label = grow(image[tp], seed, t)

        ret = np.zeros_like(image, dtype=int)
        ret[tp] = label
    
        return ret, None
    else:
        return np.zeros_like(image[1:2]), None

def grow(img, seed, threshold):
    """
    img: ndarray, ndim=3
        An image volume.
    
    seed: tuple, len=3
        Region growing starts from this point.

    t: int
        The image neighborhood radius for the inclusion criteria.
    """
    t = 1
    seg = np.zeros(img.shape, dtype=int)
    checked = np.zeros(seg.shape, dtype=bool)

    seg[seed] = 1
    checked[seed] = True
    needs_check = get_nbhd(seed, checked, img.shape)

    while len(needs_check) > 0:
        pt = needs_check.pop()

        # Its possible that the point was already checked and was
        # put in the needs_check stack multiple times.
        if checked[pt]: continue

        checked[pt] = True

        # Handle borders.
        imin = max(pt[0]-t, 0)
        imax = min(pt[0]+t, img.shape[0]-1)
        jmin = max(pt[1]-t, 0)
        jmax = min(pt[1]+t, img.shape[1]-1)
        kmin = max(pt[2]-t, 0)
        kmax = min(pt[2]+t, img.shape[2]-1)

        if img[pt] >= threshold:
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            seg[pt] = 1
            needs_check += get_nbhd(pt, checked, img.shape)

    return seg

def get_nbhd(pt, checked, dims):
    nbhd = []
    if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2], 0]:
        nbhd.append((pt[0]-1, pt[1], pt[2]))
    if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2], 0]:
        nbhd.append((pt[0], pt[1]-1, pt[2]))
    if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1, 0]:
        nbhd.append((pt[0], pt[1], pt[2]-1))

    if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2], 0]:
        nbhd.append((pt[0]+1, pt[1], pt[2]))
    if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2], 0]:
        nbhd.append((pt[0], pt[1]+1, pt[2]))
    if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1, 0]:
        nbhd.append((pt[0], pt[1], pt[2]+1))

    return nbhd

def filter_gvt(im, timestep_mask, timestep, structure):
    labels, no_features = ndimage.label(timestep_mask, structure=structure)
    
    filter_mask = np.zeros_like(timestep_mask)

    if no_features>0:
        alphas = []
        for i, slices in enumerate(ndimage.find_objects(labels), start=1):
            roi = labels[slices[0].start]==i
            p = np.average(im[:,roi], axis=1)
            tmax = np.argmax(p)
            t0 = timestep-2
            if tmax <= t0:
                tmax = np.argmax(p[t0+1:]) + t0 + 1
            p = scipy.ndimage.gaussian_filter1d(p, 0.7)
            alpha = [np.log(1e-30 + p[t]/p[tmax])/(1 + np.log((t - t0)/(tmax-t0)) - (t-t0)/(tmax-t0) + 1e-30) for t in range(t0+1, min(tmax+10, p.shape[0]))]
            alpha = np.average(alpha)
            k = p[-1] / np.max(p)
            k = np.log(k+1e-10)
            a_denom = (1 + np.log((p.shape[0]-1 - t0)/(tmax-t0)) - (p.shape[0]-1-t0)/(tmax-t0))
            lowerA = (k / (a_denom + 1e-10))
            alphas.append(alpha)
            if alpha > tmax-t0 or alpha < lowerA:
                filter_mask[tmax, roi] = 2
            else:
                filter_mask[tmax, roi] = 1

    return np.array(filter_mask, dtype=int)
  
def max05total(im):
    #step 1

    q99 = np.quantile(im[:20], 0.99)
    mask = np.zeros_like(im)
    mask[im>q99] = 1

    #step2
    
    structure = np.zeros((3,3,3,3,3))
    structure[1, 1, 1, :, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, :, 1, 1, 1] = 1

    mask = scipy.ndimage.morphology.binary_opening(mask, structure)

    #step 3

    structure = np.zeros((3,3,3,3,3))
    structure[1, :, :, :, 1] = 1

    early_mask, early_timestep = filter_timestep(im, mask, structure)

    #step 4
    labels, _no_features = ndimage.label(mask, structure=structure)

    structure = np.zeros((3,3,3,3,3))
    structure[1, 1, 1, :, 1] = 1
    structure[1, 1, :, 1, 1] = 1

    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        label_mask = labels[slices]==i
        subLabels, _ = ndimage.label(label_mask, structure=structure)
        czs = 0
        if (slices[1].stop-slices[1].start) > labels.shape[1]/5:
            czs = []
            for j, subSlices in enumerate(ndimage.find_objects(subLabels), start=1):
                #logger.info(subSlices)
                #area = max((subSlices[2].stop-subSlices[2].start),(subSlices[3].stop-subSlices[3].start))**2
                area = (subSlices[2].stop-subSlices[2].start)*(subSlices[3].stop-subSlices[3].start)
                if area < 5:
                    mask[slices][subSlices][subLabels[subSlices]==j] = 0
                else:
                    area = max((subSlices[2].stop-subSlices[2].start),(subSlices[3].stop-subSlices[3].start))**2
                    #ellipsis = create_ellipse((subSlices[2].stop-subSlices[2].start)/2, (subSlices[3].stop-subSlices[3].start)/2)
                    #ellipsis[ellipsis==0] = -1
                    #ellipsis = ellipsis
                    count = np.count_nonzero(subLabels[subSlices]==j)
                    cz = count / area
                    #cz = np.sum(subLabels[subSlices[0].start, subSlices[1].start, subSlices[2].start:subSlices[2].stop, subSlices[3].start:subSlices[3].stop, subSlices[4].start]*ellipsis) / area
                    czs.append(cz)
            czs = np.array(czs)
            czs = np.count_nonzero(np.bitwise_and(czs > 0.6, czs < 0.8))

        if czs < (slices[1].stop-slices[1].start)/4.0:
            mask[slices][label_mask] = 0
            #ml[slices][label_mask]=0

    structure = np.zeros((3,3,3,3,3))
    structure[1, 1, 1, :, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, :, 1, 1, 1] = 1

    #step 5
    timestep_mask, timestep = filter_timestep(im, mask, structure)

    #filter_mask = filter_gvt(im, timestep_mask, -1, structure)
    #logger.info("{} {} {} {}".format(early_timestep, np.count_nonzero(early_mask), timestep, np.count_nonzero(timestep_mask)))

    use_early = (early_timestep > 0) and (np.count_nonzero(early_mask) > 0) and (early_timestep < timestep)

    # step 6
    if use_early or np.count_nonzero(timestep_mask) == 0:
        #return early_mask
        filter_mask = filter_gvt(im, early_mask, early_timestep, structure)
    else:
        #return timestep_mask
        filter_mask = filter_gvt(im, timestep_mask, timestep, structure)
    #logger.info(np.count_nonzero(filter_mask==1))
    
    if use_early or np.count_nonzero(filter_mask[timestep]==1) == 0: 
        timestep = early_timestep
    
    filter_mask[:timestep] = 0
    filter_mask[timestep+1:] = 0
    filter_mask = np.array(filter_mask, dtype=int)

    #step 7

    structure = np.zeros((5,5,5,5,5))
    structure[2, :, :, :, 2] = 1
    filter_mask = scipy.ndimage.morphology.binary_dilation(mask, structure)

    #step 8

    img = (im*filter_mask)[timestep]
    seed = np.unravel_index(np.argmax(img), img.shape)
    t = 1
    grow_mask = np.zeros_like(im)
    grow_mask[timestep] = grow(img, seed, t)

    filter_mask = filter_mask - grow_mask[timestep]
    filter_mask[filter_mask<0] = 0

    img = (im*filter_mask)[timestep]
    seed = np.unravel_index(np.argmax(img), img.shape)
    t = 1
    grow_mask[timestep] = grow_mask[timestep] + grow(img, seed, t)
    grow_mask[grow_mask>1] = 1

    #step 9

    structure = np.zeros((3,3,3,3))
    structure[1, :, :, 1] = 1
    grow_mask[timestep] = np.array(scipy.ndimage.morphology.binary_fill_holes(grow_mask[timestep], structure=structure), dtype=int)

    return grow_mask, timestep


def filter_timestep(image, mask, structure):
    labels, _no_features = ndimage.label(mask, structure=structure)
    #logger.info(no_features)
    im = image*mask
    #dt = np.sum((mask[1:]-mask[:-1])>0, axis=(1,2,3,4))
    c = np.swapaxes(im.reshape(im.shape[0], -1), 0,1)
    #peaks = np.max(c, axis=1)
    #peaks = peaks[peaks>0]
    #threshold_peak = np.quantile(peaks, 0.9)
    dt = np.argmax(c, axis=1)
    dt = dt[dt>0]
    dt = dt[dt<im.shape[0]/4]
    #quant = np.quantile(dt, 0.9)
    hist = np.histogram(dt, range(0, im.shape[0]//4 + 2))[0]
    
    t = np.argmax(hist)
    timestep_mask = np.zeros_like(mask, dtype=int)
    timestep_mask[t] = mask[t]
    return timestep_mask, t

def parker2003(im):
    peaks = im==np.max(im, axis=0)
    peaks[20:] = np.zeros_like(peaks[20:])
    q95 = np.quantile(im[peaks], 0.95)
    use = (im*peaks) > q95
    return use, None
    
def max05(im):
    result4d = []
    image = im
    q95 = np.quantile(image[:20], 0.95)
    for t in range(image.shape[0]):
        result3d = []
        for z in range(image.shape[1]):
            mask = np.zeros_like(image[t,z])
            threshold = max(np.quantile(image[t,z], 0.99), q95)
            mask[image[t,z]>=threshold] = 1
            result3d.append(mask)
        result4d.append(np.array(result3d))
    return np.array(result4d), None

def middle(label):
    mid = np.zeros_like(label)
    start = -1
    end = -1
    for i in range(label.shape[1]):
        if np.count_nonzero(label[:,i]) != 0 and start == -1:
            start = i
            break
    for i in range(label.shape[1]-1, 0, -1):
        if np.count_nonzero(label[:,i]) != 0 and end == -1:
            end = label.shape[1]
            break
    mid_index = (end-start)//2
    print(label.shape, mid.shape, start, end, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, None

def middlep(label):
    mid = np.zeros_like(label)
    start = -1
    end = -1
    for i in range(label.shape[1]):
        if np.count_nonzero(label[:,i]) != 0 and start == -1:
            start = i
            break
    for i in range(label.shape[1]-1, 0, -1):
        if np.count_nonzero(label[:,i]) != 0 and end == -1:
            end = label.shape[1]
            break
    mid_index = (end-start)//2 + 3
    print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, None

def middlem(label):
    mid = np.zeros_like(label)
    start = -1
    end = -1
    for i in range(label.shape[1]):
        if np.count_nonzero(label[:,i]) != 0 and start == -1:
            start = i
            break
    for i in range(label.shape[1]-1, 0, -1):
        if np.count_nonzero(label[:,i]) != 0 and end == -1:
            end = label.shape[1]
            break
    mid_index = (end-start)//2 - 3
    print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, None
def upper(label):
    mid = np.zeros_like(label)
    start = -1
    end = -1
    for i in range(label.shape[1]):
        if np.count_nonzero(label[:,i]) != 0 and start == -1:
            start = i
            break
    for i in range(label.shape[1]-1, 0, -1):
        if np.count_nonzero(label[:,i]) != 0 and end == -1:
            end = label.shape[1]
            break
    mid_index = 3*(end-start)//4
    print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, None

def lower(label):
    mid = np.zeros_like(label)
    start = -1
    end = -1
    for i in range(label.shape[1]):
        if np.count_nonzero(label[:,i]) != 0 and start == -1:
            start = i
            break
    for i in range(label.shape[1]-1, 0, -1):
        if np.count_nonzero(label[:,i]) != 0 and end == -1:
            end = label.shape[1]
            break
    mid_index = (end-start)//4
    print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, None

def left(label):
    llabel = np.zeros_like(label)
    llabel[:,1:,:] = label[:,:-1,:]
    return llabel, None

def right(label):
    llabel = np.zeros_like(label)
    llabel[:,:-1,:] = label[:,1:,:]
    return llabel, None

def up(label):
    llabel = np.zeros_like(label)
    llabel[:,:,1:] = label[:,:,:-1]
    return llabel, None

def down(label):
    llabel = np.zeros_like(label)
    llabel[:,:,:-1] = label[:,:,1:]
    return llabel, None

def fore(label):
    llabel = np.zeros_like(label)
    llabel[1:,:,:] = label[1:,:,:]
    return llabel, None

def back(label):
    llabel = np.zeros_like(label)
    llabel[:-1,:,:] = label[:-1,:,:]
    return llabel, None

def leftSide(label):
    llabel = np.zeros_like(label)
    llabel[:,:,:,:llabel.shape[3]//2] = label[:,:,:,:label.shape[3]//2]
    return llabel, None

def rightSide(label):
    llabel = np.zeros_like(label)
    llabel[:,:,:,llabel.shape[3]//2:] = label[:,:,:,label.shape[3]//2:]
    return llabel, None

def noop(image):
    return image, None

def find_aif(input_path, output_path, classes, algos, acc=[], times={}, write_files=True, use_processed=True):
    last_timestep = None
    for image, slice_image, t2image, label, slice_label, name, dso in loader.load_input(input_path, classes, label_selector=2, use_processed=use_processed):

        #print(image.shape, slice_image.shape, label.shape, slice_label.shape)
        #image = image[0]
        #slice_image = slice_image[0]
        #label = label[0]
        #slice_label = slice_label[0]

        labels = []
        if label.shape[0] > 70:
            label = label[:69]
        if image.shape[0] > 70:
            image = image[:69]
        for i in range(slice_label.shape[0]):
            labels.append(sitk.GetImageFromArray(np.array(slice_label[i][...,0]>0, dtype=np.int8)))

        #print(image.shape, slice_image.shape, label.shape, slice_label.shape)

        acc.append(label_evaluation.calc_metrics(image, slice_label, labels, slice_label, labels, name, "target", 0))
        logger.info("{} {} {}".format(acc[-1]['modelName'], acc[-1]['acc'],acc[-1]['aif_mse']))

        MaskImage = None
        if slice_label.shape[0] == 1:
            MaskImage = sitk.GetImageFromArray(slice_label[0])
            MaskImage.SetDirection(dso[0][0:3] + dso[0][4:7] + dso[0][8:11])
            MaskImage.SetSpacing(dso[1][:-1])
            MaskImage.SetOrigin(dso[2][:-1])
        else:
            mask_images = []
            for j in range(slice_label.shape[0]):
                mask_images.append(sitk.GetImageFromArray(slice_label[j]))
            MaskImage = sitk.JoinSeries(mask_images)
            MaskImage.SetDirection(dso[0])
            MaskImage.SetSpacing(dso[1])
            MaskImage.SetOrigin(dso[2])
        
        Image = None
        mask_images = []
        if image.shape[0] > 70:
            image = image[::2]
        for j in range(image.shape[0]):
            mask_images.append(sitk.GetImageFromArray(image[j]))
        Image = sitk.JoinSeries(mask_images)
        Image.SetDirection(dso[0])
        Image.SetSpacing(dso[1])
        Image.SetOrigin(dso[2])

        if(not os.path.exists(output_path)):
            os.mkdir(output_path)

        if write_files:
            sitk.WriteImage(MaskImage, os.path.join(output_path, name + '.aif_mask.target.nrrd'))
            logger.info("written file: " + output_path + name + ".aif_mask.target.nrrd")
            sitk.WriteImage(Image, os.path.join(output_path, name + '.aif_mask.noop.nrrd'))
            logger.info("written file: " + output_path + name + ".aif_mask.noop.nrrd")

        image = image-np.mean(image)
        image = image / np.var(image)

        structure = np.zeros((3,3,3,3,3))
        structure[1, :, 1, 1, 1] = 1
        structure[1, 1, :, 1, 1] = 1
        structure[1, 1, 1, :, 1] = 1
        dilate_target = lambda i, target=np.array(label[:]), structure=structure: (scipy.ndimage.morphology.binary_dilation(target, structure), None)
        dilate_target.__name__ = "target_dilated"
        erode_target = lambda i, target=np.array(label[:]), structure=structure: (scipy.ndimage.morphology.binary_erosion(target, structure), None)
        erode_target.__name__ = "target_eroded"
        middle_target = lambda i, target=np.array(label[:]): middle(target)
        middle_target.__name__ = "target_middle"
        middlep_target = lambda i, target=np.array(label[:]): middlep(target)
        middlep_target.__name__ = "target_middlep"
        middlem_target = lambda i, target=np.array(label[:]): middlem(target)
        middlem_target.__name__ = "target_middlem"
        upper_target = lambda i, target=np.array(label[:]): upper(target)
        upper_target.__name__ = "target_upper"
        lower_target = lambda i, target=np.array(label[:]): lower(target)
        lower_target.__name__ = "target_lower"
        left_target = lambda i, target=np.array(label[:]): left(target)
        left_target.__name__ = "target_left"
        right_target = lambda i, target=np.array(label[:]): right(target)
        right_target.__name__ = "target_right"
        up_target = lambda i, target=np.array(label[:]): up(target)
        up_target.__name__ = "target_up"
        down_target = lambda i, target=np.array(label[:]): down(target)
        down_target.__name__ = "target_down"
        fore_target = lambda i, target=np.array(label[:]): fore(target)
        fore_target.__name__ = "target_fore"
        back_target = lambda i, target=np.array(label[:]): back(target)
        back_target.__name__ = "target_back"
        leftSide_target = lambda i, target=np.array(label[:]): leftSide(target)
        leftSide_target.__name__ = "target_leftSide"
        rightSide_target = lambda i, target=np.array(label[:]): rightSide(target)
        rightSide_target.__name__ = "target_rightSide"
        test_target = lambda i, target=np.array(label[:]): step4(i, target, -1)
        test_target.__name__ = "target_test"
        frangi_target = lambda i, target=np.array(label[:]): step4_frangi(i, target, -1)
        frangi_target.__name__ = "target_frangi"
        sato_target = lambda i, target=np.array(label[:]): step4_sato(i, target, -1)
        sato_target.__name__ = "target_sato"
        hessian_target = lambda i, target=np.array(label[:]): step4_hessian(i, target, -1)
        hessian_target.__name__ = "target_hessian"
        
        middle_dilated = lambda i, target=np.array(label[:]): middle(dilate_target(i)[0])
        middle_dilated.__name__ = "middle_dilated"
        middle_eroded = lambda i, target=np.array(label[:]): middle(erode_target(i)[0])
        middle_eroded.__name__ = "middle_eroded"
        middle_left = lambda i, target=np.array(label[:]): middle(left(target)[0])
        middle_left.__name__ = "middle_left"
        middle_right = lambda i, target=np.array(label[:]): middle(right(target)[0])
        middle_right.__name__ = "middle_right"
        middle_up = lambda i, target=np.array(label[:]): middle(up(target)[0])
        middle_up.__name__ = "middle_up"
        middle_down = lambda i, target=np.array(label[:]): down(middle(target)[0])
        middle_down.__name__ = "middle_down"
        middle_fore = lambda i, target=np.array(label[:]): fore(middle(target)[0])
        middle_fore.__name__ = "middle_fore"
        middle_back = lambda i, target=np.array(label[:]): middle(back(target)[0])
        middle_back.__name__ = "middle_back"
        middle_leftSide = lambda i, target=np.array(label[:]): middle(leftSide(target)[0])
        middle_leftSide.__name__ = "middle_leftSide"
        middle_rightSide = lambda i, target=np.array(label[:]): middle(rightSide(target)[0])
        middle_rightSide.__name__ = "middle_rightSide"

        #algos.append(dilate_target)
        #algos.append(erode_target)

        for algo in algos + [middle_dilated, middle_eroded, middle_left, middle_right, middle_up, middle_down, middle_leftSide, middle_rightSide, middle_fore, middle_back,middle_target, middlep_target, middlem_target,upper_target,lower_target, fore_target, back_target, leftSide_target, rightSide_target, erode_target, dilate_target,left_target,right_target,up_target,down_target]:
            proctime = time.process_time()
            try:
                logger.info(name + " start:  " + algo.__name__ + " " + str(image.shape))
                mask_image, timestep = algo(np.array(image[:]))
                logger.info(name + " finish: " + algo.__name__ + " {}".format(mask_image.dtype))
                for i in range(mask_image.shape[0]):
                    if np.count_nonzero(mask_image[i]) > 0:
                        break
                #mask_image = np.array([np.sum(mask_image[i:i+3], axis=0)], dtype=mask_image.dtype)
            except Exception as e:
                traceback.print_exc()
            proctime = time.process_time() - proctime
            if algo.__name__ in times:
                times[algo.__name__].append(proctime)
            else:
                times[algo.__name__] = [proctime]
            #mask_image = uncrop_image(mask_image, crop_box)
            OutImage = None
            if mask_image.dtype == np.bool:
                mask_image = np.array(mask_image, dtype=label.dtype)
            if mask_image.shape[0] == 1:
                OutImage = sitk.GetImageFromArray(mask_image[0])
                OutImage.SetDirection(dso[0][0:3] + dso[0][4:7] + dso[0][8:11])
                OutImage.SetSpacing(dso[1][:-1])
                OutImage.SetOrigin(dso[2][:-1])
            else:
                #mask_images = []
                #for j in range(mask_image.shape[0]):
                #    mask_images.append(sitk.GetImageFromArray(mask_image[j]))
                #OutImage = sitk.JoinSeries(mask_images)
                OutImage = sitk.GetImageFromArray(np.array(np.sum(mask_image, axis=0) > 0, dtype=mask_image.dtype))
                OutImage.SetDirection(dso[0][0:3] + dso[0][4:7] + dso[0][8:11])
                OutImage.SetSpacing(dso[1][:-1])
                OutImage.SetOrigin(dso[2][:-1])
                #OutImage.SetDirection(dso[0])
                #OutImage.SetSpacing(dso[1])
                #OutImage.SetOrigin(dso[2])
            
            MaskImage = None
            mask_image = np.array(mask_image, dtype=label.dtype)
            if mask_image.shape[0] == 1:
                MaskImage = sitk.GetImageFromArray(mask_image[0])
                MaskImage.SetDirection(dso[0][0:3] + dso[0][4:7] + dso[0][8:11])
                MaskImage.SetSpacing(dso[1][:-1])
                MaskImage.SetOrigin(dso[2][:-1])
            else:
                mask_images = []
                for j in range(mask_image.shape[0]):
                    mask_images.append(sitk.GetImageFromArray(mask_image[j]))
                MaskImage = sitk.JoinSeries(mask_images)
                MaskImage.SetDirection(dso[0])
                MaskImage.SetSpacing(dso[1])
                MaskImage.SetOrigin(dso[2])
            
            mask_images = []
            #t = np.argmax(np.count_nonzero(mask_image, axis=(1,2,3,4)))
            #mask_images.append(sitk.GetImageFromArray(np.array(mask_image[t][...,0], dtype=np.int8)))
            #mask_image = mask_image[t].reshape(label.shape)
            for j in range(mask_image.shape[0]):
                mask_images.append(sitk.GetImageFromArray(np.array(mask_image[j][...,0], dtype=np.int8)))

            #print(slice_label.shape, mask_image.shape, len(labels), len(mask_images))
            
            acc.append(label_evaluation.calc_metrics(image, slice_label, labels, mask_image, mask_images, name, algo.__name__, proctime))
            if acc[-1]['modelName'] != None and acc[-1]['dice'] != None and acc[-1]['aif_mse'] != None:
                logger.info("{} {:.5} {:.5}".format(acc[-1]['modelName'], acc[-1]['dice'],acc[-1]['aif_mse']))
            else:
                logger.info("{} {} {} {} {}".format(acc[-1]['modelName'], acc[-1]['dice'],acc[-1]['aif_mse']), np.count_nonzero(label), np.count_nonzero(mask_image))

            if write_files:
                sitk.WriteImage(OutImage, os.path.join(output_path, name + '.aif_mask.' + algo.__name__ + '.nrrd'))
                logger.info("written file: " + output_path + name + ".aif_mask." + algo.__name__ + '.nrrd')
    return acc, times


if __name__ == "__main__":
    #np.seterr(all='raise')
    #input_path = "../Data/normalized/"
    classes = 1
    #images, t2images, labels, names, dsos = loader.load_input(input_path, classes)
    #images, crop_box = crop_images(images)
    #images = smooth(images)
    #algos = [dante, max05total, shi2013, gamma_variate, chen2008, max05] # + [ncut, ncut_sklearn, FastAP, HIER]
    if False:
        algos = []
        
        #algos = []
        #for i in range(7):
        #for step_mask in np.array(np.meshgrid([True,False], [True,False], [True,False],[True,False], [True,False], [True,False], [True,False], [True,False], [True,False])).T.reshape(-1,9):
        from DisLAIF import step1, step2, step4_filter, step4, step6, step3, step9, step8, pipeline
        #steps = [step1, step2, step4_filter, step4, step6, step3, step9, step8, step9]
        steps = [step1, step2, step4_filter, step4, step6, step8, step9]
        for step in range(len(steps)):
            #step_mask = [1,] + list(step_mask) + [0,]
            current_steps = list(steps[:step+1])
            #fun = lambda im, step_mask=step_mask: dante(im, step_mask=step_mask)
            fun = lambda im, step_mask=current_steps: pipeline(im, step_mask)
            #fun = lambda im, i=i: dante(im, i=i)

            name = str(step)
            #for i,b in enumerate(current_steps[1:]):
            #    if b:
            #        name += str(i+1)
            #name += str(i)
            fun.__name__ = "dante." + current_steps[-1].__name__
            algos.append(fun)

        #algos = [shi2013]


        acc, times = find_aif("../Data/", "../Data/n_out_d/", classes, algos, write_files=True)
        #acc, times = find_aif("../Data/normalized_neu/", "../Data/n_out_d/", classes, algos, acc=acc, times=times, write_files=False)

        #acc, times = find_aif("D:/Image_Data/Patient_Data/Perfusion/normalized/", "D:/Image_Data/Patient_Data/Perfusion/n_out_d/", classes, algos, write_files=False)
        #acc, times = find_aif("D:/Image_Data/Patient_Data/Perfusion/normalized_neu/", "D:/Image_Data/Patient_Data/Perfusion/n_out_d/", classes, algos, acc=acc, times=times, write_files=False)


        eval_file_path = '../Data/aif_eval_d.csv'
    elif False:
        for image, slice_image, t2image, label, slice_label, name, dso in loader.load_input('../Data_d/', classes, label_selector=2):
            try:
                if label.shape[0] > 70:
                    label = label[:69]
                if image.shape[0] > 70:
                    image = image[:69]
                print(name)
                mask_image, timestep = DisLAIF(np.array(image[:]), save_step_path='../Data_d/n_out/' + name + '.', dso=dso)
            except Exception as e:
                traceback.print_exc()

            for i in range(mask_image.shape[0]):
                if np.count_nonzero(mask_image[i]) > 0:
                    break
            mask_image = np.array([np.sum(mask_image[i:i+3], axis=0)], dtype=mask_image.dtype)
            acc = []
            times = {}
            OutImage = None
            if mask_image.dtype == np.bool:
                mask_image = np.array(mask_image, dtype=label.dtype)
            if mask_image.shape[0] == 1:
                OutImage = sitk.GetImageFromArray(mask_image[0])
                OutImage.SetDirection(dso[0][0:3] + dso[0][4:7] + dso[0][8:11])
                OutImage.SetSpacing(dso[1][:-1])
                OutImage.SetOrigin(dso[2][:-1])
            else:
                mask_images = []
                for j in range(mask_image.shape[0]):
                    mask_images.append(sitk.GetImageFromArray(mask_image[j]))
                OutImage = sitk.JoinSeries(mask_images)
                OutImage.SetDirection(dso[0])
                OutImage.SetSpacing(dso[1])
                OutImage.SetOrigin(dso[2])
            
            sitk.WriteImage(OutImage, os.path.join('../Data_d/n_out/', name + '.aif_mask.DisLAIF.nrrd'))
            logger.info("written file: " + '../Data_d/n_out/' + name + ".aif_mask.DisLAIF.nrrd")
    else:
        algos = []
        #algos = [DisLAIF, shi2013]
        #algos = [chen2008]
        algos = [parker2003, DisLAIF, shi2013, chen2008, gamma_variate]#, OldDisLAIF] # + [ncut, ncut_sklearn, FastAP, HIER]
        #algos = [parker2003, DisLAIF, shi2013, chen2008, gamma_variate, max05, dante] # + [ncut, ncut_sklearn, FastAP, HIER]
        #algos = [dante, dante_lower, dante_upper, dante_middle, dante_middlep, dante_middlem]
        #algos = [shi2013, DisLAIF, DisLAIF_2, DisLAIF_3, DisLAIF_2_9, DisLAIF_3_8_9, DisLAIF_3_9, DisLAIF_gvt_pixel]
        #algos = [DisLAIF, DisLAIF4, DisLAIF_3_9, DisLAIF_frangi, DisLAIF_sato, DisLAIF_meijering]
        acc, times = find_aif("../Data/", "../Data/n_out/", classes, algos)
        chen2008un = lambda i: chen2008(i)
        chen2008un.__name__ = "chen2008un"
        chen2008_un = lambda i: chen2008_(i)
        chen2008_un.__name__ = "chen2008_un"
        parker2003un = lambda i: parker2003(i)
        parker2003un.__name__ = "parker2003un"
        DisLAIFun = lambda i: DisLAIF(i)
        DisLAIFun.__name__ = "DisLAIFun"
        shi2013un = lambda i: shi2013(i)
        shi2013un.__name__ = "shi2013un"
        gamma_variateun = lambda i: gamma_variate(i)
        gamma_variateun.__name__ = "gamma_variateun"
        #algos = [chen2008un, parker2003un, DisLAIFun, shi2013un, gamma_variateun, chen2008_un]
        #acc, times = find_aif("../Data/", "../Data/n_out/", classes, algos, acc, times, use_processed=False)
        #acc, times = find_aif("D:/Image_Data/Patient_Data/Perfusion/", "D:/Image_Data/Patient_Data/Perfusion/n_out/", classes, algos)
        #acc, times = find_aif("../Data/normalized_neu/", "../Data/n_out/", classes, algos, acc=acc, times=times)

        #eval_file_path = 'D:/Image_Data/Patient_Data/Perfusion/aif_select.csv'
        eval_file_path = '../Data/aif_select.csv'
    
    #names = ["target", "target_dilated", "target_eroded", "target_middle", "target_middle+1", "target_middle-1", "target_upper", "target_lower"] + [algo.__name__ for algo in algos]
    names = [a['modelName'] for a in acc]
    if len(names) > 0:
        header_row = label_evaluation.make_csv_file(eval_file_path)
        for metric in sorted(acc, key = lambda a: names.index(a['modelName'])):
            label_evaluation.write_metrics_to_csv(eval_file_path, header_row, metric)    

    for key in dante_timing:
        value = np.array(dante_timing[key])
        if value.shape[0] > 0:
            logger.info("{:<15} len: {:3} avg: {:7.2f} std: {:7.2f} mean: {:7.2f} max: {:7.2f}".format(key, len(value), np.average(value), np.std(value), np.mean(value), np.max(value)))
    
    for key in times:
        value = np.array(times[key])
        if value.shape[0] > 0:
            logger.info("{:<15} len: {:3} avg: {:7.2f} std: {:7.2f} mean: {:7.2f} max: {:7.2f}".format(key, len(value), np.average(value), np.std(value), np.mean(value), np.max(value)))

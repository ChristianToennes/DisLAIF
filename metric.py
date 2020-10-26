import numpy as np
import SimpleITK as sitk
import math

### Numpy

def dice_coefficient_np(output, target, threshold=0.5, smooth=1e-5):
    if output.shape[-1] > 1:
        output = output[..., 1] > threshold
        target = target[..., 1] > threshold
    inse = np.count_nonzero(np.logical_and(output, target))
    l = np.count_nonzero(output)
    r = np.count_nonzero(target)
    hard_dice = (2 * inse + smooth) / (l + r + smooth)
    return hard_dice


def signal_to_noise_ratio_np(output, object_mask, background_mask, eps=1e-5):
    dividend = np.mean(output[object_mask])
    divisor = np.std(output[background_mask])
    with np.errstate(divide='ignore'):
        return np.abs(dividend)/(divisor)


def contrast_to_noise_ratio_np(output, object_mask, background_mask, eps=1e-5):
    dividend = np.abs(np.mean(output[object_mask]) - np.mean(output[background_mask]))
    divisor = np.std(output[background_mask])
    with np.errstate(divide='ignore'):
        return np.abs(dividend)/(divisor)

def root_mean_squared_error_np(output, target):
    squared_error = (output.ravel() - target.ravel())**2
    mean_squared_error = np.mean(squared_error)
    return np.sqrt(mean_squared_error)

def mean_squared_error_np(output, target):
    squared_error = (output.ravel() - target.ravel())**2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error

def squared_error_sum_np(output, target):
    squared_error = (output.ravel() - target.ravel())**2
    sum_squared_error = np.sum(squared_error)
    return sum_squared_error

def mean_absolute_error_np(output, target):
    absolute_error = np.abs(output.ravel() - target.ravel())
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error


def mutual_information_np(output, target, bins=200):
    #I(X, Y) = H(X) + H(Y) - H(X,Y)
    # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    output = output.ravel()
    target = target.ravel()

    c_XY = np.histogram2d(output, target, bins)[0]
    c_X = np.histogram(output, bins)[0]
    c_Y = np.histogram(target, bins)[0]

    H_X = _shannon_entropy(c_X)
    H_Y = _shannon_entropy(c_Y)
    H_XY = _shannon_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def normalized_mutual_information_np(output, target, bins=200):
    # symmetric uncertainty
    mm = mutual_information_np(output, target, bins)
    c_X = np.histogram(output, bins)[0]
    c_Y = np.histogram(target, bins)[0]
    H_X = _shannon_entropy(c_X)
    H_Y = _shannon_entropy(c_Y)
    return (2 * mm) / (H_Y + H_X)


def normalized_cross_correlation_2d_np(output, target, filter_dim=5):
    np.seterr(divide='ignore', invalid='ignore')
    sf = np.ones((filter_dim, filter_dim))
    mf = np.divide(sf, (filter_dim**2))
    assert filter_dim % 2 != 0
    padding_size = math.ceil((filter_dim-1)/2)

    mx = _convolution2d_np(np.pad(output, ((padding_size, padding_size), (padding_size, padding_size)), 'edge'), mf)
    mx = output - mx
    my = _convolution2d_np(np.pad(target, ((padding_size, padding_size), (padding_size, padding_size)), 'edge'), mf)
    my = target - my
    cc_map = np.divide(_convolution2d_np(np.multiply(mx, my), sf), np.sqrt(np.multiply(_convolution2d_np(mx ** 2, sf), _convolution2d_np(my ** 2, sf))))
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(cc_map, interpolation='none', cmap='gray')
    #plt.show()
    cc = np.mean(cc_map)
    return cc

def normalized_cross_correlation_1d_np(output, target):
    output = np.array(output.ravel(), dtype=np.longlong)
    target = np.array(target.ravel(), dtype=np.longlong)

    cc = np.sum(output*target) / (max(1,np.sum(output))*max(1,np.sum(target)))
    return cc

def cross_correlation_1d_np(output, target):
    output = output.ravel()
    target = target.ravel()
    cc = np.correlate(output, target)
    return cc

### SITK
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html

def hausdorff_metric_sitk(output, target):
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    if type(output) == list:
        avg = 0
        count = 0
        for output in output:
            for i in range(len(output)):
                try:
                    hausdorff_distance_filter.Execute(target[0][i], output[i])
                    avg = avg + hausdorff_distance_filter.GetHausdorffDistance()
                    count = count + 1
                except Exception as e:
                    #print(e)
                    pass
        if count == 0:
            return np.inf
        return avg / count
    else:
        hausdorff_distance_filter.Execute(target, output)
        return hausdorff_distance_filter.GetHausdorffDistance()


def overlap_measures_sitk(output, target):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    if type(output) == list:
        dice_coefficient = 0
        volume_similarity = 0
        false_negative = 0
        false_positive = 0
        count = 0
        for output in output:
            try:
                overlap_measures_filter.Execute(target[0], output)
                dice_coefficient = dice_coefficient + overlap_measures_filter.GetDiceCoefficient()
                volume_similarity = volume_similarity + overlap_measures_filter.GetVolumeSimilarity()
                false_negative = false_negative + overlap_measures_filter.GetFalseNegativeError()
                false_positive =  false_positive + overlap_measures_filter.GetFalsePositiveError()
                count = count + 1
            except Exception as e:
                #print(e)
                pass
        if count == 0:
            return np.inf, np.inf, np.inf, np.inf
        return dice_coefficient/count, volume_similarity/count, false_negative/count, false_positive/count
    else:
        overlap_measures_filter.Execute(target, output)
        dice_coefficient = overlap_measures_filter.GetDiceCoefficient()
        volume_similarity = overlap_measures_filter.GetVolumeSimilarity()
        false_negative = overlap_measures_filter.GetFalseNegativeError()
        false_positive = overlap_measures_filter.GetFalsePositiveError()
        return dice_coefficient, volume_similarity, false_negative, false_positive

def symmetric_surface_measures_sitk(lloutput, lltarget):
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    try:
        all_surface_distances = []
        for loutput, ltarget in zip(lloutput, lltarget):
            for output, target in zip(loutput, ltarget):
                if np.count_nonzero(sitk.GetArrayFromImage(target)) == 0:
                    continue
                segmented_surface = sitk.LabelContour(output)
                reference_surface = sitk.LabelContour(target)
                reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
                segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(output, squaredDistance=False))
                seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
                ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)
                statistics_image_filter = sitk.StatisticsImageFilter()
                statistics_image_filter.Execute(reference_surface)
                num_reference_surface_pixels = int(statistics_image_filter.GetSum())
                statistics_image_filter.Execute(segmented_surface)
                num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
                seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
                seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
                seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
                ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
                ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
                ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

                all_surface_distances += seg2ref_distances + ref2seg_distances
        
        if len(all_surface_distances) > 0:
            mean_symmetric_surface_distance = np.mean(all_surface_distances)
            median_symmetric_surface_distance = np.median(all_surface_distances)
            std_symmetric_surface_distance = np.std(all_surface_distances)
            max_symmetric_surface_distance = np.max(all_surface_distances)
            return mean_symmetric_surface_distance, median_symmetric_surface_distance, std_symmetric_surface_distance, max_symmetric_surface_distance
        else:
            return np.nan, np.nan, np.nan, np.nan
    except Exception as e:
        print(e)
        raise
        return np.nan, np.nan, np.nan, np.nan

def surface_measures_sitk(output, target):
    label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
    if type(output) == list:
        mean_surface_distance = 0
        median_surface_distance = 0
        std_surface_distance = 0
        max_surface_distance = 0
        count = 0
        for output in output:
            for i in range(len(output)):
                try:
                    segmented_surface = sitk.LabelContour(output[i])
                    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target[0][i], squaredDistance=False))
                    label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)
                    label = 1
                    mean_surface_distance = mean_surface_distance + label_intensity_statistics_filter.GetMean(label)
                    median_surface_distance = median_surface_distance + label_intensity_statistics_filter.GetMedian(label)
                    std_surface_distance = std_surface_distance + label_intensity_statistics_filter.GetStandardDeviation(label)
                    max_surface_distance = max_surface_distance + label_intensity_statistics_filter.GetMaximum(label)        
                    count = count + 1
                except Exception as e:
                    #print(e)
                    pass
        if count == 0:
            return np.inf, np.inf, np.inf, np.inf
        return mean_surface_distance/count, median_surface_distance/count, std_surface_distance/count, max_surface_distance/count
    else:
        segmented_surface = sitk.LabelContour(output)
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
        label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)
        label = 1
        mean_surface_distance = label_intensity_statistics_filter.GetMean(label)
        median_surface_distance = label_intensity_statistics_filter.GetMedian(label)
        std_surface_distance = label_intensity_statistics_filter.GetStandardDeviation(label)
        max_surface_distance = label_intensity_statistics_filter.GetMaximum(label)
        return mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance


### Helper Functions

def _convolution2d_np(image, kernel, bias=1e-5):
    k_s = kernel.shape
    if k_s[0] == k_s[1]:
        i_s = image.shape
        y = i_s[0] - (k_s[0]-1)
        x = i_s[1] - (k_s[0]-1)
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+k_s[0], j:j+k_s[0]]*kernel) + bias
    return new_image


def _shannon_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H
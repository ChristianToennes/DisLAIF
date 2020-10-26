import dice_coef
import metric
import csv
import numpy as np
from scipy import ndimage

metrics = [
'modelName',
'modelIndex',
'fileName',
'time',
'acc',  
'recall',
'precision',
'dice',
#'mse', 
#'normalized_mutual_information', 
#'cross_correlation', 
#'normalized_cross_correlation',
#'delta_t_peak',
#'delta_peak',
#'delta_auc',
#'ratio_t_peak',
#'ratio_peak',
#'ratio_auc',
#'aif_mse',
#'aif_sse',
#'aif_normalized_cross_correlation',
#'aif_normalized_mutual_information',
#'hausdorff', 
#['surface', 'mean_surface_distance', 'median_surface_distance', 'std_surface_distance', 'max_surface_distance'],
#['overlap', 'overlap_dice', 'overlap_volume_similarity', 'overlap_false_negative', 'overlap_false_positive'],
#'peak_t',
#'auc',
#'peak',
]

def calc_aif(image, output, fileName, modelName, modelIndex):
    aif = image*output / np.max(image)
    #print(image.shape, output.shape, aif.shape)
    if len(aif.shape) == 5:
        num = np.sum(aif, axis=(1,2,3,4))
    else:
        num = np.sum(aif, axis=(1,2,3))
    if len(output.shape) == 5:
        denom = np.count_nonzero(output, axis=(1,2,3,4))
    else:
        denom = np.count_nonzero(output, axis=(1,2,3))
    if denom.shape[0] == 1:
        if denom[0] == 0:
            num = np.zeros_like(num)
            denom = 1
    else:
        num[denom==0] = 0
        denom[denom==0] = 1

    custom_layers = {
            'fileName': fileName,
            'modelName': modelName,
            'modelIndex': modelIndex,
            'aif': num / denom,
    }
    return custom_layers
    
def calc_metrics(image, target_image, Target, output_image, Output, fileName, modelName, modelIndex):
    if target_image.shape[-1] == 2:
        target_mask = target_image[...,1:]>0
    else:
        target_mask = target_image>0
    if output_image.shape[-1] == 2:
        output_mask = output_image[...,1:]>0
    else:
        output_mask = output_image>0
    if np.count_nonzero(target_mask) == 0:
            custom_layers = {
            'fileName': fileName,
            'modelName': modelName,
            'modelIndex': modelIndex,
            'time': None,
            'acc': None,
            'recall': None,
            'precision': None,
            'dice': None,
            #'mse': None,
            #'normalized_mutual_information': None,
            #'cross_correlation': None,
            #'normalized_cross_correlation': None,
            #'delta_t_peak': None,
            #'delta_peak': None,
            #'delta_auc': None,
            #'ratio_t_peak': None,
            #'ratio_peak': None,
            #'ratio_auc': None,
            #'aif_mse': None,
            #'aif_sse': None,
            #'aif_normalized_cross_correlation': None,
            #'aif_normalized_mutual_information': None,
            #'hausdorff': None,
            #'surface': None,
            #'overlap': None,
            #'peak_t': None,
            #'auc': None,
            #'peak': None,
        }
    else:
        #output = output[np.argmax(np.count_nonzero(output, axis=(1,2,3,4)))]
        #output = output.reshape((1,) + output.shape)
        
        #if(len(image.shape) == 5):
        #    num_true = np.sum(image*target_mask, axis=(1,2,3,4)) / np.max(image*target_mask)
        #    denom_true = np.count_nonzero(target_mask, axis=(1,2,3,4))
        #    num = np.sum(image*output_mask, axis=(1,2,3,4)) / np.max(image*target_mask)
        #    denom = np.count_nonzero(output_mask, axis=(1,2,3,4))
        #else:
        #    num_true = np.sum(image*target_mask, axis=(1,2,3)) / np.max(image*target_mask)
        #    denom_true = np.count_nonzero(target_mask, axis=(1,2,3))
        #    num = np.sum(image*output_mask, axis=(1,2,3)) / np.max(image*target_mask)
        #    denom = np.count_nonzero(output_mask, axis=(1,2,3))
        #if denom_true.shape[0] == 1:
        #    if denom_true[0] == 0:
        #        num_true = np.zeros_like(num_true)
        #        denom_true = 1
        #else:
        #    num_true[denom_true==0] = 0
        #    denom_true[denom_true==0] = 1
        #c_true = num_true / denom_true
        
        
        #if denom.shape[0] == 1:
        #    if denom[0] == 0:
        #        num = np.zeros_like(num)
        #        denom = 1
        #else:
        #    num[denom==0] = 0
        #    denom[denom==0] = 1
        #c = num / denom
        
        custom_layers = {
            'fileName': fileName,
            'modelName': modelName,
            'modelIndex': modelIndex,
            'time': 0,
            'acc': np.max([dice_coef.acc(target_mask, output) for output in output_mask]),
            'recall': np.max([dice_coef.recall(target_mask, output) for output in output_mask]),
            'precision': np.max([dice_coef.precision(target_mask, output) for output in output_mask]),
            'dice': np.max([metric.dice_coefficient_np(target_mask, output) for output in output_mask]),
            #'mse': metric.mean_squared_error_np(np.array([output]), target),
            #'normalized_mutual_information': metric.normalized_mutual_information_np(np.array([output]), target),
            #'cross_correlation': metric.cross_correlation_1d_np(np.array([output]), target)[0],
            #'normalized_cross_correlation': metric.normalized_cross_correlation_1d_np(np.array([output]), target),
            #'delta_t_peak': abs(np.argmax(c)-np.argmax(c_true)),
            #'delta_peak': abs(np.max(c)-np.max(c_true)),
            #'delta_auc': abs(np.sum(c)-np.sum(c_true)),
            #'ratio_t_peak': (np.argmax(c)-np.argmax(c_true)),
            #'ratio_peak': 100*(np.max(c)-np.max(c_true))/np.max(c_true),
            #'ratio_auc': np.sum(c-c_true),
            #'aif_mse': metric.mean_squared_error_np(c, c_true),
            #'aif_sse': metric.squared_error_sum_np(c, c_true),
            #'aif_normalized_cross_correlation': metric.normalized_cross_correlation_1d_np(c, c_true),
            #'aif_normalized_mutual_information': metric.normalized_mutual_information_np(c, c_true),
            #'hausdorff': metric.hausdorff_metric_sitk(Output, Target),
            #'surface': metric.symmetric_surface_measures_sitk([Output], [Target]),
            #'overlap': metric.overlap_measures_sitk([Output], [Target]),
            #'peak_t': np.argmax(c),
            #'auc': np.sum(c),
            #'peak': np.max(c),
        }
    return custom_layers


def make_csv_file(eval_file_path):
    with open(eval_file_path, 'w', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = make_csv_header()
        
        expanded_row = []
        for field in header_row:
            if type(field) == list:
                expanded_row = expanded_row + field[1:]
            else:
                expanded_row.append(field)

        eval_csv_writer.writerow(expanded_row)
        return header_row

def make_csv_header():
    header_row = metrics
    return header_row

def write_metrics_to_csv(eval_file_path, header_row, result_metrics):
    with open(eval_file_path, 'a', newline='') as evaluation_file:
        eval_csv_writer = csv.writer(evaluation_file, delimiter=',', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
        eval_csv_writer.writerow(make_csv_row(header_row, result_metrics))

def read_metrics_csv(eval_file_path):
    all_metrics = []
    with open(eval_file_path, 'r') as evaluation_file:
        eval_csv_reader = csv.DictReader(evaluation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in eval_csv_reader:
            metric = {}
            for k in metrics:
                if type(k) == type([]):
                    continue
                metric[k] = row[k]
            all_metrics.append(metric)
    return all_metrics

def write_aif_to_csv(aif_file_path, aifs):
    if len(aifs) == 0:
        return
    with open(aif_file_path, "w", newline='') as csv_file:
        header_row = []
        for aif in aifs:
            header_row.append(aif["modelName"] + " " + aif["fileName"])
        csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(header_row), dialect='excel')
        csv_writer.writeheader()
        
        for i in range(len(aifs[0]["aif"])):
            row = {}
            for j,k in enumerate(header_row):
                row[k] = aifs[j]["aif"][i]
            csv_writer.writerow(row)
        

def read_aif_csv(aif_file_path):
    aifs = []
    with open(aif_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        first = True
        for row in csv_reader:
            if first:
                first = False
                for k in row:
                    model, fileName = k.split(' ')
                    aifs.append({"modelName": model, "fileName": fileName, "aif": []})
            else:
                for i, k in enumerate(row):
                    try:
                        aifs[i]["aif"].append(float(k))
                    except Exception:
                        print(k)
    return aifs

def make_csv_row(header_row, result_metrics):
    row = []
    for field in header_row:
        if type(field) == list:
            for i in range(len(field)-1):
                if type(result_metrics[field[0]]) == list:
                    row.append(result_metrics[field[0]][i])
                else:
                    row.append(np.inf)
        else:
            row.append(result_metrics[field])
    return row

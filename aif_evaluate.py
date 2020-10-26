import numpy as np
import SimpleITK as sitk
import label_evaluation
import os

import matplotlib.pyplot as plt

import logging

logger = logging.getLogger('perfusion.scope')
logger.setLevel(logging.INFO)

file_log_handler = logging.FileHandler('logfile.log')
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler()
logger.addHandler(stdout_log_handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stdout_log_handler.setFormatter(formatter)

def evaluate_aif(path, accs=[], aifs=[]):
    files = {}
    for entry in os.scandir(path):
        parts = entry.name.split('.')
        if entry.is_file() and parts[-1] == "nrrd":
            if parts[0] not in files:
                files[parts[0]] = {}
            files[parts[0]][parts[2]] = entry.path
    
    for key in files:
        if "target" not in files[key]:
            continue

        print(key)

        Target = sitk.ReadImage(files[key]["target"])
        target = np.array(sitk.GetArrayFromImage(Target), dtype=int)
        target = target.reshape((1,) + target.shape + (1,))
        Image = sitk.ReadImage(files[key]["noop"])
        image = sitk.GetArrayFromImage(Image)
        image = image.reshape(image.shape + (1,))
        #aifs.append(label_evaluation.calc_aif(image, target, key, "target", 0))
        accs = []
        aifs = []

        null = np.zeros_like(target)
        labels = []
        targets = []
        labels.append(sitk.GetImageFromArray(null[0][...,0]))
        targets.append(sitk.GetImageFromArray(target[0][...,0]))
        
        accs.append(label_evaluation.calc_metrics(image, target, [targets], null, [labels], key, "null", 0))
        aifs.append(label_evaluation.calc_aif(image, null, key, "null", 0))

        for algo in files[key]:
            if algo in ("noop", "max05total"):
                continue
            Label = sitk.ReadImage(files[key][algo])
            label = np.array(sitk.GetArrayFromImage(Label), dtype=int)
            label = label.reshape(label.shape + (1,))
            if len(label.shape) == 4:
                label = label.reshape((1,) + label.shape)
            
            t = np.argmax(np.count_nonzero(label, axis=(1,2,3,4)))
            if label.size != target.size:
                print(key, algo, label.shape, target.shape)
                continue
            label = label[t].reshape(target.shape)
            
            labels = []
            targets = []
            labels.append(sitk.GetImageFromArray(label[0][...,0]))
            targets.append(sitk.GetImageFromArray(target[0][...,0]))

            #print(image.shape, label.shape, label.dtype, target.shape, target.dtype)
            #print(labels[0].GetPixelIDValue(), labels[0].GetSize(), targets[0].GetPixelIDValue(), targets[0].GetSize())
            
            accs.append(label_evaluation.calc_metrics(image, target, [targets], label, [labels], key, algo, 0))
            aifs.append(label_evaluation.calc_aif(image, label, key, algo, 0))
        #print(key)
        yield accs, aifs

        if False:
            accs = []
            aifs = []

            target = np.array(sitk.GetArrayFromImage(Target), dtype=int)
            target = target.reshape((1,) + target.shape + (1,))
            target = target[:,:,:,:target.shape[3]//2,:]
            
            image = sitk.GetArrayFromImage(Image)
            image = image.reshape(image.shape + (1,))
            image = image[:,:,:,:image.shape[3]//2,:]

            null = np.zeros_like(target)
            labels = []
            targets = []
            labels.append(sitk.GetImageFromArray(null[0][...,0]))
            targets.append(sitk.GetImageFromArray(target[0][...,0]))
            
            accs.append(label_evaluation.calc_metrics(image, target, [targets], null, [labels], key, "null Left", 0))
            aifs.append(label_evaluation.calc_aif(image, null, key, "null Left", 0))

            for algo in files[key]:
                if algo in ("noop", "max05total"):
                    continue
                Label = sitk.ReadImage(files[key][algo])
                label = np.array(sitk.GetArrayFromImage(Label), dtype=int)
                label = label.reshape(label.shape + (1,))
                if len(label.shape) == 4:
                    label = label.reshape((1,) + label.shape)

                label = label[:,:,:,:label.shape[3]//2,:]
                
                t = np.argmax(np.count_nonzero(label, axis=(1,2,3,4)))
                label = label[t].reshape(target.shape)
                
                labels = []
                targets = []
                labels.append(sitk.GetImageFromArray(label[0][...,0]))
                targets.append(sitk.GetImageFromArray(target[0][...,0]))
                
                accs.append(label_evaluation.calc_metrics(image, target, [targets], label, [labels], key, algo + " Left", 0))
                aifs.append(label_evaluation.calc_aif(image, label, key, algo + " Left", 0))
            #print(key)
            yield accs, aifs

            accs = []
            aifs = []

            target = np.array(sitk.GetArrayFromImage(Target), dtype=int)
            target = target.reshape((1,) + target.shape + (1,))
            target = target[:,:,:,target.shape[3]//2:,:]
            
            image = sitk.GetArrayFromImage(Image)
            image = image.reshape(image.shape + (1,))
            image = image[:,:,:,image.shape[3]//2:,:]

            null = np.zeros_like(target)
            labels = []
            targets = []
            labels.append(sitk.GetImageFromArray(null[0][...,0]))
            targets.append(sitk.GetImageFromArray(target[0][...,0]))
            
            accs.append(label_evaluation.calc_metrics(image, target, [targets], null, [labels], key, "null Right", 0))
            aifs.append(label_evaluation.calc_aif(image, null, key, "null Right", 0))

            for algo in files[key]:
                if algo in ("noop", "max05total"):
                    continue
                Label = sitk.ReadImage(files[key][algo])
                label = np.array(sitk.GetArrayFromImage(Label), dtype=int)
                label = label.reshape(label.shape + (1,))
                if len(label.shape) == 4:
                    label = label.reshape((1,) + label.shape)

                label = label[:,:,:,label.shape[3]//2:,:]
                
                t = np.argmax(np.count_nonzero(label, axis=(1,2,3,4)))
                label = label[t].reshape(target.shape)
                
                labels = []
                targets = []
                labels.append(sitk.GetImageFromArray(label[0][...,0]))
                targets.append(sitk.GetImageFromArray(target[0][...,0]))
                
                accs.append(label_evaluation.calc_metrics(image, target, [targets], label, [labels], key, algo + " Right", 0))
                aifs.append(label_evaluation.calc_aif(image, label, key, algo + " Right", 0))
            #print(key)
            yield accs, aifs
        #return accs, aifs
                

if __name__ == "__main__":
    
    #accs, aifs = evaluate_aif("../Data/n_out/")
    #accs = evaluate_aif("../Data/normalized_neu/", accs=accs)

    eval_file_path = '../Data/aif_eval.csv'
    aif_file_path = '../Data/aif.csv'
    #header_row = label_evaluation.make_csv_file(eval_file_path)
    #for metric in sorted(accs, key = lambda a: a['modelName']):
    #    label_evaluation.write_metrics_to_csv(eval_file_path, header_row, metric)    

    #label_evaluation.write_aif_to_csv(aif_file_path, aifs)
    all_aifs = []
    all_accs = []
    for accs, aifs in evaluate_aif('../Data/n_out/'):
        all_aifs += aifs
        all_accs += accs
        #fileAifs = {}
        #for aif in aifs:
        #    if aif["fileName"] not in fileAifs:
        #        fileAifs[aif["fileName"]] = {}
        #    fileAifs[aif["fileName"]][aif["modelName"]] = aif["aif"]
        
        #for k in fileAifs.keys():
        #    plt.figure(figsize=(30,15))
        #    x = range(0, len(fileAifs[k]["target"]))
        #    for algorithm in sorted(fileAifs[k].keys()):
        #        if algorithm in ("max05", "max05total", "noop", "chen2008", "shi2013", "gamma_variate"):
        #            continue
        #        plt.plot(x, fileAifs[k][algorithm], label=algorithm)

        #    plt.legend(loc="upper right", fontsize='large')

        #    plt.savefig(os.path.join('../Data/aifs/', k + ".png"))
        #    plt.close()
        label_evaluation.write_aif_to_csv(aif_file_path, all_aifs)

        header_row = label_evaluation.make_csv_file(eval_file_path)
        for metric in sorted(all_accs, key = lambda a: a['modelName']):
            label_evaluation.write_metrics_to_csv(eval_file_path, header_row, metric) 
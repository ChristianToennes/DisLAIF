import numpy as np
import csv
import seaborn as sns
#from matplotlib.pyplot import savefig, violinplot, boxplot, figure, subplot, close, plot, legend, tight_layout, scatter, yscale, xscale, text
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import warnings
import pydicom
import scipy.stats
import loader

x_cache = {}

def get_t(path, name):
    if name in x_cache:
        return x_cache[name]
    path = os.path.join(os.path.normpath(path), name)
    dceFiles = []
    for dirpath, _, filenames in os.walk(path):
        for fileName in filenames:
            if fileName.endswith('.dcm'):
                if dirpath.split('\\')[-1].startswith("t2"):
                    pass
                else:
                    dceFiles.append(os.path.join(dirpath, fileName))
    time = []
    currentTimeSeries = -1
    for fileName in dceFiles:
        ds = pydicom.read_file(fileName)
        if(currentTimeSeries != ds[0x20,0x12].value):
            currentTimeSeries = ds[0x20,0x12].value
            h = float(ds[0x08,0x32].value[0:2])
            m = float(ds[0x08,0x32].value[2:4])
            s = float(ds[0x08,0x32].value[4:6])
            time.append(h*60*60 + m*60 + s)
    
    t0 = False
    times = []
    for t in sorted(time):
        if not t0:
            t0 = t
        times.append((t-t0) / 60)

    x_cache[name] = np.array(times)
    return np.array(times)

def readCSV(file_path, fieldnames = None):
    data = {}
    if not os.path.exists(file_path):
        return data
    with open(file_path, newline='', encoding='utf8') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',')
        skip = False
        for row in csv_reader:
            if skip:
                skip = False
                continue
            
            if len(data) > 0 and any([k not in row.keys() for k in data]): continue
            for k in row.keys():
                if k==None or (fieldnames and k not in fieldnames):
                    continue
                if k not in data:
                    data[k] = []
                if k == "modelName" or k == "fileName":
                    data[k].append(row[k])
                else:
                    try:
                        if row[k] != None and row[k] != "" and row[k][0] != '[':
                            value = float(row[k])
                            if value > 1e8:
                                value = 1e8
                            data[k].append(value)
                        else:
                            pass
                            #data[k].append(np.nan)
                    except Exception as e:
                        print(k, row[k], e, file_path)
                        raise
    
    for k in data.keys():
        data[k] = np.array(data[k])

    return data

def transform_curves(data):
    output = {}
    for k in data.keys():
        algorithm, filename = k.rsplit(' ', maxsplit=1)
        if filename not in output:
                output[filename] = {}
        if "-" in algorithm and "-1" not in algorithm:
            netname, fold, loss_function, params = algorithm.split('-')
            if loss_function in y_labels:
                loss_function = y_labels[loss_function]
            else:
                #print(loss_function)
                pass
            if netname in y_labels:
                netname = y_labels[netname]
            else:
                #print(netname)
                pass
            algorithm = netname + "-" + fold + "-" + loss_function + "-" + params
        else:
            if algorithm in y_labels:
                algorithm = y_labels[algorithm]
            #else:
            #    print(algorithm)
        output[filename][algorithm] = np.array(data[k])
        output[filename]["x"] = get_t(os.path.join(dataDir, "input_"), filename)[:-1]
    return output

def get_datafilter(data, key, ignore, key2):
    output = []
    names = []
    if key not in data:
        return np.array(output), np.array(names)
    for k in sorted(list(set(data[key]))):
        if k in ignore or (type(k) == str and k.rsplit(" ",1)[-1] in ("Left", "Right") and k.rsplit(" ",1)[0] in ignore): 
            continue
        datafilter = data[key]==k
        if key2:
            for k2 in sorted(list(set(data[key2][datafilter]))):
                if k2 in ignore or (type(k2) == str and k2.rsplit(" ", 1)[-1] in ("Left", "Right") and k2.rsplit(" ", 1)[0] in ignore):
                    continue
                datafilter2 = np.bitwise_and(data[key2]==k2, datafilter)
                output.append(datafilter2)
                if k in y_labels:
                    names.append(y_labels[k] + " " + str(k2))
                else:
                    names.append(str(k) + " " + str(k2))
        else:
            output.append(datafilter)
            if k in y_labels:
                names.append(y_labels[k])
            else:
                if '.' in k:
                    a, n = k.split('.')
                    if a in y_labels:
                        names.append(y_labels[a] + "." + n)
                    else:
                        names.append(a)
                else:
                    names.append(str(k))
    return np.array(output), np.array(names)

x_labels = { "time": "t [s]", "acc": "accuracy", "label_acc": "accuracy", "dice": "Dice-Sørensen-Coefficient", "delta_t_peak": "Δt/min", "delta_peak": "ΔS(t)", 
             "delta_auc": "", "ratio_t_peak": "", "ratio_peak": "%", "ratio_auc": "", "aif_mse": "Mean Square Error Sum", "overlap_dice": "Dice Coefficient", 
             "overlap_false_negative": "False Negative", "overlap_false_positive": "False Positive", "mean_surface_distance": "Mean Surface Distance", 
             "median_surface_distance": "Median Surface Distance", "recall": "Recall", "precision": "Precision",
             "Fp": "Fp [ml/min/100ml]", "vp": "vp [ml/100ml]", "PS": "PS [ml/min/100ml]", "ve": "ve [ml/100ml]", "MTT": "MTT [min]", 
             "Ktrans": "Ktrans [1/min]", "E": "extraction fraction [%]", "χ": "",
             "%Fp": "% Fp", "%vp": "%vp", "%PS": "% PS", "%ve": "% ve", "%MTT": "% MTT", 
             "%Ktrans": "% Ktrans", "%E": "% extraction fraction", "%χ": "" , "aif_sse": "Squared Error Sum", "aif_normalized_cross_correlation": "Normalized Cross Correlation",
             "aif_normalized_mutual_information": "Normalized Mutual Information", "std_surface_distance": "Std Surface Distance", "max_surface_distance": "Maximal Surface Distance",
             "overlap_volume_similarity": "Overlap Volume Similarity"}

y_labels = {"time": "Time", "acc": "Accuracy", "label_acc": "Label Accuracy", "dice": "Dice", "delta_t_peak": "Δt_peak", "delta_peak": "ΔS(t_peak)", 
"delta_auc": "Δ∫S(t)", "ratio_t_peak": "t_peak/t_peak_target", "ratio_peak": "S(t_peak)/(t_peak_target)", "ratio_auc": "∫S(t)dt / ∫S_target(t)dt", 
"aif_mse": "Mean Square Error", "overlap_dice": "Dice", "overlap_false_negative": "False Negative", "overlap_false_positive": "False Positive", 
"Fp": "Fp", "vp": "vp", "PS": "PS", "ve": "ve", "MTT": "MTT", "Ktrans": "Ktrans", "E": "E", "χ": "χ",
"%Fp": "Fp", "%vp": "vp", "%PS": "PS", "%ve": "ve", "%MTT": "MTT", "%Ktrans": "Ktrans", "%E": "E", "%χ": "χ",
"mean_surface_distance": "Mean Surface Distance [mm]", "median_surface_distance": "Median Surface Distance [mm]", "recall": "Recall", "precision": "Precision",
"dante": "Dante", "DisLAIF": "Dis LAIF", "DisLAIF_opening": "DisLAIF Opening", 
"target": "Target", "target_eroded": "Eroded", "target_middle": "Middle Slice", "target_upper": "Upper Slice", "target_lower": "Lower Slice", 
"target_leftSide": "Left Artery", "target_rightSide": "Right Artery", "middle_back": "Middle Back Shift", "middle_fore": "Middle Fore Shift", 
"target_back": "Back Shift", "target_fore": "Fore Shift",
"middle_left": "Middle Left Shift", "middle_right": "Middle Right Shift", "middle_up": "Middle Up Shift", "middle_down": "Middle Down Shift", 
"middle_leftSide": "Middle Left Side", "middle_rightSide": "Middle Right Side", "middle_eroded": "Middle Eroded", "middle_dilated": "Middle Dilated",
"target_middlem": "Middle Slice -3", "target_middlep": "Middle Slice +3", "target_right": "Right Shift", "target_left": "Left Shift", "target_up": "Up Shift", "target_down": "Down Shift",
"target_dilated": "Dilated", "chen2008": "Chen 2008", "shi2013": "Shi 2013", "parker2003": "Parker 2003", "max05": "Max 0.5%", "gamma_variate": "GVF Fit", 
"max05total": "max05total", "noop": "No Operation", "DisLAIF_sato": "DisLAIF Sato", "DisLAIF_frangi": "DisLAIF_frangi", "DisLAIF_hession": "DisLAIF Hessian",

"PTU": "PTU", "CTU": "CTU", "AATH": "AATH", "2CXM": "2CXM", "DP": "DP", "Tofts": "Tofts",
"TNC": "Truncated Newton", "least_squares": "least squares", "trust_constr": "trust constr", 0.0: "0", 1.0: "1", 2.0: "2"}

colors = {"lines": "#ffffff", "time": "#ff0000", "acc": "#00ff00", "label_acc": "#00ff00", "dice": "#0000ff", "delta_t_peak": "#00ffff", "delta_peak": "#ff00ff", "delta_auc": "#ffff00", "ratio_t_peak": "#00ffff", "ratio_peak": "#ff00ff", "ratio_auc": "#ffff00", "aif_mse": "#0000ff", "overlap_dice": "#00ff00", "overlap_false_negative": "#ff0000", "overlap_false_positive": "#00ff00", "Fp": "#ff0000", "vp": "#00ffff", "PS": "#ff00ff", "ve": "#ffff00", "MTT": "#0000ff", "Ktrans": "#00ffff", "E": "#00ff00", "χ": "#aaaaaa",
"dante": "#006600", "dante_lower": "#009900", "dante_upper": "#00AA00", "dante_middle": "#00CC00", "dante_middlem": "#00EE00", "dante_middlep": "#00FF00",
"DisLAIF": "#00ff00", "DisLAIF_lower": "#00ff00", "DisLAIF_upper": "#00ff00", "DisLAIF_middle": "#00ff00", "DisLAIF_middlep": "#00ff00", "DisLAIF_middlem": "#00ff00",
 "DisLAIF_opening": "#000000", "DisLAIF_1": "#000000", "DisLAIF_2": "#000000", "DisLAIF_3": "#000000", "DisLAIF_4": "#000000", "DisLAIF_5": "#000000", "target": "#000000", "TumorTarget": "#000000",
"DisLAIF_4_6" : "#000000", "DisLAIF_4_6_8": "#000000", "DisLAIF_4_8": "#000000", "DisLAIF_4_3_8": "#000000", "DisLAIF_3_9": "#aaffaa", "DisLAIF_3_8_9": "#000000", "DisLAIF_gvt_pixel": "#000000", "DisLAIF_2_9": "#000000",
"DisLAIF_sato": "#00aa00", "DisLAIF_frangi": "#00aa00", "DisLAIF_hessian": "#00aa00", "target_sato": "#000000", "target_frangi": "#000000", "target_hessian": "#000000", 
"target_middle": "#FF0000", "target_middlep": "#FF00AA", "target_middlem": "#FFAA00", "target_upper": "#0000FF", "target_lower": "#000099", "target_eroded": "#FF0000", "target_dilated": "#990000", 
"target_right": "#000000", "target_left": "#000000", "target_up": "#000000", "target_down": "#000000", "target_fore":"#000000", "target_back":"#000000", "target_rightSide": "#000000", "target_leftSide": "#000000", "rightSide": "#000000", "leftSide": "#000000",
"chen2008": "#ff0000", "shi2013": "#0000ff", "max05": "#ff00ff", "parker2003": "#ff00ff", "gamma_variate": "#00ffff", "max05total": "#0000ff", 
"noop": "#aaaaaa", "null": "#aaaaaa", "OldDisLAIF": "#aaaaaa", "DisLAIF_size": "#aaaaaa", "DisLAIF_grow" : "#aaaaaa", "DisLAIF_dilated": "#aaaaaa", "DisLAIF_eroded": "#aaaaaa",
"target_test": "#000000", "DisLAIF_meijering": "#00aa00", "DisLAIF_frangi2": "#00aa00", "DisLAIF_sato2": "#00aa00", "DisLAIF4": "#00aa00",
"middle_left": "#FF0000", "middle_right": "#FF0000", "middle_up": "#FF0000", "middle_down": "#00FF00", "middle_fore": "#ff0000", "middle_back": "#ff0000",
"middle_leftSide": "#FF0000", "middle_rightSide": "#FF0000", "middle_eroded": "#FF5500", "middle_dilated": "#995500",

"PTU": "#ff0000", "CTU": "#ffff00", "AATH": "#ff00ff", "2CXM": "#00ff00", "DP": "#0000ff", "Tofts": "#00ffff",
"dice_loss": "#ff0000", "logits_loss": "#00ff00", "tversky_loss": "#0000ff",
"4f1e": "#000000", "4f10000e": "#660000", "4f10e": "#006600", "4f100e": "#000066", "4f500e": "#660066", "4f1000e": "#666666", "4f3000e": "#666600", "4f5000e": "#006666",
"8f1e": "#000000", "8f10000e": "#990000", "8f10e": "#009900", "8f100e": "#000099", "8f500e": "#990099", "8f1000e": "#999999", "8f2000e": "#009900", "8f3000e": "#999900", "8f5000e": "#009999",
"16f1e": "#000000", "16f10000e": "#cc0000", "16f10e": "#00cc00", "16f100e": "#0000cc", "16f500e": "#cc00cc", "16f1000e": "#cccccc", "16f2000e": "#00cc00", "16f3000e": "#cccc00", "16f5000e": "#00cccc",
"32f1e": "#000000", "32f10000e": "#ff0000", "32f10e": "#00ff00", "32f100e": "#0000ff", "32f500e": "#ff00ff", "32f1000e": "#ffffff", "32f2000e": "#00ff00", "32f3000e": "#ffff00", "32f5000e": "#00ffff",
"TNC": "#ff0000", "least_squares": "#00ff00", "trust_constr": "#0000ff",
"UNet-": "#ff0000", "U": "#ff0000", "UHU": "#00ff00", "Deeper": "#00ff00", "UHUNet_deeper": "#009900", "MCUNet": "#ff0000", "UNet": "#ff0000", "VoxResNet": "#ff00ff", "ResNet": "#ff00ff", "MCVoxResNet": "#ff00ff", "SegNet": "#00ffff", "*SegNet": "#559999", "MCUNet3D": "#0000ff", "UNet3D": "#0000ff", "*3D-UNet": "#555599", 
"DeepVesselNet": "#0000ff", "V-Net": "#ffff00" }

#colors_dark = {"time": "#aa0000", "acc": "#00aa00", "label_acc": "#00aa00", "dice": "#0000aa", "delta_t_peak": "#00aaaa", "delta_peak": "#aa00aa", "delta_auc": "#aaaa00", "ratio_t_peak": "#00aaaa", "ratio_peak": "#aa00aa", "ratio_auc": "#aaaa00", "aif_mse": "#0000aa", "overlap_dice": "#00aa00", "overlap_false_negative": "#aa0000", "overlap_false_positive": "#00aa00", "Fp": "#aa0000", "vp": "#00aaaa", "PS": "#aa00aa", "ve": "#aaaa00", "MTT": "#0000aa", "Ktrans": "#00aaaa", "E": "#00aa00", "χ": "#000000",
#"dante": "#00aa00", "target": "#0000aa", "target_eroded": "#aa0000", "target_dilated": "#aaaa00", "chen2008": "#00aaaa", "shi2013": "#aa00aa", "max05": "#aa0000", "gamma_variate": "#aaaa00", "max05total": "#0000aa", "noop": "#aaaaaa"}

titles = {"dice": "Dice", "tversky": "Tversky", "logits": "Cross Entropy", "crossentropy": "Cross Entropy", "tanimoto": "Tanimoto",
"dice_loss": "", "tversky_loss": "Tversky Loss", "logits_loss": "Cross Entropy", "crossentropy_loss": "Cross Entropy Loss",
 "UNet-": "UNet", "UHU": "UHUNet", "UHUNet_deeper": "Deeper UHUNet", "MCUNet": "Multi-Channel UNet", "tanimoto_loss": "Tanimoto Loss",
 "UNet": "UNet", "UHUNet": "UHUNet", "null": "Null", "MCSegNet": "Multi-Channel SegNet", "MCUNet3D": "Multi-Channel 3D-UNet",
"VoxResNet": "ResNet", "SegNet": "SegNet", "UNet3D": "3D-UNet", "MCVoxResNet": "Multi-Channel ResNet",
"4": "#Filter: 4", "8": "#Filter: 8", "16": "#Filter: 16", "32": "#Filter: 32", 
"TNC": "Truncated newton", "least_squares": "least squares", "trust_constr": "trust constr",
"2CXM": "2CXM", "DP": "DP", "PTU": "PTU", "CTU": "CTU", "AATH": "AATH", "Tofts": "Tofts",
"delta_auc": "|ΔAUC|", "delta_t_peak": "|Δt_peak|", "delta_peak": "|Δc(t_peak)|", "aif_mse": "Mean squared error", "ratio_peak": "Δc(t_peak) in %",
"ratio_t_peak": "Δt_peak", "recall": "Recall", "precision": "Precision",
"mean_surface_distance": "Mean Surface Distance", "median_surface_distance": "Median Surface Distance",
"DUHUNet2D": "Deeper UHUNet", "UHUNet2D": "UHUNet", "UNet2D": "U-Net", "FCN2D": "DeepVesselNet", "VNet2D": "V-Net", "ResNet2D": "ResNet",
"DUHUNet3D": "Deeper UHUNet 3D", "UHUNet3D": "UHUNet 3D", "UNet3D": "U-Net 3D", "FCN3D": "DeepVesselNet 3D", "VNet3D": "V-Net 3D", "ResNet3D": "ResNet 3D",
"0": "Trans", "1": "Cor", "2": "Sag", "both": "Both", "dce": "DCE", "t2": "T2"}

for k in colors.keys():
    colors[k] = colors[k].replace("#ffffff", "#000000")

for k in list(colors.keys()):
    if k in y_labels:
        colors[y_labels[k]] = colors[k]
        
for k in list(colors.keys()):
    if k in titles:
        colors[titles[k]] = colors[k]

for k in list(colors.keys()):
    if ' ' in k:
        colors[k.split(' ')[-1]] = colors[k]

colors_dark = {}
for k in list(colors.keys()):
    colors_dark[k] = colors[k].replace('ff', 'aa')


for p in test:
    x_labels[p] = p
    y_labels[p] = p
    titles[p] = p

for i, p in enumerate(pats):
    x_labels[p] = "Pat-" + str(i)
    y_labels[p] = "Pat-" + str(i)
    titles[p] = "Pat-" + str(i)

pats = pats + [titles[p] for p in pats]

def get_color(label):
    try:
        if label in colors:
            return colors[label]
        if " - " in label:
            if label.split(' - ')[0] in colors:
                return colors[label.split(" - ")[0]]
            else:
                return colors[label.split(" - ")[1]]
        if ":" in label:
            return colors[label.split(":")[0]]
        if '-' in label and '.' in label and ' ' in label:
            return colors[label.split('-')[2].split('.')[0].split(' ')[0]]
        if '-' in label:
            return get_color(label.split('-')[0])
        if '.' in label:
            return colors[label.split('.')[0]]
        if ' ' in label:
            if label.endswith(" Slice") or label.endswith(" Right") or label.endswith(" Left"):
                return colors[label.split(' ')[-2]]
            if label.endswith('T2') or label.endswith('DCE') or label.endswith('Both') or ' + ' in label:
                return colors[label.split(' ')[0]]
            return colors[label.split(' ')[-1]]
    except Exception as e:
        #print(e)
        pass
    return "k"

def get_dark_color(label):
    try:
        if label in colors_dark:
            return colors_dark[label]
        if " - "  in label:
            if label.split(' - ')[0] in colors_dark:
                return colors_dark[label.split(" - ")[0]]
            else:
                return colors_dark[label.split(" - ")[1]]
        if ":" in label:
            return colors_dark[label.split(":")[0]]
        if '-' in label and '.' in label and ' ' in label:
            return colors_dark[label.split('-')[2].split('.')[0].split(' ')[0]]
        if '-' in label:
            return get_dark_color(label.split('-')[0])
        if '.' in label:
            return colors_dark[label.split('.')[0]]
        if ' ' in label:
            if label.endswith(" Slice") or label.endswith(" Right") or label.endswith(" Left"):
                return colors_dark[label.split(' ')[-2]]
            if label.endswith('T2') or label.endswith('DCE') or label.endswith('Both'):
                return colors_dark[label.split(' ')[0]]
            return colors_dark[label.split(' ')[-1]]
    except Exception as e:
        #print(e)
        pass
    return "k"

matplotlib.pyplot.rcParams.update(
    {"ytick.color" : get_color("lines"),
    "xtick.color" : get_color("lines"),
    "grid.color": get_color("lines"),
    "text.color": get_color("lines"),
    "axes.labelcolor" : get_color("lines"),
    "axes.edgecolor" : get_color("lines"),
    "legend.edgecolor" : get_color("lines"),
    "legend.framealpha": 0}
)

def save_plot(data, data_type, title, y_label, file_path, stats = False, stretch_target=True, showmeans=True, showfliers=False, annotated_median=True, log_scale=False, show_box=True, significance_thresh=0):
    f = plt.figure(figsize=(30,1*len(data)+5))
    try:
        ax = plt.subplot(111)
        [i.set_linewidth(5) for i in ax.spines.values()]

        prefix = ""
        if data_type[0] == "Δ":
            prefix = "Δ"
            data_type = data_type[1:]
        

        if title in titles:
            ax.set_title(titles[title], pad=20)
        else:
            ax.set_title(title, pad=20)

        if log_scale:
            
            #index = np.argsort([np.quantile(data,0.95) for data in data])[-3:]
            if type(log_scale) == int:
                index = np.argsort([np.median(data) for data in data])[-log_scale:]
            else:
                index = np.argsort([np.median(data) for data in data])[-2:]
            
            imax = np.max([np.max(data) for data in data])
            #print(imax)
            sdata = []
            slabels = []
            for i,d in enumerate(data):
                if i in index:
                    continue
                sdata.append(d)
                slabels.append(y_label[i])
            if len(sdata) > 0:
                p = plt.boxplot(sdata, notch=False, whis=[0,100], showmeans=showmeans, showbox=True, meanline=True, showfliers=showfliers, vert=False, widths=0.9, labels=slabels)
            else:
                p = plt.boxplot(data, notch=False, whis=[0,100], showmeans=showmeans, showbox=True, meanline=True, showfliers=showfliers, vert=False, widths=0.9, labels=y_label)
            with plt.warnings.catch_warnings():
                #warnings.simplefilter("error")
                try:
                    plt.tight_layout()
                except Exception as e:
                    print(e, prefix, data_type, title, y_label, file_path)
            xlim = ax.get_xlim()
            plt.close()
            f = plt.figure(figsize=(30,1*len(data)+5))
            ax = plt.subplot(111)
            [i.set_linewidth(5) for i in ax.spines.values()]

        #prefix = ""
        #if data_type[0] == "Δ":
        #    prefix = "Δ"
        #    data_type = data_type[1:]

        if title in titles:
            ax.set_title(titles[title], pad=20)
        else:
            ax.set_title(title, pad=20)


        if show_box:
            p = plt.boxplot(data, notch=False, whis=[0,100], showmeans=showmeans, meanline=True, showfliers=showfliers, vert=False, widths=0.9, labels=y_label)
        else:
            p = {}

        
        with plt.warnings.catch_warnings():
            #warnings.simplefilter("error")
            try:
                plt.tight_layout()
            except Exception as e:
                print(e, prefix, data_type, title, y_label, file_path)


        if not log_scale:
            xlim = ax.get_xlim()

        #[t.label1.set_color(get_color(label)) for label, t in zip(y_label,ax.yaxis.get_major_ticks())]

        y_label2 = [label for label in y_label for _ in range(2)]

        if stretch_target:
            if "Target" in y_label:
                target_i = y_label.index("Target")
            elif "TumorTarget" in y_label:
                target_i = y_label.index("TumorTarget")
            elif "Pat-0 Target-TumorTarget" in y_label:
                target_i = y_label.index("Pat-0 Target-TumorTarget")
            elif "Pat-0 Target-TumorTarget - TumorTarget" in y_label:
                target_i = y_label.index("Pat-0 Target-TumorTarget - TumorTarget")
            else:
                #print(y_label)
                target_i = -1

        if "boxes" in p:
            [box.set_color(get_color(label)) for label, box in zip(y_label,p["boxes"])]
            [box.set_linewidth(5) for box in p["boxes"]]

        pvalues = {l:0 for l in y_label}
        if "whiskers" in p:
            [whisker.set_color(get_dark_color(label)) for label, whisker in zip( y_label2,p["whiskers"])]
            [whisker.set_linewidth(5) for whisker in p["whiskers"]]
            
            sig = []
            for i, x in enumerate(data):
                for j, y in enumerate(data[i+1:], start=i+1):
                    if False and (y_label[i] == "Target" or y_label[j] == "Target"):
                        pass
                    elif y_label[j] == "Dis LAIF" or y_label[i] == "Dis LAIF":
                        pass
                    elif len(y_label[j].split()) > 1 and len(y_label[i].split()) > 1 and y_label[j].split()[-2] == y_label[i].split()[-2]:
                        pass
                    #elif abs(i-j) == 1:
                    #    pass
                    else:
                        continue
                    try:
                        #_, pvalue = scipy.stats.mannwhitneyu(x, y, alternative='two-sided')
                        #_, pvalue3 = scipy.stats.ranksums(x, y)
                        #_, pvalue4 = scipy.stats.wilcoxon(x,y)
                        #_, pvalue5 = scipy.stats.kruskal(x, y)
                        _, pvalue = scipy.stats.ttest_rel(x, y)

                        #print(pvalue1, pvalue3, pvalue4, pvalue5)
                        #pvalue = np.median((pvalue1, pvalue3, pvalue4, pvalue5))
                        if y_label[j] == "Dis LAIF" or y_label[i] == "Dis LAIF":
                            pvalues[y_label[i]] = pvalue
                        if pvalue < significance_thresh:
                            sig.append((j-i, i, j, pvalue))
                    except Exception as e: 
                        pvalues[y_label[i]] = -1
                        #if not str(e).startswith("All "):
                        #    print(e, i, j, y_label[i], y_label[j])
                        #raise


            if significance_thresh > 0 and len(sig) > 0:
                minx = np.array([[True for _ in range(len(data)-1) ] for _ in range(len(sig))], dtype=bool)
                for d, i, j, pvalue in sorted(sig, key=lambda i: (i[0], i[2])):
                    lx = max(np.max(p["whiskers"][i*2].get_xdata()), np.max(p["whiskers"][i*2+1].get_xdata()))
                    ly = max(np.median(p["whiskers"][i*2].get_ydata()), np.median(p["whiskers"][i*2+1].get_ydata()))
                    rx = max(np.max(p["whiskers"][j*2].get_xdata()), np.max(p["whiskers"][j*2+1].get_xdata()))
                    ry = max(np.median(p["whiskers"][j*2].get_ydata()), np.median(p["whiskers"][j*2+1].get_ydata()))
                    
                    #dy = ry-ly
                    dy = j-i - 1.5# (len(data))//2
                

                    #bx = np.max([np.max(w.get_xdata()) for i, w in enumerate(p["whiskers"]) if y_label[i//2] != "Target"])

                    bx = xlim[1]
                    #bx = np.max([np.max(w.get_xdata()) for i, w in enumerate(p["whiskers"])])

                    ly = ly - 0.1*dy
                    ry = ry + 0.1*dy
                    rx = bx + 0.02*bx
                    lx = bx + 0.02*bx

                    for k in range(minx.shape[0]):
                        if minx[k,i:j].all():
                            bx = bx + (k+2)*0.02*bx
                            minx[k,i:j] = False
                            break
                    
                    #bx = bx + 0.05*bx*(j-i)

                    #if (j-i) > 1:
                    #    bx = bx + 0.01*bx*(i-1)
                    #rx = rx + (bx-rx)/2
                    #lx = lx + (bx-lx)/2

                    #print(i, j, ly, ry, bx)
                    plt.plot([lx, bx], [ly, ly], c=p["whiskers"][i*2].get_color(), clip_on=False)
                    plt.plot([bx,bx], [ly, ry], c='black', clip_on=False)
                    plt.plot([rx, bx], [ry, ry], c=p["whiskers"][j*2].get_color(), clip_on=False)
                    #plt.text(bx, (ly+ry)/2, "{:.4f}".format(pvalue) )

                #if log_scale:
                #    for k in reversed(range(minx.shape[0])):
                #        if minx[k].any():
                #            #xlim = (xlim[0], xlim[1] + (k+3)*0.02*xlim[1])
                #            break

        if "medians" in p:
            if stretch_target and target_i >= 0:
                p["medians"][target_i].set_ydata([ np.min([y for median in p["medians"] for y in median.get_ydata()]), np.max([y for median in p["medians"] for y in median.get_ydata()]) ])
                #p["medians"][target_i].set_xdata([ np.min([y for median in p["medians"] for y in median.get_xdata()]), np.max([y for median in p["medians"] for y in median.get_xdata()]) ])
            #if annotated_median:
            #    ydata = [ np.min([y for median in p["medians"] for y in median.get_ydata()]), np.max([y for median in p["medians"] for y in median.get_ydata()]) ]
            #    [median.set_ydata(ydata) for median in p["medians"]]
            [median.set_color(get_dark_color(label)) for label, median in zip(y_label,p["medians"])]
            [median.set_linewidth(5) for median in p["medians"]]
        if "means" in p:
            [mean.set_color(get_dark_color(label)) for label, mean in zip(y_label,p["means"])]
            [mean.set_linewidth(5) for mean in p["means"]]
        if "caps" in p:
            [cap.set_color(get_dark_color(label)) for label, cap in zip(y_label2,p["caps"])]
            [cap.set_linewidth(5) for cap in p["caps"]]
        if "fliers" in p:
            [flier.set_color(get_dark_color(label)) for label, flier in zip(y_label,p["fliers"])]
            [flier.set_markerfacecolor(get_dark_color(label)) for label, flier in zip(y_label,p["fliers"])]
            [flier.set_fillstyle("full") for label, flier in zip(y_label,p["fliers"])]

        #pos = range(0, len(data))
        #ax.set_yticks(pos)
        #ax.set_yticklabels(y_label)

        #max_length = str(max(len(title), np.max([len(l) for l in y_label])) + 2)

        if stats:
            #tabular = "\\begin{table}\n\\centering\n\\resizebox{\\linewidth}{!}{\n\\begin{tabular}{| c | c | c | c | c | c | c | c |} \n \\hline \n"
            #tabular = "\\begin{longtable}{| c | c | c | c | c | c | c | c |}\n\hline\n"
            tabular = "\\begin{longtable}{| c | c | c | c | c | c |}\n\hline\n"
            #tabular += prefix + y_labels[data_type] + " & Median & Average & Std & P \\\\ \\hline \n"
            #tabular += "Architecture & Loss & Filter & Epochs & Orientation & Median & Average & Std \\\\ \\hline \n \hline \n"
            tabular += "Architecture & Loss & Filter & Epochs & Slice & Average \\\\ \\hline \n \hline \n"
            #print(("{:" + max_length + "} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}").format(title, "min", "q25", "median", "q75", "max", "avg"))
        for i, (d, l) in enumerate(zip(data, y_label)):
            if(len(d) == 0): continue
            if not show_box:
                plt.scatter(d, [i+1]*len(d), color=get_color(l), linewidths="9")
                means = plt.scatter(np.mean(d), [i+1,], color=get_dark_color(l), linewidths="120", marker="|")
                for m in means.get_paths():
                    m.vertices = [[m.vertices[0][0], m.vertices[0][1]-3], [m.vertices[1][0], m.vertices[1][1]+3]]

            avg = np.mean(d)
            q0 = np.quantile(d, 0)
            q25 = np.quantile(d, 0.25)
            q5 = np.median(d)
            q75 = np.quantile(d, 0.75)
            q1 = np.quantile(d, 1)
            std = np.std(d)

            if annotated_median:
                ax.annotate("median: {:.2f} average: {:.2f}".format(q5,avg), xy=(1, (i+0.5) / len(data)), xycoords='axes fraction', xytext=(5,0), textcoords='offset pixels', horizontalalignment='left', verticalalignment='center')

            if stats:
                #tabular += ("{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \\hline \n").format(l.split(' - ')[-1], q0, q25, q5, q75, q1, avg, std).replace('_', ' ')
                if ': ' in l:
                    name, l1 = l.split(' - ')[-1].replace('_',' ').split(': ', maxsplit=1)
                    #if len(l1.split(' ')) < 7:
                    #    print(l1)
                    loss = l1.split(' ')[0]
                    filters = l1.split(' ')[2]
                    epochs = l1.split(' ')[3]
                    orientation = l1.split(' ')[5]
                else:
                    name = l
                    loss = ""
                    filters = ""
                    epochs = ""
                    orientation = ""
                
                if loss in titles:
                    #tabular += ("{} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline \n").format(name.replace("Multi-Channel", "MC"), titles[loss], filters, epochs, orientation, q5, avg, std).replace('_', ' ')
                    tabular += ("{} & {} & {} & {} & {} & ${:.5f} \\pm {:.5f}$ \\\\ \\hline \n").format(name.replace("Multi-Channel", "MC"), titles[loss], filters, epochs, orientation, avg, std).replace('_', ' ')
                else:
                    #tabular += ("{} & {} & {} & {} & {} & {:.2f} & {:.2f} & {:.2f} \\\\ \\hline \n").format(name.replace("Multi-Channel", "MC"), loss, filters, epochs, orientation, q5, avg, std).replace('_', ' ')
                    tabular += ("{} & {} & {} & {} & {} & ${:.5f} \\pm {:.5f}$ \\\\ \\hline \n").format(name.replace("Multi-Channel", "MC"), loss, filters, epochs, orientation, avg, std).replace('_', ' ')
                #print(("{:" + max_length + "} {:15.4f} {:15.4f} {:15.4f} {:15.4f} {:15.4f} {:15.4f}").format(l, q0, q25, q5, q75, q1, avg))
            #if np.isfinite(q5) and annotated_median:
                #ax.annotate("median = {:.2f}".format(q5), xy=(q5, i+1), xycoords='data', xytext=(10, 27), 
                #                    textcoords='offset pixels', arrowprops=None, horizontalalignment='left', 
                #                    verticalalignment='center')
                #ax.annotate("median = {:.2f}".format(q5), xy=(q5, i+1), xycoords='data', xytext=(1.01, 0), 
                #                    textcoords=('axes fraction', 'offset pixels'), arrowprops=None, horizontalalignment='left', 
                #                    verticalalignment='center')
        ax.set_xlabel(prefix + x_labels[data_type])
        
        if not show_box:
            ts = [t for t in ax.get_yticklabels()]
            for i,label in enumerate(y_label):
                if len(ts) == i+1:
                    break
                ts[i+1].set_text(label)
                ts[i+1].set_color(get_color(label))
            ax.set_yticklabels(ts)

        if log_scale:
            #ax.set_xscale('symlog', linthershx=0.001)
            #ax.set_xscale('log')
            #ax.set_xscale('logit')
            ax.set_xscale('linear')
            ax.set_xlim(xlim)
        else:
            ax.set_xscale('linear')
            ax.set_xlim(xlim)
        #ax.set_yticklabels(ts)
        #[t.label1.set_color(get_color(label)) for label, t in zip(y_label,ax.yaxis.get_major_ticks())]
        #ax.set_yticklabels(y_label)
        #[t.label1.set_text(label) for label, t in zip(y_label,ax.yaxis.get_major_ticks())]

        with plt.warnings.catch_warnings():
            #warnings.simplefilter("error")
            try:
                plt.tight_layout()
            except Exception as e:
                print(e, prefix, data_type, title, len(y_label), file_path)

        if stats:
            tabular += "\\caption{}\n"
            tabular += "\\label{tab:" + file_path.split('\\')[-1] + "} \n\end{longtable}\n"
            with open(file_path + ".tex", "w", encoding="utf8") as f:
                f.write(tabular)
        #savefig(file_path + ".svg", transparent=True)
        plt.savefig(file_path + ".png", transparent=True)
    except Exception as e:
        raise e
    finally:
        plt.close()


def save_plot_clean(data, title, y_label, file_path, stretch_target=True, showmeans=True, showfliers=False, significance_thresh=0):
    f = plt.figure(figsize=(30,1*len(data)+5))
    ax = plt.subplot(111)
    [i.set_linewidth(5) for i in ax.spines.values()]

    ax.set_title(title, pad=20)

    p = plt.boxplot(data, notch=False, whis=[5,95], showmeans=showmeans, meanline=True, showfliers=showfliers, vert=False, widths=0.9, labels=y_label)

    if stretch_target:
        if "Target" in y_label:
            target_i = y_label.index("Target")
        elif "TumorTarget" in y_label:
            target_i = y_label.index("TumorTarget")
        elif "Pat-0 Target-TumorTarget" in y_label:
            target_i = y_label.index("Pat-0 Target-TumorTarget")
        elif "Pat-0 Target-TumorTarget - TumorTarget" in y_label:
            target_i = y_label.index("Pat-0 Target-TumorTarget - TumorTarget")
        else:
            target_i = -1

    y_label2 = [label for label in y_label for _ in range(2)]

    if "boxes" in p:
        [box.set_color(get_color(label)) for label, box in zip(y_label,p["boxes"])]
        [box.set_linewidth(5) for box in p["boxes"]]
    if "medians" in p:
        if stretch_target and target_i >= 0:
            p["medians"][target_i].set_ydata([ np.min([y for median in p["medians"] for y in median.get_ydata()]), np.max([y for median in p["medians"] for y in median.get_ydata()]) ])
        [median.set_color(get_dark_color(label)) for label, median in zip(y_label,p["medians"])]
        [median.set_linewidth(5) for median in p["medians"]]
    if "means" in p:
        [mean.set_color(get_dark_color(label)) for label, mean in zip(y_label,p["means"])]
        [mean.set_linewidth(5) for mean in p["means"]]
    if "caps" in p:
        [cap.set_color(get_dark_color(label)) for label, cap in zip(y_label2,p["caps"])]
        [cap.set_linewidth(5) for cap in p["caps"]]
    if "fliers" in p:
        [flier.set_color(get_dark_color(label)) for label, flier in zip(y_label,p["fliers"])]
        [flier.set_markerfacecolor(get_dark_color(label)) for label, flier in zip(y_label,p["fliers"])]
        [flier.set_fillstyle("full") for label, flier in zip(y_label,p["fliers"])]
    if "whiskers" in p:
        [whisker.set_color(get_dark_color(label)) for label, whisker in zip( y_label2,p["whiskers"])]
        [whisker.set_linewidth(5) for whisker in p["whiskers"]]
        
        if significance_thresh > 0:
            xlim = ax.get_xlim()
            sig = []
            for i, x in enumerate(data):
                for j, y in enumerate(data[i+1:], start=i+1):
                    if y_label[i] == "Target" or y_label[j] == "Target":
                        pass
                    if y_label[j] != "Dis LAIF" and y_label[i] != "Dis LAIF":
                        pass
                    try:
                        _, pvalue = scipy.stats.mannwhitneyu(x, y, alternative='two-sided')
                        if pvalue < significance_thresh:
                            sig.append((j-i, i, j, pvalue))
                    except Exception as e: 
                        print(e, i, j, y_label[i], y_label[j])


            if len(sig) > 0:
                minx = np.array([[True for _ in range(len(data)-1) ] for _ in range(len(sig))], dtype=bool)
                for d, i, j, pvalue in sorted(sig, key=lambda i: (i[0], i[2])):
                    lx = max(np.max(p["whiskers"][i*2].get_xdata()), np.max(p["whiskers"][i*2+1].get_xdata()))
                    ly = max(np.median(p["whiskers"][i*2].get_ydata()), np.median(p["whiskers"][i*2+1].get_ydata()))
                    rx = max(np.max(p["whiskers"][j*2].get_xdata()), np.max(p["whiskers"][j*2+1].get_xdata()))
                    ry = max(np.median(p["whiskers"][j*2].get_ydata()), np.median(p["whiskers"][j*2+1].get_ydata()))
                    
                    dy = j-i - (len(data))//2
                
                    bx = xlim[1]
                    
                    ly = ly - 0.1*dy
                    ry = ry + 0.1*dy
                    rx = bx + 0.02*bx
                    lx = bx + 0.02*bx

                    for k in range(minx.shape[0]):
                        if minx[k,i:j].all():
                            bx = bx + (k+2)*0.02*bx
                            minx[k,i:j] = False
                            break
                    
                    plt.plot([lx, bx], [ly, ly], c=p["whiskers"][i*2].get_color(), clip_on=False)
                    plt.plot([bx,bx], [ly, ry], c='black', clip_on=False)
                    plt.plot([rx, bx], [ry, ry], c=p["whiskers"][j*2].get_color(), clip_on=False)
            ax.set_xlim(xlim)
                    

    with plt.warnings.catch_warnings():
        try:
            plt.tight_layout()
        except Exception as e:
            print(e, title, y_label, file_path)

    plt.savefig(file_path + ".png", transparent=True)
    plt.close()


if __name__ == "__main__":

    export_aif_stats = False
    export_aif_curves = False
    export_aif_eval = False
    
    dataDir = "./"

    if not os.path.exists(os.path.join(dataDir, "graphs")):
        os.makedirs(os.path.join(dataDir, "graphs"))
    if not os.path.exists(os.path.join(dataDir, "perf")):
        os.makedirs(os.path.join(dataDir, "perf"))
    if not os.path.exists(os.path.join(dataDir, "aifs")):
        os.makedirs(os.path.join(dataDir, "aifs"))
    if not os.path.exists(os.path.join(dataDir, "tumors")):
        os.makedirs(os.path.join(dataDir, "tumors"))

    matplotlib.rc("font", size=40)
    matplotlib.rc("lines", linewidth=3)

    if export_aif_stats:
        aif_csv = readCSV(os.path.join(dataDir, "aif.csv"))

        tabular = "\\begin{tabular}{|c||c|c|c||c|c|c|}\n\hline\n& \\multicolumn{3}{c||}{Mean Error compared to Manual Annotation} & \\multicolumn{3}{c|}{Median Error}\\\\ \n \\hline \n"
        tabular += "AIF Algorithm & $\\Delta t_{peak}$ & $\\Delta \\hat{S}(t_{peak})$ & $\\Delta AUC$ & $\\Delta t_{peak}$ & $\\Delta \\hat{S}(t_{peak})$ & $\\Delta AUC$ \\\\ \\hhline{|=||===||===|}\n"
        stats = {}
        aifs = {}
        for filename in aif_csv.keys():
            algorithm, _ = filename.rsplit(' ', maxsplit=1)
            if algorithm not in ["target", "DisLAIF", "chen2008", "shi2013", "gamma_variate", "parker2003"]:
                continue
            if algorithm not in aifs:
                aifs[algorithm] = []
            aifs[algorithm].append(aif_csv[filename])

        stats["target"] = {'peak': [], 'peak_t': [], 'auc': [], 'n_auc': [], 'n_peak': [], 'n_peak_t': []}
        for aif in aifs["target"]:
            stats["target"]["peak"].append(np.max(aif))
            stats["target"]["peak_t"].append(np.argmax(aif))
            stats["target"]["auc"].append(np.sum(aif))
            stats["target"]["n_peak"].append(np.max((aif-aif[0])))
            stats["target"]["n_auc"].append(np.sum((aif-aif[0]) / stats["target"]["n_peak"][-1] ))
            stats["target"]["n_peak_t"].append(np.argmax(aif-aif[0]))

        for algorithm in aifs:
            if algorithm not in stats:
                stats[algorithm] = {'peak': [], 'peak_t': [], 'auc': [], 'n_auc': [], 'n_peak': [], 'n_peak_t': []}
            else:
                continue
            for i, aif in enumerate(aifs[algorithm]):    
                stats[algorithm]['peak'].append(np.max(aif))
                stats[algorithm]['peak_t'].append(np.argmax(aif))
                stats[algorithm]['auc'].append(np.sum(aif))

                stats[algorithm]['n_peak'].append(np.abs(1-np.max((aif-aif[0])/stats["target"]["n_peak"][i])))
                stats[algorithm]['n_auc'].append( np.sum(np.abs((aifs["target"][i]-aifs["target"][0])/stats["target"]["n_peak"][i] - (aif-aif[0])/stats["target"]["n_peak"][i])))
                stats[algorithm]['n_peak_t'].append( np.abs(np.argmax(aif)-stats["target"]["peak_t"][i]) )

        for algorithm, values in stats.items():
            tabular += ("{} & ${:.3f} \\pm {:.3f}$ & ${:.3f} \\pm {:.3f}$ & ${:.3f} \\pm {:.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:.3f}$ \\\\ \\hline \n").format(algorithm, 
                        np.mean(values['n_peak_t']), np.std(values['n_peak_t']), np.mean(values['n_peak']), np.std(values['n_peak']),
                        np.mean(values['n_auc']), np.std(values['n_auc']), np.median(values['n_peak_t']), np.median(values['n_peak']), np.median(values['n_auc']) )

        tabular += "\n\end{tabular}\n"
        with open(os.path.join(dataDir, 'aifs/', "aif_stats.tex"), "w", encoding="utf8") as f:
            f.write(tabular)

    if export_aif_curves:
        aif_csv = readCSV(os.path.join(dataDir, "aif.csv"))
        aif = transform_curves(aif_csv)

        #print(aif.keys())
        for filename in aif.keys():
            break
            plt.figure(figsize=(30,15))
            x = aif[filename]["x"]*60
            if len(x) != len(aif[filename]["Target"]):
                x = np.arange(0,len(aif[filename]["Target"]))

            target_peak = np.max(aif[filename]["Target"])
            ax = plt.subplot(111)
            #if filename in titles:
            #    ax.set_title(titles[filename])
            #else:
            #    ax.set_title(filename)
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("t / s", fontsize=40)
            ax.set_ylabel("avg [$\hat{S}(t)$]", fontsize=40)
                        
            for algorithm in sorted(aif[filename].keys()):
                if algorithm not in [y_labels[l] for l in ("target", "target_eroded", "target_dilated", "target_right", "target_left", "target_up", "target_down")]:
                    continue
                plt.plot(x, aif[filename][algorithm]/target_peak, label=algorithm, linewidth=5, color=get_color(algorithm))
                
                xv = x[np.argmax(aif[filename][algorithm])]
                yv = np.max(aif[filename][algorithm])
                if algorithm == "Target":
                    ax.annotate("max S(t) = {:.0f}".format(yv), xy=(xv,yv), xycoords='data', xytext=(0.3, 0.6), 
                                textcoords='axes fraction', arrowprops={'facecolor':get_color("lines"), 'shrink':0.05}, horizontalalignment='left', 
                                verticalalignment='center')
                else:    
                    ax.annotate("max S(t) = {:.0f} ≙ {:2.0f}%".format(yv, 100*yv/target_peak), xy=(xv,yv), xycoords='data', xytext=(0.3, 0.7 if algorithm=="Eroded" else 0.5), 
                                textcoords='axes fraction', arrowprops={'facecolor':get_color("lines"), 'shrink':0.05}, horizontalalignment='left', 
                                verticalalignment='center')
                
            plt.legend(loc="upper right", fontsize=40)

            plt.savefig(os.path.join(dataDir, 'aifs/', "aif_diff-" + filename + ".png"), transparent=True)
            plt.close()

            
        
        for filename in aif.keys():
            break
            plt.figure(figsize=(30,15))
            x = aif[filename]["x"]*60
            if len(x) != len(aif[filename]["Target"]):
                x = np.arange(0,len(aif[filename]["Target"]))

            target_peak = np.max(aif[filename]["Target"])
            ax = plt.subplot(111)
            if filename in titles:
                ax.set_title(titles[filename])
            else:
                ax.set_title(filename)
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("t / s", fontsize=40)
            ax.set_ylabel("avg S(t)", fontsize=40)
            for algorithm in sorted(aif[filename].keys()):
                if algorithm not in [y_labels[l] for l in ("target", "target_middle", "target_upper", "target_lower", "target_middlep", "target_middlem")]:
                    continue
                plt.plot(x, aif[filename][algorithm]/target_peak, label=algorithm, linewidth=5, color=get_color(algorithm))
                
                xv = x[np.argmax(aif[filename][algorithm])]
                yv = np.max(aif[filename][algorithm])
                if algorithm == "Target":
                    ax.annotate("max S(t) = {:.0f}".format(yv), xy=(xv,yv), xycoords='data', xytext=(0.3, 0.6), 
                                textcoords='axes fraction', arrowprops={'facecolor':get_color("lines"), 'shrink':0.05}, horizontalalignment='left', 
                                verticalalignment='center')
                else:    
                    ax.annotate("max S(t) = {:.0f} ≙ {:2.0f}%".format(yv, 100*yv/target_peak), xy=(xv,yv), xycoords='data', xytext=(0.3, 0.7 if algorithm=="Eroded" else 0.5), 
                                textcoords='axes fraction', arrowprops={'facecolor':get_color("lines"), 'shrink':0.05}, horizontalalignment='left', 
                                verticalalignment='center')
            plt.legend(loc="upper right", fontsize=40)

            plt.savefig(os.path.join(dataDir, '/aifs/', "aif_2d-" + filename + ".png"), transparent=True)
            plt.close()
        
        for i, filename in enumerate(aif.keys()):
            break
            #data = []
            #for algorithm in aif[filename]:
            #    data.append(aif[filename][algorithm])
        
            plt.figure(figsize=(30,15))
            x = aif[filename]["x"]*60
            if len(x) != len(aif[filename]["Target"]):
                x = np.arange(0,len(aif[filename]["Target"]))
            target_peak = np.max(aif[filename]["Target"])
            ax = plt.subplot(111)
            #if filename in titles:
            #    ax.set_title(titles[filename])
            #else:
            #    ax.set_title(filename)
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("t / s", fontsize=40)
            ax.set_ylabel("avg S(t)", fontsize=40)
            for algorithm in sorted(aif[filename].keys()):
                if algorithm in ("x", "max05", "max05total", "No Operation", "Chen 2008", "Shi 2013", "GVF Fit"):
                    continue
                if not (algorithm.startswith("dante") or algorithm in ("Target", "Middle")):
                    continue
                plt.plot(x, aif[filename][algorithm]/target_peak, label=algorithm, linewidth=5, color=get_color(algorithm))

            plt.legend(loc="upper right", fontsize=30)

            plt.savefig(os.path.join(dataDir, '/aifs/', "aif-dante-" + str(i) + ".png"), transparent=True)
            plt.close()
    

        f = open("aif.csv", "w")
        for i,filename in enumerate(aif.keys()):
            plt.figure(figsize=(30,15))
            x = aif[filename]["x"]*60
            if len(x) != len(aif[filename]["Target"]):
                x = np.arange(0,len(aif[filename]["Target"]))
            target_peak = np.max(aif[filename]["Target"])
            ax = plt.subplot(111)
            
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("t / s", fontsize=40)
            ax.set_ylabel("avg $S(t)$", fontsize=40)
            for algorithm in sorted(aif[filename].keys()):
                if algorithm not in ("Dis LAIF", "Target", "Parker 2003", "Shi 2013", "Chen 2008", "GVF Fit"):
                    continue
                plt.plot(x, aif[filename][algorithm], label=algorithm, linewidth=5, color=get_color(algorithm))

            plt.legend(loc="upper right", fontsize=30)

            d = aif[filename]["Dis LAIF"]
            t = aif[filename]["Target"]
            f.write(str(i) + "," + str(np.abs(np.max(d)-np.max(t))) + "," + str(np.abs(np.argmax(d)-np.argmax(t))) + "," + str(np.abs(np.sum(d)-np.sum(t))) + "\n")

            plt.savefig(os.path.join(dataDir, '/aifs/', "aif-" + str(i) + ".png"), transparent=True)
            plt.close()
        
        f.close()
    
        for i,filename in enumerate(aif.keys()):
            plt.figure(figsize=(30,15))
            x = np.arange(0,len(aif[filename]["Target"]))
            target_peak = np.max(aif[filename]["Target"])
            ax = plt.subplot(111)
            
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("timestep", fontsize=40)
            ax.set_ylabel("avg $S(t)$", fontsize=40)
            for algorithm in sorted(aif[filename].keys()):
                if algorithm not in ("Dis LAIF", "Target", "Parker 2003", "Shi 2013", "Chen 2008", "GVF Fit"):
                    continue
                plt.plot(x, aif[filename][algorithm], label=algorithm, linewidth=5, color=get_color(algorithm))

            plt.legend(loc="upper right", fontsize=30)

            plt.savefig(os.path.join(dataDir, '/aifs/', "aif-no_time-" + str(i) + ".png"), transparent=True)
            plt.close()
    
        f = open("aif_norm.csv", "w")
        for i,filename in enumerate(aif.keys()):
            #data = []
            #for algorithm in aif[filename]:
            #    data.append(aif[filename][algorithm])
        
            plt.figure(figsize=(30,15))
            x = aif[filename]["x"]*60
            if len(x) != len(aif[filename]["Target"]):
                x = np.arange(0,len(aif[filename]["Target"]))
            target_peak = np.max(aif[filename]["Target"]-aif[filename]["Target"][0])
            ax = plt.subplot(111)
            #if i in titles:
            #    ax.set_title(titles[i])
            #else:
            #    ax.set_title("Patient " + str(i))
            ax.set_facecolor(get_color("lines"))
            ax.set_xlabel("t / s", fontsize=40)
            ax.set_ylabel("avg $\hat{S}(t)$", fontsize=40)
            for algorithm in sorted(aif[filename].keys()):
                if algorithm in ("x", "max05", "max05total", "No Operation", "Dante"):
                    continue
                if algorithm not in ("Dis LAIF", "Target", "Parker 2003", "Shi 2013", "Chen 2008", "GVF Fit"):
                    continue
                plt.plot(x, (aif[filename][algorithm]-aif[filename][algorithm][0])/target_peak, label=algorithm, linewidth=5, color=get_color(algorithm))

            plt.legend(loc="upper right", fontsize=30)

            d = aif[filename]["Dis LAIF"]
            t = aif[filename]["Target"]
            t = t-t[0]
            d = (d-d[0]) / np.max(t)
            t = t / np.max(t)
            f.write(str(i) + "," + str(np.abs(np.max(d)-np.max(t))) + "," + str(np.abs(np.argmax(d)-np.argmax(t))) + "," + str(np.abs(np.sum(d)-np.sum(t))) + "\n")

            plt.savefig(os.path.join(dataDir, 'aifs/', "aif-norm-" + str(i) + ".png"), transparent=True)
            plt.close()

    if export_aif_eval:
        aif_eval = readCSV(os.path.join(dataDir, "aif_eval.csv"))
        datafilters, names = get_datafilter(aif_eval, "modelName", ["noop", "null", "max05total", "max05", "shi2013", "chen2008", "parker2003", 
        "gamma_variate", "dante", "Dis LAIF", "DisLAIF_dilated", "DisLAIF", "DisLAIF_grow", "DisLAIF_opening", "OldDisLAIF", "DisLAIF_middle", 
        "DisLAIF_middlep", "DisLAIF_middlem", "DisLAIF_upper", "DisLAIF_lower", "dante_upper", "dante_lower", "dante_middle", "dante_middlep", 
        "dante_middlem"], None)
        for y in ("time", "acc", "label_acc", "dice", "delta_t_peak", "delta_peak", "delta_auc", "ratio_t_peak", "ratio_peak", "ratio_auc", "aif_mse", "overlap_dice", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            break
            if y not in aif_eval:
                continue
            data = []
            for datafilter in datafilters:
                if y == "time":
                    data.append(aif_eval["modelIndex"][datafilter])
                else:
                    data.append(aif_eval[y][datafilter])
            #print(y, len(data), [len(d) for d in data], [name for name in names])
            meanData = [np.median(data) for data in data]
            avgData = [np.average(data) for data in data]
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "mean_surface_distane", "median_surface_distance"):
                indexes = np.lexsort((avgData, meanData))[::-1]
            elif y in ("abs_ratio_peak", "ratio_peak", "ratio_t_peak", "ratio_auc"):
                indexes = np.lexsort((np.abs(avgData), np.abs(meanData)))[::-1]
            else:
                indexes = np.lexsort((avgData, meanData))
            sdata = []
            snames = []
            for i in indexes:
                sdata.append(data[i])
                snames.append(names[i])
            save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_diff-" + y), showmeans=(y!="median_surface_distance"))

        datafilters, names = get_datafilter(aif_eval, "modelName", ["noop", "null", "max05total", "max05", "shi2013", "chen2008", "parker2003", 
            "gamma_variate", "DisLAIF_dilated", "DisLAIF_grow", "DisLAIF_opening", "DisLAIF_middle", 
            "middle_up", "middle_down", "middle_fore", "middle_back", "middle_left", "middle_right", "_middle_eroded", "_middle_dilated", 
            "DisLAIF_middlep", "DisLAIF_middlem", "DisLAIF_upper", "DisLAIF_lower", "dante", "dante_upper", "dante_lower", "dante_middle", "dante_middlep", 
            "dante_middlem", "target_right", "target_left", "target_up", "target_down", "target_fore", "target_back", "target_eroded", "target_dilated",
            "DisLAIF_4_8", "DisLAIF_3_9", "OldDisLAIF", "DisLAIF4", "target_leftSide", "target_rightSide",
            "DisLAIF_5", "DisLAIF_1", "DisLAIF_4_3_8", "DisLAIF_4_6_8", "DisLAIF_size", "DisLAIF_4", "DisLAIF_4_6", "DisLAIF_gvt_pixel", "DisLAIF_3_8_9",
            "DisLAIF_3", "DisLAIF_2", "DisLAIF_2_9", "DisLAIF_frangi", "DisLAIF_frangi2", "DisLAIF_sato", "DisLAIF_sato2", "DisLAIF_meijering",
            "target_sato", "target_frangi", "target_hessian", "DisLAIF_hessian", "target_test",
        "parker2003un", "shi2013un", "DisLAIFun", "gamma_variateun", "chen2008un"], None)
        for y in ("time", "acc", "label_acc", "dice", "delta_t_peak", "delta_peak", "delta_auc", "ratio_t_peak", "ratio_peak", "ratio_auc", "aif_mse", "overlap_dice", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            if y not in aif_eval:
                continue
            data = []
            used_names = []
            for datafilter, name in zip(datafilters, names):
                if name.endswith("Left") or name.endswith("Right"):
                    continue
                if y == "time":
                    data.append(aif_eval["modelIndex"][datafilter])
                    used_names.append(name)
                else:
                    data.append(aif_eval[y][datafilter])
                    used_names.append(name)
            #print(y, len(data), [len(d) for d in data], [name for name in names])
            avgData = [np.median(data) for data in data]
            meanData = [np.average(data) for data in data]
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "mean_surface_distane", "median_surface_distance"):
                indexes = np.lexsort((avgData, meanData))[::-1]
            elif y in ("abs_ratio_peak", "ratio_peak", "ratio_t_peak", "ratio_auc"):
                indexes = np.lexsort((np.abs(avgData), np.abs(meanData)))[::-1]
            else:
                indexes = np.lexsort((avgData, meanData))
            sdata = []
            snames = []
            for i in indexes:
                sdata.append(data[i])
                snames.append(used_names[i])
            save_plot(sdata, y, "", snames, os.path.join(dataDir, "graphs", "aif_2d-" + y), showmeans=(y not in ("median_surface_distance","mean_surface_distance")), showfliers=False, stats=True, annotated_median=False, significance_thresh=0.05)


        datafilters, names = get_datafilter(aif_eval, "modelName", ["noop", "null", "max05total", "OldDisLAIF", "DisLAIF_dilated", "DisLAIF",
        "DisLAIF_opening", "DisLAIF_grow", "DisLAIF_upper", "DisLAIF_lower", "DisLAIF_middle", "DisLAIF_middlep", "DisLAIF_middlem", 
        "DisLAIF_frangi2", "DisLAIF_hessian", "DisLAIF_sato2", "target_hessian", "DisLAIF4", "DisLAIF_meijering", "DisLAIF_frangi", "DisLAIF_sato",
        "target_meijering", "target_frangi", "target_sato", "target_test", "DisLAIF_3_9",
        "shi2013", "chen2008", "max05", "gamma_variate", "parker2003", "target_middlep", "target_middlem",
        "middle_leftSide", "middle_rightSide", "middle_up", "middle_down", "middle_left", "middle_right", "middle_back", "middle_fore", "middle_eroded", "middle_dilated",
        "dante", "dante_lower", "dante_upper", "dante_middle", "dante_middlep", "dante_middlem",
        "parker2003un", "shi2013un", "DisLAIFun", "gamma_variateun", "chen2008un"], None)
        for y in ("time", "acc", "label_acc", "dice", "delta_t_peak", "delta_peak", "delta_auc", "ratio_t_peak", "ratio_peak", "abs_ratio_peak", "ratio_auc", "aif_mse", "overlap_dice", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            if y not in aif_eval and y != "abs_ratio_peak":
                continue
            data = []
            use_names = []
            for datafilter, name in zip(datafilters, names):
                if y not in ("acc", "dice") and name == "null":
                    continue
                if name.endswith("Left") or name.endswith("Right"):
                    continue
                if y == "time":
                    data.append(aif_eval["modelIndex"][datafilter])
                elif y == "abs_ratio_peak":
                    data.append(np.abs(aif_eval["ratio_peak"][datafilter]))
                else:
                    data.append(aif_eval[y][datafilter])
                use_names.append(name)
            meanData = [np.median(data) for data in data]
            avgData = [np.average(data) for data in data]
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "mean_surface_distane", "median_surface_distance"):
                indexes = np.lexsort((avgData, meanData))[::-1]
            elif y in ("abs_ratio_peak", "ratio_peak", "ratio_t_peak", "ratio_auc"):
                indexes = np.lexsort((np.abs(avgData), np.abs(meanData)))[::-1]
            else:
                indexes = np.lexsort((avgData, meanData))
            sdata = []
            snames = []
            for i in indexes:
                sdata.append(data[i])
                snames.append(use_names[i])
            if y == "abs_ratio_peak":
                save_plot(sdata, "ratio_peak", "ratio_peak", snames, os.path.join(dataDir, "graphs", "aif_targ-" + y), True)
            else:
                save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_targ-" + y), True, showmeans=(y not in ("median_surface_distance","mean_surface_distance")), showfliers=True, log_scale=y=="aif_mse_")

        datafilters, names = get_datafilter(aif_eval, "modelName", ["noop", "null", "max05total", "DisLAIF_dilated", "dante",
        "DisLAIF_opening", "DisLAIF_grow", "DisLAIF_upper", "DisLAIF_lower", "DisLAIF_middle", "DisLAIF_middlep", "DisLAIF_middlem",
        "_target_middlep", "target_middlem", "target_up", "target_down", "target_left", "target_right", "mouridsen2006", "OldDisLAIF",
        "target_test", "target_sato", "target_frangi", "middle_up", "middle_down", "middle_left", "middle_right", "middle_dilated", "middle_eroded", 
        "middle_fore", "middle_back", "target_fore", "target_back", "target_leftSide", "target_rightSide",
        "middle_leftSide", "middle_rightSide", "DisLAIF_frangi", "DisLAIF_sato", "DisLAIF_meijering", "DisLAIF4", "DisLAIF_hessian", "DisLAIF_frangi2",
        "DisLAIF_sato2", "target_hessian", "target_eroded", "target_dilated", "target_middle", "target_middlep", "target_middlem", "target_lower", "target_upper", "max05",
        "DisLAIF_3_9", "DisLAIF_2_9", "DisLAIF_2", "DisLAIF_3", "DisLAIF_3_8_9", "DisLAIF_gvt_pixel", "DisLAIF_5",
        "DisLAIF_4_6", "DisLAIF_4", "DisLAIF_4_6_8", "DisLAIF_4_3_8", "DisLAIF_1", "DisLAIF_4_8", "leftSide", "rightSide", "DisLAIF_size",
        "dante_upper", "dante_lower", "dante_middle", "dante_middlem", "dante_middlep", "gamma_variate",
        "parker2003un", "shi2013un", "DisLAIFun", "gamma_variateun", "chen2008un", "chen2008_", "chen2008_un"], None)
        for y in ("time", "acc", "label_acc", "dice", "delta_t_peak", "delta_peak", "delta_auc", "ratio_t_peak", "ratio_peak", "abs_ratio_peak", "ratio_auc", "aif_mse", "overlap_dice", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            if y not in aif_eval and y != "abs_ratio_peak":
                continue
            data = []
            use_names = []
            for datafilter, name in zip(datafilters, names):
                if y not in ("acc", "dice") and name == "null":
                    continue
                if name.endswith("Left") or name.endswith("Right"):
                    continue
                if y == "time":
                    data.append(aif_eval["modelIndex"][datafilter])
                    use_names.append(name)
                elif y == "abs_ratio_peak":
                    data.append(np.abs(aif_eval["ratio_peak"][datafilter]))
                    use_names.append(name)
                else:
                    #if name.startswith("Dis LAIF") and(aif_eval[y][datafilter]>0.1).any():
                    #    data.append(aif_eval[y][datafilter][aif_eval[y][datafilter]>0.1])
                    #    use_names.append(name)
                    #else:
                        data.append(aif_eval[y][datafilter])
                        use_names.append(name)
            
            avgData = [np.median(data) for data in data]
            meanData = [np.average(data) for data in data]
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "mean_surface_distane", "median_surface_distance"):
                indexes = np.lexsort((avgData, meanData))[::-1]
            elif y in ("abs_ratio_peak", "ratio_peak", "ratio_t_peak", "ratio_auc"):
                indexes = np.lexsort((np.abs(avgData), np.abs(meanData)))[::-1]
            else:
                indexes = np.lexsort((avgData, meanData))
            sdata = []
            snames = []
            for i in indexes:
                sdata.append(data[i])
                snames.append(use_names[i])
            if y == "abs_ratio_peak":
                save_plot(sdata, "ratio_peak", "ratio_peak", snames, os.path.join(dataDir, "graphs", "aif_eval-" + y), stats=True, significance_thresh=0.05, annotated_median=False)
            else:
                save_plot(sdata, y, "", snames, os.path.join(dataDir, "graphs", "aif_eval-" + y), stats=True, showmeans=(y not in ("median_surface_distance","mean_surface_distance")), log_scale=False, annotated_median=False, significance_thresh=0.05)

        aif_select = readCSV(os.path.join(dataDir, "aif_select.csv"))
        datafilters, names = get_datafilter(aif_select, "modelName", ["parker2003un", "chen2008un", "shi2013un", "chen2008_un", "gamma_variateun", "DisLAIFun", "chen2008_"], None)
        for y in ("time", "acc", "label_acc", "dice", "delta_t_peak", "delta_peak", "delta_auc", "ratio_t_peak", "ratio_peak", "ratio_auc", "aif_mse", "overlap_dice", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            if y not in aif_select:
                continue
            data = []
            use_names = []
            for datafilter, name in zip(datafilters, names):
                if name not in ["Target", "Dis LAIF", "Shi 2013", "Chen 2008", "Parker 2003"]: continue
                use_names.append(name)
                if y == "time":
                    if name == "Target":
                        data.append(aif_select["modelIndex"][datafilter]+5*60)
                    else:
                        data.append(aif_select["modelIndex"][datafilter])
                else:
                    data.append(aif_select[y][datafilter])
            meanData = [np.median(data) for data in data]
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "ratio_peak", "overlap_false_negative", "overlap_false_positive"):
                indexes = np.argsort(meanData)[::-1]
            else:
                indexes = np.argsort(meanData)
            sdata = []
            snames = []
            for i in indexes:
                sdata.append(data[i])
                snames.append(use_names[i])
            save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_select-" + y), stats=True, significance_thresh=0.05, annotated_median=False)

        aif_eval_d = readCSV(os.path.join(dataDir, "aif_eval_d.csv"))
        datafilters2, names2 = get_datafilter(aif_eval_d, "modelName", [], None)
        datafilters = []
        names = []
        for d,n in zip(datafilters2,names2):
            if '5' not in n and '3' not in n or '5' in n and '3' in n:
                if n == "Dante":
                    names.append("Steps: 123456789")
                else:
                    names.append(n.replace("Dante.", "Steps: 1"))
                datafilters.append(d)
        datafilter = np.array(datafilter)
        names = np.array(names)
        best = {}
        for y in ("time", "delta_t_peak", "delta_peak", "delta_auc", "aif_mse", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            best[y] = set(names)
            if y not in aif_eval_d or len(aif_eval_d[y])==0:
                continue
            data = []
            for datafilter in datafilters:
                if y == "time":
                    data.append(aif_eval_d["modelIndex"][datafilter])
                else:
                    data.append(aif_eval_d[y][datafilter])
            
            #data = np.array(data)
            #print(data.shape, np.median(data,axis=1).shape, names.shape)
            #meanData = np.array([np.median(data) for data in data])
            meanData = np.array([np.average(data) for data in data])
            #meanData = np.median(data, axis=1)
            indexes = np.argsort(meanData)
            if y in ("time", "delta_t_peak", "delta_peak", "delta_auc", "aif_mse", "overlap_false_negative", "overlap_false_positive"):
                #best[y] = best[y].intersection(set( names[ meanData <= meanData[indexes][10] ] ))
                if len(meanData[indexes]) > 10:
                    best[y] = best[y].intersection(set( names[ meanData <= meanData[indexes][10] ] ))
            sdata = []
            snames = []
            for i in indexes:
                if '5' not in names[i] and '3' not in names[i] or '5' in names[i] and '3' in names[i]:
                    sdata.append(data[i])
                    snames.append(names[i].replace(".2", ".12"))
            #print(data.shape, np.median(data,axis=1).shape, names.shape)
            save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_eval_d-" + y))
        for y in ("acc", "label_acc", "dice", "overlap_dice"):
            best[y] = set(names)
            if y not in aif_eval_d or len(aif_eval_d[y])==0:
                continue
            data = []
            for datafilter in datafilters:
                data.append(aif_eval_d[y][datafilter])
            
            #data = np.array(data)
            #meanData = np.array([np.median(data) for data in data])
            meanData = np.array([np.average(data) for data in data])
            indexes = np.argsort(meanData)[::-1]
        
            if y in ("acc", "label_acc", "dice", "overlap_dice"):
                #print(best)
                if len(meanData[indexes]) > 10:
                    best[y] = best[y].intersection(set(names[ meanData >= meanData[indexes][10] ]))
                #print(y, best)
            #print(indexes.shape)
            sdata = []
            snames = []
            for i in indexes:
                if '5' not in names[i] and '3' not in names[i] or '5' in names[i] and '3' in names[i]:
                    sdata.append(data[i])
                    snames.append(names[i].replace(".2", ".12"))
            #print(data.shape, np.median(data,axis=1).shape, names.shape)
            save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_eval_d-" + y))
        for y in ("time", "acc", "label_acc", "dice", "overlap_dice","time", "delta_t_peak", "delta_peak", "delta_auc", "aif_mse", "overlap_false_negative", "overlap_false_positive", "mean_surface_distance", "median_surface_distance"):
            if y not in aif_eval_d or len(aif_eval_d[y])==0:
                continue
            data = []
            for datafilter in datafilters:
                if y == "time":
                    data.append(aif_eval_d["modelIndex"][datafilter])
                else:
                    data.append(aif_eval_d[y][datafilter])
            
            meanData = np.array([np.median(data) for data in data])
            #avgData = np.average(data,axis=1)
            avgData = np.array([np.average(data) for data in data])
            if y in ("time", "aif_mse", "delta_t_peak", "delta_peak", "delta_auc", "mean_surface_distance", "median_surface_distance"):
                #indexes = np.lexsort((avgData, meanData))[::-1]
                indexes = np.lexsort((meanData, avgData))[::-1]
            elif y in ("abs_ratio_peak", "ratio_peak", "ratio_t_peak", "ratio_auc"):
                #indexes = np.lexsort((np.abs(avgData), np.abs(meanData)))[::-1]
                indexes = np.lexsort((np.abs(meanData), np.abs(avgData)))[::-1]
            else:
                #indexes = np.lexsort((avgData, meanData))
                indexes = np.lexsort((meanData, avgData))
            sdata = []
            snames = []
            for i in indexes:
                if names[i] in best[y]:
                    if '5' not in names[i] and '3' not in names[i] or '5' in names[i] and '3' in names[i]:
                        sdata.append(data[i])
                        snames.append(names[i].replace(".2", ".12"))
            #print(data.shape, np.median(data,axis=1).shape, names.shape)
            save_plot(sdata, y, y, snames, os.path.join(dataDir, "graphs", "aif_eval_d-best-" + y), annotated_median=False)
        
        #print(best["dice"].intersection(best["aif_mse"]))
        #print(best["acc"].intersection(best["aif_mse"]))
        #print(best["dice"].intersection(best["acc"]))
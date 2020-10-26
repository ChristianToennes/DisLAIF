import numpy as np
import time
import scipy
from scipy import ndimage
import skimage
import SimpleITK as sitk
import os.path

dante_timing = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 'sum': []}

def pipeline(image, steps, timing = False, save_step_path=None, dso=None):
    mask = np.zeros_like(image)
    timestep = 0

    if timing and 'sum' not in timing:
        timing['sum'] = []

    for step in steps:
        if timing:
            proctime = time.process_time()

        mask, timestep = step(image, mask, timestep)

        if timing:
            proctime = time.process_time() - proctime
            if step.__name__ not in timing:
                timing[step.__name__] = []
            timing[step.__name__].append(proctime)
        
        if save_step_path:
            OutImage = None
            mask_image = np.array(mask, dtype=image.dtype)
            if mask_image.dtype == np.bool:
                mask_image = np.array(mask_image, dtype=image.dtype)
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
            
            sitk.WriteImage(OutImage, save_step_path + step.__name__ + '.nrrd')

    if timing:
        timing['sum'].append(np.sum([dante_timing[i][-1] for i in timing.keys() if i != "sum"]))

    if timing:
        return mask, timestep, timing
    else:
        return mask, timestep

def DisLAIF(im, save_step_path=None, dso=None):
    steps = [step1, step2, step4_filter, step4, step6, step3, step9, step8, step9]
    mask, timestep = pipeline(im, steps, save_step_path=save_step_path, dso=dso)
    return mask, None

def DisLAIF4(im):
    steps = [step1, step2, step4_filter, step3, step6, step9, step8]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_frangi(im):
    steps = [step1, step2, step4_filter, step4_frangi, step3, step6, step9, step8]#], step9]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_sato(im):
    steps = [step1, step2, step4_filter, step4_sato, step3, step6, step9, step8]#, step9]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_meijering(im):
    steps = [step1, step2, step4_filter, step4_meijering, step3, step6, step9, step8]#], step9]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_frangi2(im):
    steps = [step1, step2, step4_filter, step4_frangi, step9, step8]#], step9]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_sato2(im):
    steps = [step1, step2, step4_filter, step4_sato, step9, step8]#, step9]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_hessian(im):
    steps = [step1, step2, step4_hessian, step3]#, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_gvt_pixel(im):
    steps = [step1, step2, step6_, step4, step3, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_1(im):
    steps = [step1, step2, step4, step3, step6, step8]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_2(im):
    steps = [step1, step2, step4, step3, step6]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_3(im):
    steps = [step1, step2, step4, step3]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_2_9(im):
    steps = [step1, step2, step4, step3, step6, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_3_9(im):
    steps = [step1, step2, step4, step3, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_3_8_9(im):
    steps = [step1, step2, step4, step3, step8, step9]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_4(im):
    steps = [step1, step2, step4]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_4_8(im):
    steps = [step1, step2, step4, step8]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_4_6(im):
    steps = [step1, step2, step4, step6]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_4_6_8(im):
    steps = [step1, step2, step4, step6, step8]
    mask, timestep = pipeline(im, steps)
    return mask, timestep
    
def DisLAIF_4_3_8(im):
    steps = [step1, step2, step4, step3, step8]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_5(im):
    steps = [step1, step2]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_upper(im):
    steps = [step1, step2, step4, step3, step6, step8, step9, upper]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_middle(im):
    steps = [step1, step2, step4, step3, step6, step8, step9, middle]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_middlep(im):
    steps = [step1, step2, step4, step3, step6, step8, step9, middlep]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_middlem(im):
    steps = [step1, step2, step4, step3, step6, step8, step9, middlem]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def DisLAIF_lower(im):
    steps = [step1, step2, step4, step3, step6, step8, step9, lower]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def OldDisLAIF(im):
    #step_mask = [True, False, True, False, True, False, True, True, False, False, False, True]
    step_mask = {1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 1, 10: 0}
    mask, timestep = dante(im, step_mask = step_mask)
    return mask, timestep

def DisLAIF_vessel(im):
    #step_mask = [True, False, True, False, True, False, True, True, False, False, False, False]
    step_mask = {1: 2, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 1, 10: 0}
    steps = [step1_vessel, step2, step4, step3, step6, step8, step9]
    mask, timestep = pipeline(im, steps)
    mask2, _ = dante(im, step_mask = step_mask)
    if (mask != mask2).any():
        print("error")
    return mask, timestep

def DisLAIF_grow(im):
    #step_mask = [True, False, True, False, True, False, False, True, False, False, True, True]
    step_mask = {1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2, 9: 1, 10: 0}
    steps = [step1, step2, step4, step3, step6, step8_, step9]
    mask, timestep = pipeline(im, steps)
    mask2, _ = dante(im, step_mask = step_mask)
    if (mask != mask2).any():
        print("error")
    return mask, timestep

def DisLAIF_dilated(im):
    #step_mask = [True, False, True, False, True, False, True, True, True, False, False, True]
    step_mask = {1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 1, 10: 1}
    steps = [step1, step2, step4, step3, step6, step8, step9, step10]
    mask, timestep = pipeline(im, steps)
    mask2, _ = dante(im, step_mask = step_mask)
    if (mask != mask2).any():
        print("error")
    return mask, timestep

def DisLAIF_opening(im):
    #step_mask = [True, False, True, False, True, False, True, True, True, True, False, True]
    step_mask = {1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 3, 9: 1, 10: 0}
    steps = [step1, step2, step4, step3, step6, step8, step2, step9]
    mask, timestep = pipeline(im, steps)
    mask2, _ = dante(im, step_mask = step_mask)
    if (mask != mask2).any():
        print("error")
    return mask, timestep

def DisLAIF_size(im):
    steps = [step1, step2, step4, step3, step6, step8, step9_]
    mask, timestep = pipeline(im, steps)
    return mask, timestep

def dante_upper(im):
    mask, timestep = dante(im)
    mask, timestep = upper(im, mask, timestep)
    return mask, timestep

def dante_middle(im):
    mask, timestep = dante(im)
    mask, timestep = middle(im, mask, timestep)
    return mask, timestep

def dante_middlep(im):
    mask, timestep = dante(im)
    mask, timestep = middlep(im, mask, timestep)
    return mask, timestep

def dante_middlem(im):
    mask, timestep = dante(im)
    mask, timestep = middlem(im, mask, timestep)
    return mask, timestep

def dante_lower(im):
    mask, timestep = dante(im)
    mask, timestep = lower(im, mask, timestep)
    return mask, timestep

def dante(im, timing=False, step_mask = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 0}):

    structure = np.zeros((3,3,3,3,3))
    structure[1, :, :, :, 1] = 1

    if timing:
        proctime = time.process_time()
    
    mask = np.zeros_like(im)
    timestep = 0

    if step_mask[1] == 1:
        mask, timestep = step1(im, mask, timestep)
    elif step_mask[1] == 2:
        mask, timestep = step4_vessel(im, mask, timestep)
        mask = np.array(mask > 0.5, dtype=int)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[1].append(proctime)
        
        proctime = time.process_time()
    
    if step_mask[2]==1:
        mask, timestep = step2(im, mask, timestep)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[2].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[3] == 1:
        early_mask, early_timestep = step3(im, mask, 0)
    else:
        early_mask = mask
        early_timestep = mask.shape[0]-1
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[3].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[4] == 1:
        mask, timestep = step4_filter(im, mask, 0)
        mask, timestep = step4(im, mask, 0)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[4].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[5] == 1:
        mask, timestep = step5(im, mask, timestep, early_timestep, early_mask)
    else:
        mask, timestep = step3(im, mask, timestep)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[5].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[6] == 1:
        mask, timestep = step6(im, mask, timestep)
    elif step_mask[6] == 2:
        mask, timestep = step6_(im, mask, timestep)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[6].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[7] == 1:
        mask, timestep = step7(im, mask, timestep)

    if timing:
        proctime = time.process_time() - proctime
        dante_timing[7].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[8] == 1:
        mask, timestep = step8(im, mask, timestep)
    elif step_mask[8] == 2:
        mask, timestep = step8_(im, mask, timestep)
    elif step_mask[8] == 3:
        mask, timestep = step8(im, mask, timestep)
        mask, timestep = step2(im, mask, timestep)

    if timing:
        proctime = time.process_time() - proctime
        dante_timing[8].append(proctime)
    

    if timing:
        proctime = time.process_time()
    
    if step_mask[9] == 1:
        mask, timestep = step9(im, mask, timestep)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[9].append(proctime)
    
        proctime = time.process_time()
    
    if step_mask[10] == 1:
        mask, timestep = step10(im, mask, timestep)
    
    if timing:
        proctime = time.process_time() - proctime
        dante_timing[10].append(proctime)
    
        dante_timing['sum'].append(np.sum([dante_timing[i][-1] for i in range(1,10)]))

    return mask, timestep

def step1(im, mask, timestep):
    q99 = np.quantile(im[:15], 0.99)
    mask = np.zeros_like(im)
    #mask[im>q99] = 1
    mask[:15][im[:15]>q99] = 1
    mask[:,:,mask.shape[2]//2:,:,:] = 0
    #mask[:,:,:,mask.shape[2]//3:-mask.shape[2]//3,:] = 0

    return mask, timestep

def step2(im, mask, timestep):
    structure = np.zeros((1,3,3,3,1))
    structure[0, :, 1, 1, 0] = 1
    structure[0, 1, :, 1, 0] = 1
    structure[0, 1, 1, :, 0] = 1

    new_mask = scipy.ndimage.morphology.binary_opening(mask, structure)
    return new_mask, timestep

def step3(image, mask, timestep):
    histLs = []
    iL = []
    histRs = []
    iR = []
    for i in range(mask.shape[0]):
        m = mask[i]>0
        if np.count_nonzero(m) == 0:
            continue
        #    histLs.append(np.histogram(np.zeros((image.shape[0]//4 + 2,)), range(0, image.shape[0]//4 + 2))[0])
        #    histRs.append(np.histogram(np.zeros((image.shape[0]//4 + 2,)), range(0, image.shape[0]//4 + 2))[0])
        im = (image*m)[:,:,:,:image.shape[3]//2,:]
        c = np.swapaxes(im.reshape(im.shape[0], -1), 0,1)
        dt = np.argmax(c, axis=1)
        dt = dt[dt>0]
        dt = dt[dt<im.shape[0]/4]
        histL = np.histogram(dt, range(0, im.shape[0]//4 + 2))[0]
        iL.append(i)

        histLs.append(histL)

        im = (image*m)[:,:,:,image.shape[3]//2:,:]
        c = np.swapaxes(im.reshape(im.shape[0], -1), 0,1)
        dt = np.argmax(c, axis=1)
        dt = dt[dt>0]
        dt = dt[dt<im.shape[0]/4]
        histR = np.histogram(dt, range(0, im.shape[0]//4 + 2))[0]
        iR.append(i)

        histRs.append(histR)

    if len(histLs) == 0 and len(histRs) == 0:
        return mask, timestep

    #histLSort = np.array([np.argsort(np.argsort(histL)) for histL in histLs])
    #histRSort = np.array([np.argsort(np.argsort(histR)) for histR in histRs])
    histLSort = np.array([histL for histL in histLs])
    histRSort = np.array([histR for histR in histRs])
    
    #print(histLSort.shape, np.max(histLSort, axis=1), np.argmax(histLSort, axis=1))
    #print(np.max(histRSort, axis=1), np.argmax(histRSort, axis=1))
    
    histLArgmax = np.argmax(histLSort,axis=1)
    histLMax = np.max(histLSort,axis=1)[histLArgmax<histLSort.shape[1]-1]
    iL = np.array(iL)[histLArgmax<histLSort.shape[1]-1]
    histLArgmax = histLArgmax[histLArgmax<histLSort.shape[1]-1]
    histLMax = histLMax[histLArgmax>0]
    iL = iL[histLArgmax>0]
    histLArgmax = histLArgmax[histLArgmax>0]
    #print(histLArgmax, histLMax, iL)

    histRArgmax = np.argmax(histRSort,axis=1)
    histRMax = np.max(histRSort,axis=1)[histRArgmax<histRSort.shape[1]-1]
    iR = np.array(iR)[histRArgmax<histRSort.shape[1]-1]
    histRArgmax = histRArgmax[histRArgmax<histRSort.shape[1]-1]
    histRMax = histRMax[histRArgmax>0]
    iR = iR[histRArgmax>0]
    histRArgmax = histRArgmax[histRArgmax>0]
    #print(histRArgmax, histRMax, iR)

    if len(iL) == 0:
        iL = iR
        histLMax = histRMax
        histLArgmax = histRArgmax
    if len(iR) == 0:
        iR = iL
        histRMax = histLMax
        histRArgmax = histLArgmax

    if len(iL) == 1:
        tL = iL[0]
        if tL in iR:
            tR = tL
        elif (tL-1) in iR:
            tR = tL-1
        elif (tL+1) in iR:
            tR = tL+1
        else:
            tR = iR[np.argmin(np.abs(iR-tL))]
    elif len(iR) == 1:
        tR = iR[0]
        if tR in iL:
            tL = tR
        elif (tR-1) in iL:
            tL = tR-1
        elif (tR+1) in iL:
            tL = tR+1
        else:
            tL = iL[np.argmin(np.abs(iL-tR))]
    else:
        tR = iR[np.argmax(histRMax)]
        tL = iL[np.argmax(histLMax)]
        if np.abs(tR-tL) > 2:
            if (np.abs(iL-tR)<2).any():
                tL = iL[np.argmin(np.abs(iL-tR))]
            elif (np.abs(iR-tL)<2).any():
                tR = iR[np.argmin(np.abs(iR-tL))]

    #t = np.argmax(np.count_nonzero(mask, axis=(1,2,3,4)))
    nz = np.sum(mask, axis=(1,2,3,4))
    t = tR if (nz[tR] > nz[tL]) else tL

    #print(np.sum(mask, axis=(1,2,3,4)), t, tR, tL, histRArgmax[np.argmax(histRMax)], histLArgmax[np.argmax(histLMax)])
    
    timestep_mask = np.zeros_like(mask, dtype=int)
    #timestep_mask[t] = np.array((mask[t-1] + mask[t] + mask[t+1])>0, dtype=int)
    #timestep_mask[t] = mask[t]
    timestep_mask[t] = (mask[tL] + mask[tR]) > 0
    return timestep_mask, t

def gvf(im):
    im = im-np.min(im)
    im = im / np.max(im)
    g = np.gradient(im)
    magSquared = g[0]*g[0]+g[1]*g[1]+g[2]*g[2]
    u = g[0]
    v = g[1]
    w = g[2]
    mu = 1

    for i in range(10):
        u = u + mu*6*scipy.ndimage.filters.laplace(u) - (u-g[0])*magSquared
        v = v + mu*6*scipy.ndimage.filters.laplace(v) - (v-g[1])*magSquared
        w = w + mu*6*scipy.ndimage.filters.laplace(w) - (w-g[2])*magSquared
    
    return np.array([u / np.linalg.norm(u),v / np.linalg.norm(v),w / np.linalg.norm(w)])

def step4_vessel(im, mask, timestep):
    Ts = []
    for i, im in enumerate(im):
        if i < 20:
            V = gvf(im[:,:,:,0])
            H = np.zeros(im.shape[:-1] + (im.ndim-1, im.ndim-1), dtype=im.dtype) 
            for k, grad_k in enumerate(V):
                # iterate over dimensions
                # apply gradient again to every component of the first derivative.
                tmp_grad = np.gradient(grad_k) 
                for l, grad_kl in enumerate(tmp_grad):
                    H[:, :, :, k, l] = grad_kl

            H[np.isnan(H)] = 0
            eigvals = np.abs(np.linalg.eigvals(H))
            l1 = eigvals[:, :, :, 0]
            l2 = eigvals[:, :, :, 1]
            l3 = eigvals[:, :, :, 2]
            Ra = l1 / np.sqrt(l2*l3)
            Rb = l2 / l3
            S = np.sqrt(l1*l1 + l2*l2 + l3*l3)

            a = 0.5
            b = 0.5
            T = (1 - np.exp(-Ra*Ra/(2*a*a)) ) * np.exp(-Rb*Rb/(2*b*b))
            T[l2>0] = 0
            T[l3>0] = 0
            
            Ts.append(T[:,:,:,np.newaxis])
        else:
            Ts.append(np.zeros_like(im))
    return np.array(Ts), timestep

def step4_filter(im, mask, timestep):
    structure = np.zeros((3,3,3,3,3))
    #structure[1, :, :, :, 1] = 1
    structure[1, :, 1, 1, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, 1, 1, :, 1] = 1
    labels, _no_features = ndimage.label(mask, structure=structure)

    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        if (slices[1].stop-slices[1].start) < labels.shape[1]/2.5:
            mask[slices][labels[slices]==i] = 0
            continue
        if slices[3].start < labels.shape[3]//2 and slices[3].stop > labels.shape[3]//2:
            mask[slices][labels[slices]==i] = 0
            continue

    return mask, timestep

def step4_frangi(im, mask, timestep):
    out_mask = np.zeros_like(im, dtype=float)
    structure = np.zeros((1,3,3,3,1))
    structure[0, :, :, :, 0] = 1
    #structure[1, :, :, 1, 1] = 1
    #structure[1, 1, :, :, 1] = 1
    #structure[1, :, 1, :, 1] = 1
    #labels, _no_features = ndimage.label(mask, structure=structure)
    
    #for i,slices in enumerate(ndimage.find_objects(labels), start=1):
    #    if (slices[1].stop-slices[1].start) < labels.shape[1]/2.0:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    #    if slices[3].start < labels.shape[3]//2 and slices[3].stop > labels.shape[3]//2:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    
    labels, _no_features = ndimage.label(mask, structure=structure)

    #print(im.shape, mask.shape)

    for i in range(im.shape[0]):
        if i >= mask.shape[0] or np.count_nonzero(mask[i]) > 0:
            out_mask[i,...,0] = skimage.filters.frangi(im[i,...,0], black_ridges=False)

    out_mask = out_mask*mask
    out_mask = out_mask / np.quantile(out_mask[out_mask>0], 0.99)
    
    rmask = np.zeros_like(out_mask, dtype=int)
    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.9), np.count_nonzero(out_mask[slices] > 0.9), area)
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.5) / np.count_nonzero(labels[slices]==i))
        #print(np.quantile(out_mask[slices][labels[slices]==i], 0.9), np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10], np.count_nonzero(out_mask[slices][labels[slices]==i]>1))
        if np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10] < 0.7:
            rmask[labels==i] = 0
        else:
            rmask[labels==i] = 1
    return rmask, timestep

def step4_sato(im, mask, timestep):
    out_mask = np.zeros_like(im, dtype=float)
    structure = np.zeros((1,3,3,3,1))
    #structure[1, :, :, :, 1] = 1
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    #labels, _no_features = ndimage.label(mask, structure=structure)

    #for i,slices in enumerate(ndimage.find_objects(labels), start=1):
    #    if (slices[1].stop-slices[1].start) < labels.shape[1]/2.0:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    #    if slices[3].start < labels.shape[3]//2 and slices[3].stop > labels.shape[3]//2:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    
    labels, _no_features = ndimage.label(mask, structure=structure)

    #print(im.shape, mask.shape)
    out_mask = np.zeros_like(im, dtype=float)
    
    for i in range(im.shape[0]):
        if np.count_nonzero(mask[i]) > 0:
            out_mask[i,...,0] = skimage.filters.sato(im[i,...,0], black_ridges=False)

    out_mask = out_mask*mask
    out_mask = out_mask / np.quantile(out_mask[out_mask>0], 0.99)
    
    mask = np.zeros_like(out_mask, dtype=int)
    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.9), np.count_nonzero(out_mask[slices] > 0.9), area)
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.5) / np.count_nonzero(labels[slices]==i))
        #print(np.quantile(out_mask[slices][labels[slices]==i], 0.9), np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10], np.count_nonzero(out_mask[slices][labels[slices]==i]>1))
        if np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10] < 0.7:
            mask[slices][labels[slices]==i] = 0
        else:
            mask[slices][labels[slices]==i] = 1
    return mask, timestep

def step4_meijering(im, mask, timestep):
    out_mask = np.zeros_like(im, dtype=float)
    structure = np.zeros((1,3,3,3,1))
    #structure[1, :, :, :, 1] = 1
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    #labels, _no_features = ndimage.label(mask, structure=structure)

    #for i,slices in enumerate(ndimage.find_objects(labels), start=1):
    #    if (slices[1].stop-slices[1].start) < labels.shape[1]/2.0:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    #    if slices[3].start < labels.shape[3]//2 and slices[3].stop > labels.shape[3]//2:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    
    labels, _no_features = ndimage.label(mask, structure=structure)

    #print(im.shape, mask.shape)
    out_mask = np.zeros_like(im, dtype=float)
    
    for i in range(im.shape[0]):
        if np.count_nonzero(mask[i]) > 0:
            out_mask[i,...,0] = skimage.filters.meijering(im[i,...,0], black_ridges=False)

    out_mask = out_mask*mask
    out_mask = out_mask / np.quantile(out_mask[out_mask>0], 0.99)
    
    mask = np.zeros_like(out_mask, dtype=int)
    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.9), np.count_nonzero(out_mask[slices] > 0.9), area)
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.5) / np.count_nonzero(labels[slices]==i))
        #print(np.quantile(out_mask[slices][labels[slices]==i], 0.9), np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10], np.count_nonzero(out_mask[slices][labels[slices]==i]>1))
        if np.partition(out_mask[slices][labels[slices]==i].flatten(), -10)[-10] < 0.7:
            mask[slices][labels[slices]==i] = 0
        else:
            mask[slices][labels[slices]==i] = 1
    return mask, timestep

def build_rotation_matrix(ax, ay, az, inverse=False):
    """Build a Euler rotation matrix.
    Rotation order is X, Y, Z (right-hand coordinate system).
    Expected vector is [x, y, z].
    Arguments:
        ax {float} -- rotation angle around X (radians)
        ay {float} -- rotation angle around Y (radians)
        az {float} -- rotation angle around Z (radians)
    Keyword Arguments:
        inverse {bool} -- Do inverse rotation (default: {False})
    Returns:
        [numpy.array] -- rotation matrix
    """

    if inverse:
        ax, ay, az = -ax, -ay, -az

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])

    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])

    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def make_ellipsoid_image(shape, center, radii, angle):
    """Draw a 3D binary image containing a 3D ellipsoid.
    Arguments:
        shape {list} -- image shape [z, y, x]
        center {list} -- center of the ellipsoid [x, y, z]
        radii {list} -- radii [x, y, z]
        angle {list} -- rotation angles [x, y, z]
    Raises:
        ValueError -- arguments are wrong
    Returns:
        [numpy.array] -- image with ellipsoid
    """

    if len(shape) != 3:
        raise ValueError('Only 3D ellipsoids are supported.')

    if not (len(center) == len(radii) == len(shape)):
        raise ValueError('Center, radii of ellipsoid and image shape have different dimensionality.')

    # Do opposite rotation since it is an axes rotation.
    angle = -1 * angle
    R = build_rotation_matrix(*angle)

    # Convert to numpy
    radii = np.array(radii)

    # Build a grid and get its points as a list
    xi = tuple(np.linspace(0, s-1, s) - np.floor(0.5 * s) for s in shape)

    # Build a list of points forming the grid
    xi = np.meshgrid(*xi, indexing='ij')
    points = np.array(list(zip(*np.vstack(map(np.ravel, xi)))))

    # Reorder coordinates to match XYZ order and rotate
    points = points[:, ::-1]
    points = np.dot(R, points.T).T

    # Find grid center and rotate
    grid_center = np.array(center) - 0.5*np.array(shape[::-1])
    grid_center = np.dot(R, grid_center)

    # Reorder coordinates back to ZYX to match the order of numpy array axis
    points = points[:, ::-1]
    grid_center = grid_center[::-1]
    radii = radii[::-1]

    # Draw the ellipsoid
    # dx**2 + dy**2 + dz**2 = r**2
    # dx**2 / r**2 + dy**2 / r**2 + dz**2 / r**2 = 1
    dR = (points - grid_center)**2
    dR = dR / radii**2
    # Sum dx, dy, dz / r**2
    nR = np.sum(dR, axis=1).reshape(shape)

    ell = (nR <= 1).astype(np.uint8)

    return ell
    
def step4_hessian(im, mask, timestep):
    out_mask = np.zeros_like(im, dtype=float)
    structure = np.zeros((1,3,3,3,1))
    #structure[1, :, :, :, 1] = 1
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    #labels, _no_features = ndimage.label(mask, structure=structure)

    #for i,slices in enumerate(ndimage.find_objects(labels), start=1):
    #    if (slices[1].stop-slices[1].start) < labels.shape[1]/2.0:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    #    if slices[3].start < labels.shape[3]//2 and slices[3].stop > labels.shape[3]//2:
    #        mask[slices][labels[slices]==i] = 0
    #        continue
    
    labels, _no_features = ndimage.label(mask, structure=structure)

    #print(im.shape, mask.shape)
    out_mask = np.zeros_like(im, dtype=float)
    
    for i in range(im.shape[0]):
        if np.count_nonzero(mask[i]) > 0:
            out_mask[i,...,0] = skimage.filters.hessian(im[i,...,0], black_ridges=False)

    out_mask = out_mask*mask
    out_mask = out_mask / np.quantile(out_mask[out_mask>0], 0.9)
    
    mask = np.zeros_like(out_mask, dtype=int)
    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.9), np.count_nonzero(out_mask[slices] > 0.9), area)
        #print(i, slices[0].start, slices[1], np.count_nonzero(out_mask[slices][labels[slices]==i] > 0.5) / np.count_nonzero(labels[slices]==i))
        #print(np.quantile(out_mask[slices][labels[slices]==i], 0.9), np.count_nonzero(out_mask[slices][labels[slices]==i]>1))
        if np.quantile(out_mask[slices][labels[slices]==i], 0.9) < 0.7:
            mask[slices][labels[slices]==i] = 0
        else:
            mask[slices][labels[slices]==i] = 1
    return mask, timestep

def step4(im, mask, timestep):
    structure = np.zeros((3,3,3,3,3))
    #structure[1, :, :, :, 1] = 1
    structure[1, :, 1, 1, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, 1, 1, :, 1] = 1
    labels, _no_features = ndimage.label(mask, structure=structure)

    structure = np.zeros((3,3,3,3,3))
    structure[1, 1, 1, :, 1] = 1
    structure[1, 1, :, 1, 1] = 1

    for i,slices in enumerate(ndimage.find_objects(labels), start=1):
        label_mask = labels[slices]==i
        subLabels, _ = ndimage.label(label_mask, structure=structure)
        czs = 0
        czs = {}
        for j, subSlices in enumerate(ndimage.find_objects(subLabels), start=1):
            area = (subSlices[2].stop-subSlices[2].start)*(subSlices[3].stop-subSlices[3].start)
            if area < 3:
                mask[slices][subSlices][subLabels[subSlices]==j] = 0
            else:
                square_area = max((subSlices[2].stop-subSlices[2].start),(subSlices[3].stop-subSlices[3].start))**2
                count = np.count_nonzero(subLabels[subSlices]==j)
                cz = count / square_area
                if subSlices[1].start not in czs:
                    czs[subSlices[1].start] = [cz]
                else:
                    czs[subSlices[1].start].append(cz)

        #if slices[0].start==7:
            #print(czs)

        czs1 = np.array([np.average(czs[k]) for k in sorted(czs.keys())])
        czs1 = np.count_nonzero(np.bitwise_and(czs1 > 0.5, czs1 < 0.8))
        #print(slices[2].start, slices[3].start, czs1/(slices[1].stop-slices[1].start), [(k, czs[k]) for k in sorted(czs.keys())])

        if czs1 < (slices[1].stop-slices[1].start)/4.0:
            mask[slices][label_mask] = 0

    return mask, timestep

def step5(im, mask, timestep, early_timestep, early_mask):
    timestep_mask, timestep = step3(im, mask, timestep)

    use_early = (early_timestep > 5) and (np.count_nonzero(early_mask) > 0) and ((early_timestep < timestep) or (early_timestep == timestep and np.count_nonzero(early_mask) > np.count_nonzero(timestep_mask)))
    print(early_timestep, timestep, use_early)

    if use_early or np.count_nonzero(timestep_mask) == 0:
        timestep_mask = np.zeros_like(mask, dtype=int)
        timestep_mask[early_timestep] = mask[early_timestep]
        return timestep_mask, early_timestep
    else:
        return timestep_mask, timestep

def step6(im, timestep_mask, timestep):
    structure = np.zeros((3,3,3,3,3))
    #structure[1, :, :, :, 1] = 1
    structure[1, :, 1, 1, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, 1, 1, :, 1] = 1
    filter_mask = filter_gvt(im, timestep_mask, timestep, structure)
    
    #nz = np.count_nonzero(filter_mask==1, axis=(1,2,3,4))
    #nz = np.array(range(nz.shape[0]))[nz!=0]
    #timestep = int(round(np.average(nz)))

    #filter_mask, timestep = step3(im, filter_mask, timestep)

    #filter_mask = np.max(filter_mask, axis=0)

    #out_mask = np.zeros_like(timestep_mask)
    #out_mask[timestep] = filter_mask

    return filter_mask, timestep

def step6_(im, timestep_mask, timestep):
    structure = np.zeros((1,3,3,3,1))
    #structure[1, :, :, :, 1] = 1
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    filter_mask = filter_gvt_pixel(im, timestep_mask, timestep, structure)
    
    return filter_mask, timestep

    nz = np.count_nonzero(filter_mask==1, axis=(1,2,3,4))
    nz = np.array(range(nz.shape[0]))[nz!=0]
    timestep = int(round(np.average(nz)))

    #_filter_mask, timestep = step3(im, filter_mask==1, timestep)

    filter_mask = np.max(filter_mask, axis=0)

    out_mask = np.zeros_like(timestep_mask)
    out_mask[timestep] = filter_mask

    return out_mask, timestep

def step7(im, mask, timestep):
    structure = np.zeros((1,3,3,3,1))
    #structure[1, :, :, :, 1] = 1
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    small_mask = scipy.ndimage.morphology.binary_erosion(mask, structure)
    small_mask = scipy.ndimage.morphology.binary_dilation(small_mask, structure)
    if np.count_nonzero(small_mask) > 0.7*np.count_nonzero(mask):
        mask = small_mask
    return np.array(mask,dtype=int), timestep

def step8_(im, filter_mask, timestep):
    peaks = np.max(im, axis=0)
    peak_mask = im==peaks
    mask = filter_mask*peak_mask
    structure = np.zeros((1,3,3,3,1))
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    mask = np.array(scipy.ndimage.morphology.binary_fill_holes(mask, structure=structure), dtype=int)
    return mask, timestep

def step8(im, filter_mask, timestep):
    structure = np.zeros((3,3,3,3))
    #structure[:, :, :, 1] = 1
    structure[:, 1, 1, 1] = 1
    structure[1, :, 1, 1] = 1
    structure[1, 1, :, 1] = 1
    grow_mask = np.zeros_like(im)

    for timestep in range(1,filter_mask.shape[0]-1):
        #imt = im[timestep]*0.75 + im[timestep+1]*0.25
        imt = im[timestep]
        if np.count_nonzero(filter_mask[timestep]) == 0:
            continue
        labels, _no_features = ndimage.label(filter_mask[timestep], structure=structure)
        used_mask = np.array(filter_mask[timestep], dtype=int)
        
        structure = np.zeros((3,3,3,3))
        #structure[:, :, :, 1] = 1
        structure[:, 1, 1, 1] = 1
        structure[1, :, 1, 1] = 1
        structure[1, 1, :, 1] = 1
        
        #for _ in range(6):
        while np.count_nonzero(imt*used_mask) > 0:
            img = imt*used_mask
            seed = np.unravel_index(np.argmax(img), img.shape)
            if img[seed] == 0:
                break
            
            object_index = labels[seed]
            object_voxel = imt[labels==object_index]
            object_voxel = object_voxel[object_voxel>0]
            object_voxel = object_voxel[object_voxel>=np.quantile(object_voxel, 0.2)]
            avg = np.average(object_voxel)
            std = np.std(object_voxel)
            t = avg-std
            #print(avg, std, t, labels[seed])
            if t < 0:
                used_mask[labels==labels[seed]] = 0
                continue

            new_grow = grow(imt, seed, t)
            grow_mask[timestep] = grow_mask[timestep] + new_grow
            new_grow = np.array(scipy.ndimage.morphology.binary_dilation(new_grow, structure, iterations=2), dtype=int)
            used_mask = used_mask - new_grow
            used_mask[used_mask<0] = 0
            used_mask[labels==labels[seed]] = 0

    grow_mask[grow_mask>1] = 1

    return grow_mask, timestep

def step9(im, grow_mask, timestep):
    structure = np.zeros((3,3,3,3,3))
    structure[1, :, 1, 1, 1] = 1
    structure[1, 1, :, 1, 1] = 1
    structure[1, 1, 1, :, 1] = 1
    labels, no_features = scipy.ndimage.label(grow_mask, structure=structure)
    
    sizes = []
    for i in range(1, no_features+1):
        #if np.count_nonzero(labels[:,:,:labels.shape[2]//2,:labels.shape[3]//2,:]==i) > 0:
        if np.count_nonzero(labels[:,:,:,:labels.shape[3]//2,:]==i) > 0:
        #if np.count_nonzero(labels[:,:,:labels.shape[3]//2,:,:]==i) > 0:
            sizes.append(np.count_nonzero(labels==i))
        else:
            sizes.append(0)
    if np.count_nonzero(sizes) > 0:
        i1 = np.argsort(sizes)[-1]+1
    else:
        i1 = -1

    #print(sizes, i1)
    
    sizes = []
    for i in range(1, no_features+1):
        if i == i1:
            sizes.append(0)
            continue
        #if np.count_nonzero(labels[:,:,:labels.shape[2]//2,labels.shape[3]//2:,:]==i) > 0:
        if np.count_nonzero(labels[:,:,:,labels.shape[3]//2:,:]==i) > 0:
        #if np.count_nonzero(labels[:,:,labels.shape[3]//2:,:,:]==i) > 0:
            sizes.append(np.count_nonzero(labels==i))
        else:
            sizes.append(0)
    if np.count_nonzero(sizes) > 0:
        i2 = np.argsort(sizes)[-1]+1
    else:
        i2 = -1

    #print(sizes, i2)

    grow_mask = np.bitwise_or(labels==i1, labels==i2)

    return np.array(grow_mask, dtype=int), timestep

def step10(im, mask, timestep):
    smooth_mask = ndimage.gaussian_filter(mask, (0, 1, 1, 1, 0), mode='nearest')
    #return smooth_mask, timestep
    return np.array(smooth_mask>0.3, dtype=mask.dtype), timestep

def step9_(im, grow_mask, timestep):
    structure = np.zeros((1,3,3,3,1))
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1
    labels, no_features = scipy.ndimage.label(grow_mask, structure=structure)
    
    size1 = 0
    size2 = 0
    sizes = []
    for i in range(1, no_features+1):
        sizes.append(np.count_nonzero(labels[:,:,:,:labels.shape[3]//2,:]==i))
    if np.count_nonzero(sizes) > 0:
        i1 = np.argsort(sizes)[-1]+1
        size1 = sizes[i1-1]
    else:
        i1 = -1
    
    sizes = []
    for i in range(1, no_features+1):
        sizes.append(np.count_nonzero(labels[:,:,:,grow_mask.shape[3]//2:,:]==i))
    if np.count_nonzero(sizes) > 0:
        i2 = np.argsort(sizes)[-1]+1
        size2 = sizes[i2-1]
    else:
        i2 = -1

    if size2 > 0 and size1/size2 < 0.5:
        i1 = -1
    elif size1 > 0 and size2/size1 < 0.5:
        i2 = -1

    grow_mask = np.bitwise_or(labels==i1, labels==i2)

    return np.array(grow_mask, dtype=int), timestep

def step10_(im, grow_mask, timestep):
    
    structure = np.zeros((1,3,3,3,1))
    structure[0, :, :, 1, 0] = 1
    structure[0, 1, :, :, 0] = 1
    structure[0, :, 1, :, 0] = 1

    grow_mask = scipy.ndimage.morphology.binary_dilation(grow_mask, structure=structure)
        
    grow_mask = np.array(scipy.ndimage.morphology.binary_fill_holes(grow_mask, structure=structure), dtype=int)

    return grow_mask, timestep


def middle(im, label, timestep):
    mid = np.zeros_like(label)
    mid_index = label.shape[1]//2
    #print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, timestep

def middlep(im, label, timestep):
    mid = np.zeros_like(label)
    mid_index = label.shape[1]//2 + 2
    #print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, timestep

def middlem(im, label, timestep):
    mid = np.zeros_like(label)
    mid_index = label.shape[1]//2 - 2
    #print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, timestep

def upper(im, label, timestep):
    mid = np.zeros_like(label)
    mid_index = 3*label.shape[1]//4
    #print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, timestep

def lower(im, label, timestep):
    mid = np.zeros_like(label)
    mid_index = label.shape[1]//4
    #print(label.shape, mid.shape, mid_index)
    mid[:,mid_index] = label[:,mid_index]

    return mid, timestep

def filter_gvt_pixel(im, timestep_mask, timestep, structure):
    labels, no_features = ndimage.label(timestep_mask, structure=structure)

    filter_mask = np.zeros_like(timestep_mask)

    if no_features>0:
        alphas = []
        for i, slices in enumerate(ndimage.find_objects(labels), start=1):
            roi = labels[slices[0].start]==i
            for index, in_roi in np.ndenumerate(roi):
                if not in_roi:
                     continue
                p = im[:,index[0], index[1], index[2], index[3]]
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
                    filter_mask[tmax,index[0], index[1], index[2], index[3]] = 0
                else:
                    filter_mask[tmax,index[0], index[1], index[2], index[3]] = 1
    
    return np.array(filter_mask, dtype=int)

def filter_gvt(im, timestep_mask, timestep, structure):
    labels, no_features = ndimage.label(timestep_mask, structure=structure)
    
    filter_mask = np.zeros_like(timestep_mask, dtype=int)

    if no_features>0:
        alphas = []
        for i, slices in enumerate(ndimage.find_objects(labels), start=1):
            roi = labels[slices[0].start]==i
            p = np.average(im[:,roi], axis=1)
            tmax = np.argmax(p)
            if tmax > 15:
                continue
            prev = p[tmax]
            t0 = 0
            for tmin in range(tmax,0,-1):
                if p[tmin] <= prev:
                    prev = p[tmin]
                else:
                    t0 = tmin+1
                    break
            if tmax <= t0:
                tmax = np.argmax(p[t0+1:]) + t0 + 1
            #p = scipy.ndimage.gaussian_filter1d(p, 0.7)
            alpha = [np.log(1e-30 + p[t]/p[tmax])/(1 + np.log((t - t0)/(tmax-t0)) - (t-t0)/(tmax-t0) + 1e-30) for t in range(t0+1, p.shape[0])]
            alpha = np.average(alpha)
            k = p[-1] / np.max(p)
            k = np.log(k+1e-10)
            a_denom = (1 + np.log((p.shape[0]-1 - t0)/(tmax-t0)) - (p.shape[0]-1-t0)/(tmax-t0))
            lowerA = (k / (a_denom + 1e-10))
            alphas.append(alpha)
            if alpha > tmax-t0 or alpha < lowerA:
                filter_mask[tmax, roi] += 0
            else:
                filter_mask[tmax, roi] += 1


    labels, no_features = ndimage.label(filter_mask, structure=structure)
    thr = np.max(filter_mask[:,:,:,labels.shape[3]//2:,:])*0.5
    for i, slices in enumerate(ndimage.find_objects(labels[:,:,:,labels.shape[3]//2:,:]), start=1):
        if np.size(filter_mask[:,:,:,labels.shape[3]//2:,:][slices][labels[:,:,:,labels.shape[3]//2:,:][slices]==i]) > 0:
            if np.max(filter_mask[:,:,:,labels.shape[3]//2:,:][slices][labels[:,:,:,labels.shape[3]//2:,:][slices]==i]) <= thr:
                filter_mask[:,:,:,labels.shape[3]//2:,:][slices][labels[slices]==i] = 0
    
    thr = np.max(filter_mask[:,:,:,:labels.shape[3]//2,:])*0.5
    for i, slices in enumerate(ndimage.find_objects(labels[:,:,:,:labels.shape[3]//2,:]), start=1):
        if np.size(filter_mask[:,:,:,:labels.shape[3]//2,:][slices][labels[:,:,:,:labels.shape[3]//2,:][slices]==i]) > 0:
            if np.max(filter_mask[:,:,:,:labels.shape[3]//2,:][slices][labels[:,:,:,:labels.shape[3]//2,:][slices]==i]) <= thr:
                filter_mask[:,:,:,:labels.shape[3]//2,:][slices][labels[slices]==i] = 0
    
    return filter_mask
    #return np.array(filter_mask>=(max(1,np.max(filter_mask)/3)), dtype=int)
    
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

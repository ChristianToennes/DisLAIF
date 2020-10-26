import numpy as np
import scipy.optimize
import scipy.special
import time
import SimpleITK as sitk
import os
import csv
import argparse
import pydicom
import math

#np.seterr(divide="raise", over="raise", under="raise", invalid="raise")
np.seterr(invalid="ignore")

def get_t(path, name):
    path = os.path.join(os.path.normpath(path), name)
    name = path.rsplit("\\", maxsplit=1)[1]
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

    return np.array(times)

def functionalAIF(A1, A2, T1, T2, sig1, sig2, alpha, beta, s, tau, t):
    aif = A1/(sig1*math.sqrt(2*math.pi))*np.exp(-(t-T1)**2/(2*sig1*sig1)) + A2/(sig2*math.sqrt(2*math.pi))*np.exp(-(t-T2)**2/(2*sig2*sig2)) + alpha * np.exp(-beta*t)/(1+np.exp(-s*(t-tau)))
    return aif

def parker2006(t, aif):
    A1 = 0.809; A2 = 0.330; T1 = 0.17046; T2 = 0.365; σ1 = 0.0563; σ2 = 0.132; α = 1.05; β = 0.1685; s = 38.078; τ = 0.483
    i = max(1, np.argmax(aif)-2)
    T1 = T1 + t[i]
    T2 = T2 + t[i]
    τ = τ + t[i]
    A1 = A1 * aif[i+2]/6
    A2 = A2 * aif[i+2]/6
    α = α * aif[i+2]/6
    #print(aif)
    #print(i, t[i], aif[i+2]/6)
    return functionalAIF(A1, A2, T1, T2, σ1, σ2, α, β, s, τ, t)

def get_aif(path, name, hct=0.45, aifs=None):
    files = {}
    for entry in os.scandir(path):
        parts = entry.name.split('.')
        if entry.is_file() and parts[0] == name and parts[-1] in ("mhd", "nrrd") and (parts[2] in ("target", "target_dilated", "target_eroded", "target_middle", "target_middlep", "target_middlem", "target_upper", "target_lower", "target_left", "target_up", "target_right", "target_down", "target_fore", "target_back", "target_rightSide", "target_leftSide", "OldDisLAIF", "DisLAIF", "shi2013", "chen2008", "parker2003", "noop", "max05", "gamma_variate", "dante") or parts[2].startswith("middle") ):
            if aifs and parts[2] not in ("target", "noop") and parts[2] not in aifs:
                continue
            if parts[0] not in files:
                files[parts[0]] = {}
            files[parts[0]][parts[2]] = entry.path
    
    for key in files:
        if "target" not in files[key]:
            continue

        Image = sitk.ReadImage(files[key]["noop"])
        image = sitk.GetArrayFromImage(Image)
        if image.shape[0] > 69:
            image = image[1::2]
        #print(key, image.shape)
        image = image.reshape(image.shape + (1,))
        for algo in files[key]:
            if algo == "noop":
                continue
            Label = sitk.ReadImage(files[key][algo])
            label = sitk.GetArrayFromImage(Label)
            #print(key, algo, label.shape)
            label = label.reshape(label.shape + (1,))
            if len(label.shape) == 4:
                label = label.reshape((1,) + label.shape)
            
            t = np.argmax(np.count_nonzero(label, axis=(1,2,3,4)))
            label = label[t].reshape((1,) + label.shape[1:])

            aif = image*label
            num = np.sum(aif, axis=(1,2,3,4))
            denom = np.count_nonzero(label, axis=(1,2,3,4))
            if denom.shape[0] == 1:
                if denom[0] == 0:
                    num = np.zeros_like(num)
                    denom = 1
            else:
                num[denom==0] = 0
                denom[denom==0] = 1

            #print(key, algo, image.shape, label.shape, num.shape, denom.shape)
            aif = num/denom
            #print(aif, num, denom, np.argmax(aif))
            avg = np.average(aif[:max(1,np.argmax(aif)-2)])
            aif = aif-avg
            #print(aif, avg)
            #aif = aif / avg
            yield aif, algo

            if algo in ("target", "DisLAIF", "parker2003"):
                t = get_t('..\\Data\\input_', name)
                yield parker2006(t, aif), algo + "-parker2006"

def get_true_tumor_c(path, count=None, pats=None):
    files = {}
    for entry in os.scandir(path):
        #if not ("40418168" in entry.name or "42175470" in entry.name or "42268224" in entry.name or "42351313" in entry.name):
        #    #print(name)
        #    continue
        parts = entry.name.split('.')
        if entry.is_file() and parts[-1] in ("mhd", "nrrd"):
            if pats and parts[0] not in ("label", "dce") and parts[0] not in pats:
                continue
            if parts[0] not in files:
                files[parts[0]] = {}
            files[parts[0]][parts[1]] = entry.path
    
    for key in files:
        if count != None:
            if count == 0:
                break
            else:
                count = count-1
        if "label" not in files[key] or "dce" not in files[key]:
            continue

        Image = sitk.ReadImage(files[key]["dce"])
        image = sitk.GetArrayFromImage(Image)
        if image.shape[0] > 69:
            image = image[1::2]
        Label = sitk.ReadImage(files[key]["label"])
        label = np.array(sitk.GetArrayFromImage(Label)==1, dtype=int)
        if label.shape[0] > 69:
            label = label[1::2]
        if len(label.shape) == 4:
            t = np.argmax(np.count_nonzero(label, axis=(1,2,3)))
            label = label[t:t+1]

        aif = image*label
        #print(image.shape, label.shape, aif.shape)
        num = np.sum(aif, axis=(1,2,3))
        denom = np.count_nonzero(label)
        #print(image.shape, label.shape, aif.shape, num.shape, denom)
        if denom == 0:
            num = np.zeros_like(num)
            denom = 1
        
        yield num/denom, image, key, -1

        Image = sitk.ReadImage(files[key]["dce"])
        image = sitk.GetArrayFromImage(Image)
        if image.shape[0] > 69:
            image = image[1::2]
        Label = sitk.ReadImage(files[key]["label"])
        label = np.array(sitk.GetArrayFromImage(Label)==1, dtype=int)
        if label.shape[0] > 69:
            label = label[1::2]
        if len(label.shape) == 4:
            t = np.argmax(np.count_nonzero(label, axis=(1,2,3)))
            label = label[t:t+1]

        #print(label.shape)
        zindex = np.count_nonzero(label, axis=(1,2,3))
        #print(zindex)
        zindex = np.argmax(zindex)
        zslice = label[zindex]
        label = np.zeros_like(label)
        label[zindex] = zslice
        #print(zindex, zslice.shape)

        aif = image*label
        #print(image.shape, label.shape, aif.shape)
        num = np.sum(aif, axis=(1,2,3))
        denom = np.count_nonzero(label)
        #print(image.shape, label.shape, aif.shape, num.shape, denom)
        if denom == 0:
            num = np.zeros_like(num)
            denom = 1
        
        yield num/denom, image, key, zindex

def get_all_tumor_c(path, image, pat_name, is_slice, true_c, count=None, nets=None):
    files = {}
    for entry in os.scandir(path):
        if not entry.is_file():
            continue
        if not entry.name.endswith("mhd") and not entry.name.endswith("nrrd") :
            continue
        #if not (entry.name.startswith("40418168") or entry.name.startswith("42175470") or entry.name.startswith("42268224") or entry.name.startswith("42351313")):
        #    continue
        if not entry.name.startswith(pat_name):
            continue
        parts = entry.name.split('.')
        s = parts[1]
        if len(parts[0].split()) != 5: continue
        pat, net, fold, loss, p = parts[0].split('-')
        if nets and net not in nets:
            continue
        name = net + "-" + fold + "-" + loss + "-" + p + "-" + s
        files[name] = entry.path

    yield true_c, "target"

    for key in files.keys():
        if count != None:
            if count == 0:
                break
            else:
                count = count-1
    
        Label = sitk.ReadImage(files[key])
        label = np.array(sitk.GetArrayFromImage(Label)==1,dtype=int)
        if label.shape[0] > 69:
            label = label[1::2]
    
        if is_slice >= 0:
            if is_slice >= label.shape[0]:
                if len(label.shape) == 4:
                    t = np.argmax(np.count_nonzero(label, axis=(1,2,3)))
                    label = label[t]
            zslice = label[is_slice]
            label = np.zeros_like(label)
            label[is_slice] = zslice

        aif = image*label
        num = np.sum(aif, axis=(1,2,3))
        denom = np.count_nonzero(label)
        if denom == 0:
            num = np.zeros_like(num)
            denom = 1
        
        yield num/denom, key

def expConvolute(m, k, time, aif, calcDer=False):

     #DT = T[1:n-1]-T[0:n-2]
     #DA = A[1:n-1]-A[0:n-2]
    #deltaT = np.zeros(m-1)
    #deltaA = np.zeros(m-1)

    #for i in range(0,m-1):
    #    deltaT[i] = time[i+1] - time[i]
    #    deltaA[i] = aif[i+1] - aif[i]

    deltaT = time[1:-1]-time[:-2]
    deltaA = aif[1:]-aif[:-1]

    
    #Z = l*DT
    #E = exp(-Z)
    #E0 = 1-E
    #E1 = Z-E0
     
    #z = np.zeros(m-1)
    #e = np.zeros(m-1)
    #e0 = np.zeros(m-1)
    #e1 = np.zeros(m-1)
    
    #for i in range(0,m-1):# (i = 0; i < m-1; i++) {
    #    z[i] = k * deltaT[i]
    #    e[i] = np.exp(z[i]*(-1))
    #    e0[i] = 1 - e[i]
    #    e1[i] = z[i] - e0[i]
    z = k*deltaT

    z[z>=700] = 700
    z[z<1e-15] = 1e-15

    e = np.exp(-z)
    e0 = 1-e
    e1 = z-e0
    
    #Il = (A[0:n-2]*E0 + DA*E1/Z)/l
    #il = np.zeros(m-1)
    #for i in range(0,m-1):# (i = 0; i < m-1; i++)
    #    il[i] = ( (aif[i] * e0[i]) + (deltaA[i] * e1[i])/z[i] )/k
    
    il = ( (aif[:-1]*e0) + (deltaA*e1)/z ) / k
    
    #Y = dblarr(n)
    y = np.zeros(m)
    #for i in range(0,m): #(i = 0; i < m; i++)
    #    y[i] = 0
    
    #for i=0L,n-2 do Y[i+1] = E[i]*Y[i] + Il[i]

    for i in range(0,m-1):# (i = 0; i < m-1; i++)
        y[i+1] = e[i]*y[i] + il[i]
    
    if not calcDer:
        return y
    #E2 = Z^2-2*E1
    #e2 = np.zeros(m-1)
    #for i in range(0,m-1):#(i = 0; i < m-1; i++)
    #    e2[i] = pow(z[i], 2) - 2*e1[i]
    e2 = z**2 - 2*e1
    
    # DIl = -DT*Il + (A[0:n-2]*E1 + DA*E2/Z )/(l^2)
    # DIl = -E*DT*Y + DIl
    #dil = np.zeros(m-1)
    #for i in range(0,m-1):# (i = 0; i < m-1; i++)
    #    dil[i] = deltaT[i]*(-1)*il[i] + ( aif[i]*e1[i] + deltaA[i]*e2[i]/z[i]) / pow(k, 2)
    
    dil = (aif*e1 + deltaA*e2/z) / (k**2) - deltaT*il

    #for i in range(0,m-1):#(i = 0; i < m-1; i++)
    #    dil[i] = e[i]*(-1)*deltaT[i]*y[i] + dil[i]

    dil = dil - e*deltaT*y
    

    #DY = dblarr(n)
    der = np.zeros(m)
    #for i in range(0,m):# (i = 0; i < m; i++)
    #    der[i] = 0
    
    #for i=0L,n-2 do DY[i+1] = E[i]*DY[i] + DIl[i]

    for i in range(0,m-1):# (i = 0; i < m-1; i++)
        der[i+1] = e[i]*der[i] + dil[i]

    return y, der

def DR2CXM(t, p, aif):
    m = len(t)
    a = p[2]-p[2]*p[3]+p[3]
    b = p[2]*(1-p[2])*p[3]*(1-p[3])
    
    ts = np.sqrt(1-4*b/pow(a, 2))
    ta = 0.5*a/p[3]
    
    tp = ta*(1+ts)
    tm = ta*(1-ts)
    
    kp = p[1]/(p[0]*tm)
    km = p[1]/(p[0]*tp)
    am = (1-tm)/(tp-tm)
    
    cp, cpd = expConvolute(m-1, kp, t, aif, True)
    cm, cmd = expConvolute(m-1, km, t, aif, True)
    
    dy = np.zeros(m-1)
    fit = np.zeros(m-1)
    for i in range(0,m-1):
        dy[i] = p[1]*(1-am)*cp[i] + p[1]*am*cm[i]
        fit[i] = dy[i]
        dy[i] = (aif[i] - dy[i])*1.0
    
    return dy


def x0s_2CXM():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,4)
    ves = np.linspace(0.01,0.99,4)
    #fps = np.linspace(1,1000,2)
    fps = np.array([50, 1000])
    es = np.linspace(0.01,0.99,3)
    
    x0s = np.array(np.meshgrid(vs, fps, ves, es)).T.reshape(-1, 4)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, np.inf), (0.001,1), (0.001,1)]

def R2CXM(t, p, aif):
    # x = ve+vp, Fp, ve / (ve+vp), E
    m = t.shape[0]
    a = p[2]-p[2]*p[3]+p[3]
    b = p[2]*(1-p[2])*p[3]*(1-p[3])
    
    ts = np.sqrt(1-4*b/pow(a, 2))
    ta = 0.5*a/p[3]
    
    tp = ta*(1+ts)
    tm = ta*(1-ts)
    
    kp = p[1]/(p[0]*tm)
    km = p[1]/(p[0]*tp)
    am = (1-tm)/(tp-tm)
    
    cp = expConvolute(m-1, kp, t, aif)
    cm = expConvolute(m-1, km, t, aif)
    
    dy = p[1]*(1-am)*cp + p[1]*am*cm
    
    return dy

def out_2CXM(x):
    x, chis, x0s = x
    #print(x, x.T)
    (av, aFp, ave, aE) = x.T
    avp = (1-ave) * av
    ave = ave * av
    amtt = (avp+ave)/aFp

    ak = aFp*aE
    aPS = aFp*aE/(1-aE)
    aFp = aFp*60
    
    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def _R2CXM(t, x, aif):
    T = (x[1]*0.01+x[3]*0.01)/(x[0])
    Te = (0.01*x[3]*(1-x[2]*0.01))/(x[0]*x[2]*0.01)
    Tc = (x[1]*0.01)/(x[0])

    sqrt = (T+Te)**2 - 4*Tc*Te

    sigmaP = (T+Te+ np.sqrt(sqrt) ) / (2*Tc*Te+1e-16)
    eSigmaP = t*sigmaP

    sigmaN = (T+Te- np.sqrt(sqrt) ) / (2*Tc*Te+1e-16)
    eSigmaN = t*sigmaN

    if eSigmaN < 1e-18:
        eSigmaN = 1
    elif eSigmaN > 700:
        eSigmaN = 0
    else:
        eSigmaN = np.exp(-eSigmaN)
    if eSigmaP < 1e-18:
        eSigmaP = 1
    elif eSigmaP > 700:
        eSigmaP = 0
    else:
        eSigmaP = np.exp(-eSigmaP)

    r = ((T*sigmaP-1)*sigmaN*eSigmaN+(1-T*sigmaN)*sigmaP*eSigmaP) / (sigmaP-sigmaN)

    res = np.convolve(aif, x[1]*r, mode='same')
    return res

def x0s_DP():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,10)
    ves = np.linspace(0.01,0.99,10)
    #fps = np.linspace(1,1000,2)
    fps = np.array([50, 1000])
    es = np.linspace(0.01,0.99,5)
    
    x0s = np.array(np.meshgrid(vs, fps, ves, es)).T.reshape(-1, 4)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, np.inf), (0.001,1), (0.001,1)]
    
def RDP(t, x, aif):
    # x = ve+vp, Fp, ve / (ve+vp), E
    # Tc = vp / Fp
    # T = (vp+ve) / Fp
    # PS = -Fp * log(1-E)
    # Te = 
    #Tc = (x[1])/(x[0])
    #T = (x[1]+x[3])/(x[0])
    #PS = -x[0]*np.log(1-x[2])
    #Te = (x[3])/(PS)

    Tc = 60* (1-x[2]*x[0]) / x[1]
    T = 60* x[0] / x[1]
    PS = -x[1]*np.log(1-x[3])
    Te = 60* x[2]*x[0] / PS

    alpha = PS/x[1]
    integral = np.zeros_like(aif)
    for i in range(0, integral.shape[0]):
        tau = np.arange(1, t[i]-Tc, 1)
        bessel = scipy.special.i1e(2*np.sqrt(alpha*tau/Te))
        eTau_Te = (tau)/(Te)
        #eTau_Te[eTau_Te>700] = 700
        #eTau_Te[eTau_Te < 1e-18] = 1e-18
        integral[i] = np.sum(np.exp(-eTau_Te) * np.sqrt(alpha/(tau*Te)) * bessel )
    #alpha[alpha>700] = 700
    eAlpha = np.exp(-alpha)
    
    r = 1-eAlpha*(1+integral)
    r[t[:-1]<Tc] = 1

    res = np.convolve(aif, x[1]*r, mode='same')
    return res

def out_DP(x):
    x, chis, x0s = x
    (av, aFp, ave, aE) = x.T

    avp = (1-ave) * av
    ave = ave * av
    amtt = (avp+ave)/aFp
    ak = aFp*aE
    aPS = -aFp*np.log(1-aE)
    aFp = aFp*60

    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def x0s_AATH():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,10)
    ves = np.linspace(0.01,0.99,10)
    #fps = np.linspace(1,1000,2)
    fps = np.array([50, 1000])
    es = np.linspace(0.01,0.99,5)
    
    x0s = np.array(np.meshgrid(vs, fps, ves, es)).T.reshape(-1, 4)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, np.inf), (0.001,1), (0.001,1)]

def RAATH(t, x, aif):
    # x = ve+vp, Fp, ve / (ve+vp), E
    Tc = (1-x[2]*x[0]) / x[1]
    r = np.ones_like(aif)
    r[t[:-1]==Tc] = 0
    
    # Ktrans = E * Fp
    # kep = Ktrans / ve
    et_Tc_kep = (t[:-1]-Tc) * x[3]*x[1]/(x[0]*x[2]) 
    et_Tc_kep[et_Tc_kep>700] = 700
    r = x[3]*np.exp(-et_Tc_kep)
    
    res = np.convolve(aif, x[1]*r, mode='same')
    return res

def out_AATH(x):
    x, chis, x0s = x
    (av, aFp, ave, aE) = x.T

    avp = (1-ave) * av
    ave = ave * av
    amtt = (avp+ave)/aFp
    ak = aFp*aE
    aPS = -aFp*np.log(1-aE)
    aFp = aFp*60

    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def x0s_CTU():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,4)
    fps = np.array([50, 1000])
    es = np.linspace(0.01,0.99,3)
    
    x0s = np.array(np.meshgrid(vs, fps, es)).T.reshape(-1, 3)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, np.inf), (0.001,1)]

def RCTU(t, x, aif):
    # x = vp, Fp, E
    # PS = E*Fp / (1-E)
    # Tp = vp / (Fp + PS)
    PS = x[2]*x[1] / (1-x[2])
    Tp = x[0]/(x[1] + PS)
    t_Tp = t[:-1]/Tp
    
    #t_Tp[t_Tp>700] = 700

    r = (1-x[2])*np.exp(-t_Tp)+x[2]

    res = np.convolve(aif, x[1]*r, mode='same')
    return res

def out_CTU(x):
    x, chis, x0s = x
    (avp, aFp, aE) = x.T

    ave = np.zeros_like(avp)-1
    amtt = avp/aFp
    ak = aFp*aE
    aPS = aFp*aE/(1-aE)
    aFp = aFp*60
    
    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def x0s_PTU():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,10)
    fps = np.array([50, 1000])
    es = np.linspace(0.01,0.99,5)
    
    x0s = np.array(np.meshgrid(vs, fps, es)).T.reshape(-1, 3)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, np.inf), (0.001,1)]

def RPTU(t, x, aif):
    # x = vp, Fp, E
    # Tc = vp / Fp
    Tc = x[0]/(x[1])
    r = np.ones_like(aif)
    r[Tc>t[:-1]] = x[2]
    
    res = np.convolve(aif, x[1]*r, mode='same')
    return res

def out_PTU(x):
    x, chis, x0s = x
    #print(x, x.T)
    (avp, aFp, aE) = x.T

    ave = np.zeros_like(avp)-1
    amtt = avp/aFp
    ak = aFp*aE
    aPS = -aFp*np.log(1-aE)
    aFp = aFp*60

    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def x0s_Tofts():
    np.random.seed(42)
    x0s = []
    vs = np.linspace(0.01,0.99,10)
    fps = np.array([0.1, 1, 10, 20])
    es = np.linspace(0.01,0.99,5)
    
    x0s = np.array(np.meshgrid(vs, es, fps)).T.reshape(-1, 3)
    #print(x0s.shape)
    return x0s, [(0.001, 1), (0.001, 1), (0.001,np.inf)]

def RTofts(t, x, aif):
    # x = ve+vp, ve / (ve+vp), Ktrans

    vp = x[0]*(1.0-x[1])
    ve = x[0]*x[1]
    r = x[2]*np.exp(-t[:-1]*x[2]/ve)
    r[0] += vp

    res = np.convolve(aif, r, mode='same')
    return res

def out_Tofts(x):
    x, chis, x0s = x
    (av, ave, ak) = x.T

    avp = (1-ave) * av
    ave = ave * av
    amtt = np.zeros_like(avp)-1
    aFp = np.zeros_like(avp)-1
    aE = np.zeros_like(avp)-1
    ak = ak*60
    aPS = ak

    return aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s

def chi(x, aif, c, t, Rfun):
    # x = ve+vp, Fp, ve / (ve+vp), E
    r = Rfun(t, x, aif)
    res = np.sum((c-r)**2)
    return res

def _chi(x, aif, c, t, Rfun):
    try:
        #if x[0] < 0 or x[1] < 0 or x[2] < 0 or x[3] < 0 or x[1] > 1 or x[2] > 1 or x[3] > 1 or (x[1]+x[3]) > 1:
        #    return 10000
        R = np.vectorize(lambda t: Rfun(t, x, aif))#(np.flip(t[:-1]))
        #t_weights = (t[1:]-t[:-1])/t[-1]
        #conv = aif*x[0]*R*t_weights
        conv = np.convolve(aif,x[0]*R(t[:-1]), mode='same')
        #if conv.shape[0] != c.shape[0]:
        #    print(conv.shape, aif.shape, R.shape, c.shape)
        res = np.sum((c-conv)**2)
        if res > 1e200 or np.isnan(res).any() or np.isinf(res).any():
            print(res)
            res = 1e200
    except Exception:
        raise
        res = 1e200
    return res

def optimize(aif, c, t, method, Rfun, get_x0s):
    # x = Fp, vp, PS, ve
    x0s, bounds = get_x0s()
    if method in ("CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
        jac = "2-point"
    elif method in ("trust-constr",):
        jac = "2-point"
    else:
        jac = None
    if method in ("Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
        hess = "2-point"
    elif method in ("trust-constr",):
        hess = scipy.optimize.BFGS()
    else:
        hess = None
    if method in ("L-BFGS-B", "TNC", "SLSQP", "trust-constr"):
        #bounds = [(0.001, np.inf), (0.001,1), (0.001,1), (0.001,1)]
        pass
    else:
        bounds = None
    if method in ("COBYLA","SLSQP"):
        constraints = ({'type': 'ineq', 'fun': lambda x: round(x[0])+abs(round(x[0]))},
                       {'type': 'ineq', 'fun': lambda x: round(x[1])+abs(round(x[1]))},
                       {'type': 'ineq', 'fun': lambda x: round(x[2])+abs(round(x[2]))},
                       {'type': 'ineq', 'fun': lambda x: round(x[3])+abs(round(x[3]))},
                       {'type': 'ineq', 'fun': lambda x: round(1-(x[1]+x[3])) + abs(round(1-(x[1]+x[3])))}
                    )
    elif method in ("trust-constr",):
        constraints = (
            scipy.optimize.LinearConstraint(np.array([[1, 0, 0, 0]]), np.array([0.001]), np.array([np.inf])),
            scipy.optimize.LinearConstraint(np.array([[0, 1, 0, 0]]), np.array([0.001]), np.array([1])),
            scipy.optimize.LinearConstraint(np.array([[0, 0, 1, 0]]), np.array([0.001]), np.array([np.inf])),
            scipy.optimize.LinearConstraint(np.array([[0, 0, 0, 1]]), np.array([0.001]), np.array([1])),
            scipy.optimize.LinearConstraint(np.array([[0, 1, 0, -1]]), np.array([0.001]), np.array([1])),
            )
    else:
        constraints = None
    options = {"maxiter": 1000, "disp": False}

    proctimes = []
    nits = []
    x = []
    status = []
    cost = []
    for x0 in x0s:
        proctime = time.process_time()
        try:
            result = scipy.optimize.minimize(chi, x0, args=(aif, c, t, Rfun), method=method, jac=jac, hess=hess, bounds=bounds, constraints=constraints, options=options)
        except Exception as ex:
            result = {"success": False, "message": str(ex)}
            print("optimize", ex)
        proctime = time.process_time()-proctime
        
        proctimes.append(proctime)
        if result["success"]:
            if "njev" not in result:
                result["njev"] = 0
            if "nhev" not in result:
                result["nhev"] = 0
            if "nit" not in result:
                result["nit"] = result["nfev"]
            if method == "trust-constr":
                result["success"] = result["status"] == 1 or result["status"] == 2
            
            cost.append(result["fun"])
            nits.append(result["nit"])
            x.append(result["x"])
            status.append(result["status"])
        else:
            print(method, result["status"], result["message"])
    
    x = np.array(x)
    cost = np.array(cost)

    best = cost <= np.min(cost)

    return x[best], cost[best]

def optimize_all(aif, c, t, Rfun, get_x0s):
    results = []
    for method in ("Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
        if method in ("Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
            continue
        try:
            x, cost = optimize(aif, c, t, method, Rfun, get_x0s)
            results.append((method, x, cost))
        except Exception as ex:
            print(method, ex)
            raise ex
    return results

def _get_x0s():
    #return np.array([[0.1,0.9,0.1,0.1],[100,0.1,0.9,0.9]])
    np.random.seed(42)
    x0s = []
    ves = np.linspace(1,99,10)
    vps = 100-ves
    #fps = np.linspace(1,1000,2)
    fps = np.array([50, 1000])
    es = np.linspace(1,99,5)
    
    x0s = np.array(np.meshgrid(fps, ves, es, vps)).T.reshape(-1, 4)
    #print(x0s.shape)
    return x0s

def optimize_L_BFGS_B(aif, c, t, Rfun, get_x0s):
    x0s, bounds = get_x0s()

    proctimes = []
    nits = []
    x = []
    ux0 = []
    status = []
    cost = []
    for x0 in x0s:
        jac = None
        hess = None
        options = {"maxiter": 1000, "disp": False, "ftol": 1e-8, "gtol": 1e-8}
        proctime = time.process_time()
        try:
            result = scipy.optimize.minimize(chi, x0, args=(aif, c, t, Rfun), method="Powell", jac=jac, hess=hess, bounds=bounds, options=options)
        except Exception as ex:
            result = {"success": False, "message": str(ex)}
            print("optimize_L_BFGS_B", ex)    
        proctime = time.process_time()-proctime
        
        proctimes.append(proctime)
        if result["success"]:
            cost.append(result["fun"])
            nits.append(result["nit"])
            x.append(result["x"])
            ux0.append(x0)
            status.append(result["status"])
        else:
            print("L_BFGS_B", result["status"], result["message"])
    
    ux0 = np.array(ux0)
    x = np.array(x)
    cost = np.array(cost)

    best = cost <= np.min(cost)

    return x[best], cost[best], ux0[best]

def optimize_COBYLA(aif, c, t, Rfun, get_x0s):
    x0s, bounds = get_x0s()

    proctimes = []
    nits = []
    x = []
    ux0 = []
    status = []
    cost = []
    for x0 in x0s:
        jac = None
        hess = None
        options = {"maxiter": 1000, "disp": False, "tol": 1e-8}
        bounds = None
        constraints = ({'type': 'ineq', 'fun': lambda x: 0 if x[0] < 0.001 else x[0]},
                       {'type': 'ineq', 'fun': lambda x: 0 if x[1] < 0.001 else x[1]},
                       {'type': 'ineq', 'fun': lambda x: 0 if x[2] < 0.001 else x[2]},
                       {'type': 'ineq', 'fun': lambda x: 0 if x[3] < 0.001 else x[3]},
                       {'type': 'ineq', 'fun': lambda x: 0 if x[2] > 1 else 1-x[2]},
                       {'type': 'ineq', 'fun': lambda x: 0 if (1-(x[1]+x[3])) < 1 else 1-(x[1]+x[3]) }
                    )
        proctime = time.process_time()
        try:
            result = scipy.optimize.minimize(chi, x0, args=(aif, c, t, Rfun), method="COBYLA", jac=jac, hess=hess, bounds=bounds, constraints=constraints, options=options)
        except Exception as ex:
            result = {"status": -1, "message": str(ex)}
            print("optimize_COBYLA", ex)
        proctime = time.process_time()-proctime
        
        proctimes.append(proctime)
        if result["status"] in (1,2):
            if "nit" not in result:
                result["nit"] = result["nfev"]
            cost.append(result["fun"])
            nits.append(result["nit"])
            x.append(result["x"])
            ux0.append(x0)
            status.append(result["status"])
        else:
            print("COBYLA", result["status"], result["message"])
    
    ux0 = np.array(ux0)
    x = np.array(x)
    cost = np.array(cost)

    best = cost <= np.min(cost)

    return x[best], cost[best], ux0[best]

    #return "", np.average(proctimes), np.average(nits), (
    #        np.average(x0s[:,0]), np.std(x0s[:,0]), np.average(x0s[:,1]), np.std(x0s[:,1]), np.average(x0s[:,2]), np.std(x0s[:,2]), np.average(x0s[:,3]), np.std(x0s[:,3]),
    #        fp[best], vp[best], ps[best], ve[best], cost[best], np.average(cost), np.std(cost),
    #        np.average(fp), np.std(fp), np.average(vp), np.std(vp), np.average(ps), np.std(ps), np.average(ve), np.std(ve),
    #        np.average(np.abs(x0s[:,0]-fp)), np.std(np.abs(x0s[:,0]-fp)), np.average(np.abs(x0s[:,1]-vp)), np.std(np.abs(x0s[:,1]-vp)), np.average(np.abs(x0s[:,2]-ps)), np.std(np.abs(x0s[:,2]-ps)), np.average(np.abs(x0s[:,3]-ve)), np.std(np.abs(x0s[:,3]-ve)),
    #        np.quantile(fp,0.25),np.quantile(fp,0.5), np.quantile(fp,0.75), np.quantile(vp,0.25),np.quantile(vp,0.5), np.quantile(vp,0.75), np.quantile(ps,0.25),np.quantile(ps,0.5), np.quantile(ps,0.75), np.quantile(vp,0.25),np.quantile(ve,0.5), np.quantile(ve,0.75),
    #        ), len(status)
            
def optimize_trust_constr(aif, c, t, Rfun, get_x0s):
    np.random.seed(42)
    x0s, bounds = get_x0s()

    proctimes = []
    nits = []
    x = []
    ux0 = []
    status = []
    cost = []
    for x0 in x0s:
        jac = "2-point"
        #hess = lambda a,b,c,d,e: np.zeros((4,4))
        hess = scipy.optimize.BFGS()
        options = {"maxiter": 250, "disp": False, "gtol": 1e-12, "xtol": 1e-12}
        constraints = ()
        proctime = time.process_time()
        try:
            result = scipy.optimize.minimize(chi, x0, args=(aif, c, t, Rfun), method="Nelder-Mead", jac=jac, hess=hess, bounds=bounds, constraints=constraints, options=options)
        except Exception as ex:
            
            result = {"status": -1, "message": str(ex)}
            print("optimize_trust_constr", ex)

        proctime = time.process_time()-proctime
        proctimes.append(proctime)
        if result["status"] in (0,1,2):
            if "nit" not in result:
                result["nit"] = result["niter"]

            cost.append(result["fun"])
            nits.append(result["nit"])
            x.append(result["x"])
            ux0.append(x0)
            status.append(result["status"])
        else:
            print("trust_constr", result["status"], result["message"])

    ux0 = np.array(ux0)
    x = np.array(x)
    cost = np.array(cost)
    
    best = cost <= np.min(cost)

    return x[best], cost[best], ux0[best]

def optimize_TNC(aif, c, t, Rfun, get_x0s):
    np.random.seed(42)
    bounds = [(0.001, 1), (0.001, None), (0.001,1), (0.001,1)]
    x0s, bounds = get_x0s()
    bounds = [ ( (None if b[0]==np.inf else b[0]), (None if b[1]==np.inf else b[1]) ) for b in bounds]
    proctimes = []
    nits = []
    x = []
    ux0 = []
    status = []
    cost = []
    for x0 in x0s:
        jac = "2-point"
        options = {"maxiter": 250, "disp": False, "ftol": 1e-12, "gtol": 1e-12, "xtol": 1e-12}
        constraints = ()
        proctime = time.process_time()
        try:
            result = scipy.optimize.minimize(chi, x0, args=(aif, c, t, Rfun), method="TNC", jac=jac, bounds=bounds, constraints=constraints, options=options, tol=1e-12)
        except Exception as ex:
            result = {"success": False, "status": -1, "message": str(ex)}
            print("optimize_TNC", ex)

        proctime = time.process_time()-proctime
        proctimes.append(proctime)
        if result["success"]:
            if "nit" not in result:
                result["nit"] = result["niter"]

            cost.append(result["fun"])
            nits.append(result["nit"])
            x.append(result["x"])
            ux0.append(x0)
            status.append(result["status"])

    ux0 = np.array(ux0)
    x = np.array(x)
    cost = np.array(cost)
    
    if len(cost) > 0:
        best = cost <= np.min(cost)

        return x[best], cost[best], ux0[best]
    else:
        return [], [], []

def optimize_least_squares(aif, c, t, Rfun, get_x0s):
    np.random.seed(42)
    bounds = ([0.001, 0.001, 0.001, 0.001], [1, np.inf, 1, 1])
    x0s, bounds = get_x0s()
    bounds = ([b[0] for b in bounds], [b[1] for b in bounds])

    proctimes = []
    nits = []
    x = []
    ux0 = []
    status = []
    cost = []
    for x0 in x0s:
        jac = "2-point"
        gtol= 1e-15
        xtol= 1e-15
        ftol= 1e-15
        x_scale = 1.0
        f_scale = 1.0
        loss="linear"
        diff_step=None
        max_nfev=100
        tr_solver="exact"
        method="trf"
        verbose=0
        proctime = time.process_time()
        try:
            result = scipy.optimize.least_squares(chi, x0, args=(aif, c, t, Rfun), jac=jac, bounds=bounds, method=method, ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale, f_scale=f_scale, max_nfev=max_nfev,tr_solver=tr_solver,loss=loss,diff_step=diff_step,verbose=verbose)
        except Exception as ex:
            print("optimize_least_squares", ex)
            result = {"status": -1, "message": str(ex)}
        #result = scipy.optimize.minimize(chi2CXM, x0, args=(aif, c), method="trust-constr", jac=jac, hess=hess, bounds=bounds, constraints=constraints, options=options)
        proctime = time.process_time()-proctime

        proctimes.append(proctime)
        if result["status"] in (0,1,2,3,4):
            cost.append(result["cost"])
            nits.append(result["nfev"])
            x.append(result["x"])
            ux0.append(x0)
            status.append(result["status"])
        #else:
        #    print("least_squares", result["status"], result["message"])
    
    ux0 = np.array(ux0)
    x = np.array(x)
    cost = np.array(cost)

    best = cost <= np.min(cost)

    return x[best], cost[best], ux0[best]

    #return "", np.average(proctimes), np.average(nits), (
    #        np.average(x0s[:,0]), np.std(x0s[:,0]), np.average(x0s[:,1]), np.std(x0s[:,1]), np.average(x0s[:,2]), np.std(x0s[:,2]), np.average(x0s[:,3]), np.std(x0s[:,3]),
    #        fp[best], vp[best], ps[best], ve[best], cost[best], np.average(set(cost)), np.std(set(cost)),
    #        np.average(fp), np.std(fp), np.average(vp), np.std(vp), np.average(ps), np.std(ps), np.average(ve), np.std(ve),
    #        np.average(np.abs(x0s[:,0]-fp)), np.std(np.abs(x0s[:,0]-fp)), np.average(np.abs(x0s[:,1]-vp)), np.std(np.abs(x0s[:,1]-vp)), np.average(np.abs(x0s[:,2]-ps)), np.std(np.abs(x0s[:,2]-ps)), np.average(np.abs(x0s[:,3]-ve)), np.std(np.abs(x0s[:,3]-ve)),
    #        np.quantile(fp,0.25),np.quantile(fp,0.5), np.quantile(fp,0.75), np.quantile(vp,0.25),np.quantile(vp,0.5), np.quantile(vp,0.75), np.quantile(ps,0.25),np.quantile(ps,0.5), np.quantile(ps,0.75), np.quantile(vp,0.25),np.quantile(ve,0.5), np.quantile(ve,0.75),
    #        ), len(status)

def read_csv(filename):
    results = {}
    if not os.path.exists(filename):
        return results
    paramindex = {"Fp": 0, "vp": 1, "PS": 2, "ve": 3, "MTT": 4, "Ktrans": 5, "E": 6, "χ": 7, "x0s": 8}
    with open(filename, "r", encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if csv_reader == None or csv_reader.fieldnames == None:
            return results
        for name in csv_reader.fieldnames:
            algo, param = name.rsplit('-', 1)
            if algo not in results:
                results[algo] = ([], [], [], [], [], [], [], [], [])
        for row in csv_reader:
            for k in row.keys():
                if row[k] == "" or row[k] == None:
                    continue
                algo, param = k.rsplit("-",1)
                if param != "x0s":
                    results[algo][paramindex[param]].append(float(row[k]))
                else:
                    results[algo][paramindex[param]].append(row[k])

    for a in results:
        results[a] = (np.array(results[a][0]), np.array(results[a][1])/100.0, np.array(results[a][2]), np.array(results[a][3])/100.0, 
                      np.array(results[a][4]), np.array(results[a][5]), np.array(results[a][6])/100.0, np.array(results[a][7]), np.array(results[a][8]))

    return results
        

def make_rows(results, out):
    rows = [[]]
    for i, algo in enumerate(sorted(results.keys())):
        if len(results[algo][0]) == 0:
            i = i-1
            continue
        
        if len(results[algo]) == 3:
            aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s = out(results[algo])
        else:
            aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s = results[algo]
        #print(aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s)
        use = np.logical_and.reduce(( np.bitwise_or(avp==-1, avp >= 0), np.bitwise_or(ave==-1, ave >= 0), avp+ave<=1, aE<=1, np.bitwise_or(aE==-1, aE>=0), ak>=0, np.bitwise_or(amtt==-1, amtt>=0), np.bitwise_or(aFp==-1, aFp>=0), aPS>=0))
        if out == out_Tofts and not (use.any()):
            print(aFp, avp, aPS, ave, amtt, ak, aE, chis, x0s)

        if not use.any():
            continue

        rows[0].append(algo + "-Fp")
        rows[0].append(algo + "-vp")
        rows[0].append(algo + "-PS")
        rows[0].append(algo + "-ve")
        rows[0].append(algo + "-MTT")
        rows[0].append(algo + "-Ktrans")
        rows[0].append(algo + "-E")
        rows[0].append(algo + "-χ")
        rows[0].append(algo + "-x0s")
        

        while len(rows) < len(aFp[use])+1:
                rows.append([])
        for j,Fp in enumerate(aFp[use]):
            while len(rows[j+1]) < i*9:
                rows[j+1].append('')
            rows[j+1].append(Fp)
        for j,vp in enumerate(avp[use]):
            while len(rows[j+1]) < i*9+1:
                rows[j+1].append('')
            rows[j+1].append(vp*100)
        for j,PS in enumerate(aPS[use]):
            while len(rows[j+1]) < i*9+2:
                rows[j+1].append('')
            rows[j+1].append(PS)
        for j,ve in enumerate(ave[use]):
            while len(rows[j+1]) < i*9+3:
                rows[j+1].append('')
            rows[j+1].append(ve*100)
        for j,mtt in enumerate(amtt[use]):
            while len(rows[j+1]) < i*9+4:
                rows[j+1].append('')
            rows[j+1].append(mtt)
        for j,k in enumerate(ak[use]):
            while len(rows[j+1]) < i*9+5:
                rows[j+1].append('')
            rows[j+1].append(k)
        for j,e in enumerate(aE[use]):
            while len(rows[j+1]) < i*9+6:
                rows[j+1].append('')
            rows[j+1].append(e*100)
        for j,chi in enumerate(chis[use]):
            while len(rows[j+1]) < i*9+7:
                rows[j+1].append('')
            rows[j+1].append(chi)
        for j,x0 in enumerate(x0s[use]):
            while len(rows[j+1]) < i*9+8:
                rows[j+1].append('')
            rows[j+1].append(x0)
    return rows

def find_start_index(aif):
    i = np.argmax(aif[:15])
    prev = aif[i]
    while i>0 and prev>=aif[i]:
        prev=aif[i]
        i = i-1
    return i+1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--aath', default=False, action="store_true")
    parser.add_argument('--dp', default=False, action="store_true")
    parser.add_argument('--ctu', default=False, action="store_true")
    parser.add_argument('--ptu', default=False, action="store_true")
    parser.add_argument('--2cxm', default=False, action="store_true")
    parser.add_argument('--tofts', default=False, action="store_true")
    parser.add_argument('--least_squares', default=False, action="store_true")
    parser.add_argument('--trust_constr', default=False, action="store_true")
    parser.add_argument('--COBYLA', default=False, action="store_true")
    parser.add_argument('--L_BFGS_B', default=False, action="store_true")
    parser.add_argument('--TNC', default=False, action="store_true")
    parser.add_argument('--all', default=False, action="store_true")

    results = {}
    results_trust_constr = {}
    results_TNC = {}
    results_least_squares = {}
    results_COBYLA = {}
    results_L_BFGS_B = {}

    args = parser.parse_args()

    oall = args.all
    if args.aath:
        Rfun = RAATH
        get_x0s = x0s_AATH
        out = out_AATH
    elif args.dp:
        Rfun = RDP
        get_x0s = x0s_DP
        out = out_DP
    elif args.ctu:
        Rfun = RCTU
        get_x0s = x0s_CTU
        out = out_CTU
        results_least_squares = read_csv('../Data/perfusion_RCTU_least_squares.csv')
    elif args.ptu:
        Rfun = RPTU
        get_x0s = x0s_PTU
        out = out_PTU
    elif args.tofts:
        Rfun = RTofts
        get_x0s = x0s_Tofts
        out = out_Tofts
        results_least_squares = read_csv('../Data/perfusion_RTofts_least_squares.csv')
    else:
        Rfun = R2CXM
        get_x0s = x0s_2CXM
        out = out_2CXM
        results_least_squares = read_csv('../Data/perfusion_R2CXM_least_squares.csv')

    useLeastSquares = args.least_squares
    useTrustConstr = args.trust_constr
    useTNC = args.TNC
    useCOBYLA = args.COBYLA
    useL_BFGS_B = args.L_BFGS_B

    

    print(Rfun.__name__, oall)

    counter = 0

    ts = []
    if os.path.isfile("../Data/acquisition_times.csv"):
        with open('../Data/acquisition_times.csv') as csv_file:
            try:
                ts = [[float(n) for n in line.split(";")] for line in csv_file.readlines()]
            except Exception:
                pass
    if len(ts) == 8:
        t1 = np.array(ts[0])
        t2 = np.array(ts[1])
        t3 = np.array(ts[2])
        t4 = np.array(ts[3])
        t5 = np.array(ts[4])
        t6 = np.array(ts[5])
        t7 = np.array(ts[6])[1::2]
        t8 = np.array(ts[7])
    else:
        t1 = get_t('../Data/input_', "35673135")
        t2 = get_t('../Data/input_', "40418168")
        # -0.55
        t3 = get_t('../Data/input_', "42005025")
        # 0.7
        t4 = get_t('../Data/input_', "42121701")
        # 0.55
        t5 = get_t('../Data/input_', "42216394")
        # 1.8
        t6 = get_t('../Data/input_', "42234636")
        # -0.4
        t7 = get_t('../Data/input_', "42306110")[1::2]
        # -2.8
        t8 = get_t('../Data/input_', "42204402")
        # 1.1


        with open('../Data/acquisition_times.csv', "w") as csv_file:
            csv_file.writelines(
                [";".join([str(n) for n in t])+"\n" for t in (t1, t2, t3, t4, t5, t6, t7, t8)]
            )
    
    use_t2 = ["40418168", "42175470", ]
    use_t3 = ["42005025", ]
    use_t4 = ["42121701", "42272900", "42292501", "42351313", "42352525"]
    use_t5 = ["42216394", ]
    use_t6 = ["42234636", ]
    use_t7 = ["42306110", ]
    use_t8 = ["42204402", ]
    
    test = ["42123446", "42125160", "42175470", "42224343"]

    for full_true_c, image, patname, is_slice in get_true_tumor_c("../Data/preprocessed"):
        fullname = patname
        if is_slice >= 0:
            fullname = patname + "_slice"

        #full_t = get_t('../Data/input_', name)[:70]
        #if name != "42306110":
        #if np.size(t) == 0:
        #    print(name)
        if patname in use_t2:
            full_t = t2
        elif patname in use_t3:
            full_t = t3
        elif patname in use_t4:
            full_t = t4
        elif patname in use_t5:
            full_t = t5
        elif patname in use_t6:
            full_t = t6
        elif patname in use_t7:
            full_t = t7
        elif patname in use_t8:
            full_t = t8
        else:
            full_t = t1

        #if (np.abs(t-old_t)>0.2).any():
        #    print(name, t-old_t)6
        #old_t = t
        #continue
        if patname not in test:
            nets = ["target"]
        else:
            nets = None
        
        nets = ["target"]

        for full_c, net in get_all_tumor_c("../Data/output", image, patname, is_slice, full_true_c, nets=nets):
            for full_aif,aif_name in get_aif("../Data/n_out", patname):
                
                k = fullname + "-" + net + "-" + aif_name
                if k in results_least_squares and aif_name not in ("DisLAIF") and "parker2006" not in aif_name:
                    print("skipped " +k)
                    continue

                start_i = find_start_index(full_aif)
                aif = full_aif[start_i:full_c.shape[0]]+1e-5
                c = full_c[start_i:]
                t = full_t[start_i:]
                true_c = full_true_c[start_i:]

                #print(aif)

                #print(patname, full_aif.shape, full_c.shape, full_true_c.shape, len(full_t))
                #print(patname, aif.shape, c.shape, true_c.shape, len(t))

                if aif_name in ("max05total", "noop"):
                    continue
                if oall:
                    for result in optimize_all(aif, c, t, Rfun, get_x0s):
                        try:
                            results[fullname + "-" + net + "-" + aif_name + "-" + result[0]] = result[1:]
                        except Exception as e:
                            results[fullname + "-" + net + "-" + aif_name + "-" + result[0]] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print(e)
                else:
                    if useTrustConstr:
                        try:
                            print("optimize trust_constr", fullname, net, aif_name, Rfun.__name__)
                            proctime = time.process_time()
                            results_trust_constr[fullname + "-" + net + "-" + aif_name] = optimize_trust_constr(aif, c, t, Rfun, get_x0s)
                            proctime = time.process_time() - proctime
                            print("trust_constr finished", proctime / 60, fullname + "-" + net + "-" + aif_name, len(results_trust_constr[fullname + "-" + net + "-" + aif_name][0]))
                        except Exception as e:
                            results_trust_constr[fullname + "-" + net + "-" + aif_name] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print(e)
                            raise
                    if useLeastSquares:
                        try:
                            print("optimize least_squares", fullname, net, aif_name, Rfun.__name__)
                            proctime = time.process_time()
                            results_least_squares[fullname + "-" + net + "-" + aif_name] = optimize_least_squares(aif, c, t, Rfun, get_x0s)
                            proctime = time.process_time() - proctime
                            print("least_squares finished", proctime / 60, fullname + "-" + net + "-" + aif_name, len(results_least_squares[fullname + "-" + net + "-" + aif_name][0]))
                        except Exception as e:
                            results_least_squares[fullname + "-" + net + "-" + aif_name] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print("least_squares finished", fullname + "-" + net + "-" + aif_name, e)
                            raise
                    if useTNC:
                        try:
                            print("optimize TNC", fullname, net, aif_name, Rfun.__name__)
                            proctime = time.process_time()
                            results_TNC[fullname + "-" + net + "-" + aif_name] = optimize_TNC(aif, c, t, Rfun, get_x0s)
                            proctime = time.process_time() - proctime
                            print("TNC finished", proctime / 60, fullname + "-" + net + "-" + aif_name, len(results_TNC[fullname + "-" + net + "-" + aif_name][0]))
                        except Exception as e:
                            results_TNC[fullname + "-" + net + "-" + aif_name] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print("TNC finished", fullname + "-" + net + "-" + aif_name, e)
                            raise
                    if useCOBYLA:
                        try:
                            results_COBYLA[fullname + "-" + net + "-" + aif_name] = optimize_COBYLA(aif, c, t, Rfun, get_x0s)
                            print("COBYLA finished", len(results_COBYLA[fullname + "-" + net + "-" + aif_name][0]))
                        except Exception as e:
                            results_COBYLA[fullname + "-" + net + "-" + aif_name] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print(e)
                    if useL_BFGS_B:
                        try:
                            results_L_BFGS_B[fullname + "-" + net + "-" + aif_name] = optimize_L_BFGS_B(aif, c, t, Rfun, get_x0s)
                            print("L_BFGS_B finished", len(results_L_BFGS_B[fullname + "-" + net + "-" + aif_name][0]))
                        except Exception as e:
                            results_L_BFGS_B[fullname + "-" + net + "-" + aif_name] = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
                            print(e)

            if oall:
                with open("../Data/perfusion_" + Rfun.__name__ + ".csv", 'w', newline='', encoding="utf8") as csv_file:
                    csv_writer = csv.writer(csv_file, dialect='excel')
                    rows = make_rows(results, out)
                    for row in rows:
                        #print(len(row))
                        csv_writer.writerow(row)
            else:
                if useTrustConstr:
                    with open("../Data/perfusion_" + Rfun.__name__ + "_trust_constr.csv", 'w', newline='', encoding="utf8") as csv_file:
                        csv_writer = csv.writer(csv_file, dialect='excel')
                        rows = make_rows(results_trust_constr, out)
                        for row in rows:
                            #print(len(row))
                            csv_writer.writerow(row)
                if useTNC:
                    with open("../Data/perfusion_" + Rfun.__name__ + "_TNC.csv", 'w', newline='', encoding="utf8") as csv_file:
                        csv_writer = csv.writer(csv_file, dialect='excel')
                        rows = make_rows(results_TNC, out)
                        for row in rows:
                            #print(len(row))
                            csv_writer.writerow(row)
                if useL_BFGS_B:
                    with open("../Data/perfusion_" + Rfun.__name__ + "_L_BFGS_B.csv", 'w', newline='', encoding="utf8") as csv_file:
                        csv_writer = csv.writer(csv_file, dialect='excel')
                        rows = make_rows(results_L_BFGS_B, out)
                        for row in rows:
                            #print(len(row))
                            csv_writer.writerow(row)
                if useLeastSquares:
                    with open("../Data/perfusion_" + Rfun.__name__ + "_least_squares.csv", 'w', newline='', encoding="utf8") as csv_file:
                        csv_writer = csv.writer(csv_file, dialect='excel')
                        rows = make_rows(results_least_squares, out)
                        #print(len(rows))
                        for row in rows:
                            #print(len(row))
                            csv_writer.writerow(row)
                if useCOBYLA:
                    with open("../Data/perfusion_" + Rfun.__name__ + "_COBYLA.csv", 'w', newline='', encoding="utf8") as csv_file:
                        csv_writer = csv.writer(csv_file, dialect='excel')
                        rows = make_rows(results_COBYLA, out)
                        for row in rows:
                            #print(len(row))
                            csv_writer.writerow(row)
from numpy import *
import numpy as np
from datetime import datetime,timedelta
import myfunc.util as util
import random
from sklearn.decomposition import IncrementalPCA
import sys, os
import calendar
from collections import deque


Year  = 2017
lMon  = [1]
ldydx = [[dy,dx] for dy in [-1,0,1] for dx in [-3,-2,-1,0,1,2,3]]
#ldydx = [[dy,dx] for dy in [-1,0,1] for dx in [-2,-1,0,1,2]]

#lisurf = range(1,15+1)  # surface type index
#lisurf = [3,1,2] + range(4,15+1)  # surface type index
#lisurf = range(8,15+1)  # surface type index
lisurf = [1]  # surface type index
ntc1 = 9
ntc2 = 4
ncomb = len(ldydx)*(ntc1+ntc2)

#********************************************
# Functions
#********************************************
def read_Tc(lDTime=None, ldydx=None, isurf=None):
    a2tc = deque([])
    for DTime in lDTime:
        Year,Mon,Day = DTime.timetuple()[:3]
        a2tcTmp = None
        for idydx,(dy,dx) in enumerate(ldydx):
            #srcDir = '/work/hk01/utsumi/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
            srcDir = '/mnt/j/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
            srcPath1=srcDir + '/Tc1.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
            srcPath2=srcDir + '/Tc2.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
            if not os.path.exists(srcPath1): continue
            atc1 = np.load(srcPath1)
            atc2 = np.load(srcPath2)
            atc  = np.c_[atc1, atc2]

            try:
                a2tcTmp = np.c_[a2tcTmp, atc]
            except ValueError:
                a2tcTmp = atc

        if a2tcTmp is None:
            continue
        else:
            a2tcTmp = np.array(a2tcTmp)
        #**********************
        a2tc.extend(a2tcTmp)

    return np.array(a2tc)

def read_Tc(lDTime=None, ldydx=None, isurf=None):
    a2tc = deque([])
    for DTime in lDTime:
        Year,Mon,Day = DTime.timetuple()[:3]
        a2tcTmp = None
        for idydx,(dy,dx) in enumerate(ldydx):
            #srcDir = '/work/hk01/utsumi/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
            srcDir = '/mnt/j/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
            srcPath1=srcDir + '/Tc1.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
            srcPath2=srcDir + '/Tc2.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
            if not os.path.exists(srcPath1): continue
            atc1 = np.load(srcPath1)
            atc2 = np.load(srcPath2)
            atc  = np.c_[atc1, atc2]

            try:
                a2tcTmp = np.c_[a2tcTmp, atc]
            except ValueError:
                a2tcTmp = atc

        if a2tcTmp is None:
            continue
        else:
            a2tcTmp = np.array(a2tcTmp)
        #**********************
        a2tc.extend(a2tcTmp)

    return np.array(a2tc)

#********************************************
for Mon in lMon:
    if Mon ==1:
        lDTime = util.ret_lDTime(datetime(Year,12,1),datetime(Year,12,31),timedelta(days=1)) \
                + util.ret_lDTime(datetime(Year,1,1),datetime(Year,2,28),timedelta(days=1))
    else:
        lDTime = util.ret_lDTime(datetime(Year,Mon-1,1),datetime(Year,Mon+1,calendar.monthrange(Year,Mon+1)[1]),timedelta(days=1))


print lDTime
print ncomb
sys.exit()

'''
    meanPath1= meanDir + '/mean1.0dy.0dx.%02dsurf.npy'%(isurf)
    meanPath2= meanDir + '/mean2.0dy.0dx.%02dsurf.npy'%(isurf)
    stdPath1 = meanDir + '/std1.0dy.0dx.%02dsurf.npy'%(isurf)
    stdPath2 = meanDir + '/std2.0dy.0dx.%02dsurf.npy'%(isurf)

    damean1[isurf] = np.load(meanPath1)
    damean2[isurf] = np.load(meanPath2)

    dastd1[isurf] = np.load(stdPath1)
    dastd2[isurf] = np.load(stdPath2)
'''

#****** Initialize ************
dinc_pca = {}
for isurf in lisurf:
    dinc_pca[isurf] = IncrementalPCA(n_components=ncomb)
#------------------------------
    
for isurf in lisurf:

    amean = deque([])
    astd  = deque([])
    for (dy,dx) in ldydx:
        amean.extend(damean1[isurf])
        amean.extend(damean2[isurf])
        astd.extend(dastd1[isurf])
        astd.extend(dastd2[isurf])

    initflag = True
    for Year,Mon in lYM:
        eDay = calendar.monthrange(Year,Mon)[1]
        iDTime = datetime(Year,Mon,1)
        eDTime = datetime(Year,Mon,eDay)
        dDTime = timedelta(days=1)
        lDTime = util.ret_lDTime(iDTime, eDTime, dDTime)

        #lDTime = lDTime[25:]  # test

        if initflag:   # To avoid PCA for too small samples
            a2tc = deque([])

        for DTime in lDTime:
            print isurf,DTime
            Day = DTime.day
    
            a2tcTmp = None
            for idydx,(dy,dx) in enumerate(ldydx):
                srcDir = '/work/hk01/utsumi/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
                srcPath1=srcDir + '/Tc1.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
                srcPath2=srcDir + '/Tc2.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
                if not os.path.exists(srcPath1): continue
                atc1 = np.load(srcPath1)
                atc2 = np.load(srcPath2)
    
                atc  = c_[atc1, atc2]
               
                try:
                    a2tcTmp = c_[a2tcTmp, atc]
                except ValueError:
                    a2tcTmp = atc    

            if a2tcTmp is None:
                continue
            else:
                a2tcTmp = array(a2tcTmp)


            #**********************
            # Screen invalid data
            #********************** 
            try:
                a1flag  = ma.masked_inside(a2tcTmp,50,350).mask.all(axis=1)
                a2tcTmp = a2tcTmp[a1flag]
            except AxisError:
                pass

            #**********************
            # Normalize
            #********************** 
            if a2tcTmp.shape[0]==0: continue
            #a2tcTmp = a2tcTmp - amean
            a2tcTmp = (a2tcTmp - amean)/astd

            #**********************
            a2tc.extend(a2tcTmp)

        if len(a2tc)>ncomb:
            a2tc = array(a2tc)
            dinc_pca[isurf].partial_fit(a2tc)
            initflag = True
            print '-----------PCA!-------------'
        else:
            initflag = False

    #** Make eigen vectors and values --
    egvec = dinc_pca[isurf].components_      # (n-th, nComb)
    egval = dinc_pca[isurf].explained_variance_
    varratio= dinc_pca[isurf].explained_variance_ratio_
    
    #** Save ----------
    coefDir = '/work/hk01/utsumi/PMM/stop/data/coef'
    util.mk_dir(coefDir)
    egvecPath = coefDir + '/egvec.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    egvalPath = coefDir + '/egval.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    varratioPath = coefDir + '/varratio.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    np.save(egvecPath, egvec)
    np.save(egvalPath, egval)
    np.save(varratioPath, varratio)
    print egvecPath
    #print evec
         

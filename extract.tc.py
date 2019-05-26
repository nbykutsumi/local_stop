import numpy as np
from numpy import *
import myfunc.util as util
from datetime import datetime, timedelta
import glob
import os,sys
import calendar
import h5py
from collections import deque
import myfunc.util as util

varName = 'nltb'
iYM = [2017,1]
eYM = [2017,1]
lYM = util.ret_lYM(iYM,eYM)
#outDir= '/work/hk01/utsumi/PMM/TPCDB/PC_COEF'
#outDir= '/work/hk01/utsumi/PMM/stop/data'
outDir= '/mnt/j/stop'
verGMI = '05'
subverGMI = 'A'
fullverGMI = '%s%s'%(verGMI,subverGMI)
miss = -9999.
ntc1  = 9
ntc2  = 4
ldy   = [-1,0,1]
ldx   = [-3,-2,-1,0,1,2,3]
Shape = [len(ldy),len(ldx)]
ldydx = [[dy,dx] for dy in ldy for dx in ldx]

#**** Function *****************
def shift_array(ain=None, dy=None,dx=None,miss=-9999):
    ny,nx,nz = ain.shape
    aout = np.ones([ny,nx,nz]).astype(ain.dtype)*miss
    if   dy<=0: iy0=0; ey0=ny-abs(dy); iy1=abs(dy); ey1=ny
    elif dy> 0: iy0=abs(dy); ey0=ny; iy1=0; ey1=ny-abs(dy)
    if   dx<=0: ix0=0; ex0=nx-abs(dx); ix1=abs(dx); ex1=nx
    elif dx> 0: ix0=abs(dx); ex0=nx; ix1=0; ex1=nx-abs(dx)
    
    aout[iy0:ey0,ix0:ex0] = ain[iy1:ey1,ix1:ex1]
    return aout
#*******************************
for (Year,Mon) in lYM:
    eDay   = calendar.monthrange(Year,Mon)[1]
    iDTime = datetime(Year,Mon,1)
    eDTime = datetime(Year,Mon,eDay)

    dDTime = timedelta(days=1)
    lDTime = util.ret_lDTime(iDTime,eDTime,dDTime)
    #matchBaseDir = '/work/hk01/utsumi/PMM/MATCH.GMI.V05A'
    matchBaseDir = '/mnt/j/PMM/MATCH.GMI.V05A'
    #-- Read list -----
    #listDir  = '/work/hk01/utsumi/PMM/TPCDB/list'
    listDir  = '/mnt/j/PMM/TPCDB/list'
    listPath = listDir + '/list.1C.V05.%04d%02d.csv'%(Year,Mon)
    f=open(listPath,'r'); lines = f.readlines(); f.close()
    dlorbit = {}
    for line in lines:
        line = map(int, line.split(','))
        oid,Year,Mon,Day,itime,etime = line
        try:
            dlorbit[Day].append(line)
        except KeyError: 
            dlorbit[Day] = [line]
    #-------------------
    for DTime in lDTime:
        Day = DTime.day
        try:
            lorbit = dlorbit[Day]
        except KeyError:
            continue
            
        #-- Initialize --
        datc1 = {}
        datc2 = {}
        for dydx in ldydx:
            dy,dx = dydx
            for isurf in range(1,15+1):
                datc1[(dy,dx,isurf)] = deque([])
                datc2[(dy,dx,isurf)] = deque([])
        
        #----------------
        #lorbit = lorbit[:2]  # test
        for orbinfo in lorbit:
            oid,Year,Mon,Day,itime,etime = orbinfo
    
            #-- Storm Top Height ----
            stopDir  = matchBaseDir + '/S1.ABp103-117.Ku.V06A.heightStormTop/%04d/%02d/%02d'%(Year,Mon,Day)
            stopPath = stopDir + '/heightStormTop.1.%06d.npy'%(oid)
            #a2stop = np.load(stopPath)
            a2stop = np.load(stopPath)
            if a2stop.max()<0: continue
    
            #-- Surface Type Index --
            surftypeDir = matchBaseDir + '/S1.ABp103-117.GMI.surfaceTypeIndex/%04d/%02d/%02d'%(Year,Mon,Day)
            surftypePath= surftypeDir + '/surfaceTypeIndex.%06d.npy'%(oid)
            #a2surftype = np.load(surftypePath)
            a2surftype = np.load(surftypePath)
            a1surftype = a2surftype.flatten()

            #-- Make flag array -----
            da1flag = {}
            a1flagStop = ma.masked_greater_equal(a2stop,0).mask.flatten()
            for isurf in range(1,15+1):
                a1flagSurf = ma.masked_equal(a1surftype,isurf).mask
                da1flag[isurf] = a1flagStop * a1flagSurf
    
            #-- Read Tb from HDF ----
            #baseDirGMI = '/work/hk01/PMM/NASA/GPM.GMI/1C/V%s'%(verGMI)
            baseDirGMI = '/mnt/j/PMM/NASA/GPM.GMI/1C/V%s'%(verGMI)
            srcDirGMI  = baseDirGMI+ '/%04d/%02d/%02d'%(Year,Mon,Day)
            #1C.GPM.GMI.XCAL2016-C.20170303-S021910-E035143.017105.V05A.HDF5
            ssearchGMI = srcDirGMI + '/1C.GPM.GMI.*.%06d.V%s.HDF5'%(oid,fullverGMI)
    
            print ssearchGMI
            tcPath = glob.glob(ssearchGMI)[0]
            with h5py.File(tcPath) as h:
                a3tc1 = h['S1/Tc'][:]
                a3tc2Org = h['S2/Tc'][:]
    
            #******************************
            # Match-up S2 to S1
            #******************************
            idxDir = matchBaseDir + '/S1.ABp000-220.GMI.S2.IDX/%04d/%02d/%02d'%(Year,Mon,Day)
            yPath= idxDir + '/Ypy.1.%06d.npy'%(oid)
            xPath= idxDir + '/Xpy.1.%06d.npy'%(oid)
            a1y = ma.masked_less(np.load(yPath).flatten(),0)
            a1x = ma.masked_less(np.load(xPath).flatten(),0)
            a1mask = a1y.mask + a1x.mask
            a2tc2 = a3tc2Org[a1y.filled(0),a1x.filled(0),:]
            a2tc2[a1mask,:] = miss
    
            ny,nx,ntc2 = a3tc2Org.shape
            a3tc2 = a2tc2.reshape(ny,nx,ntc2)


   
            #******************************
            # Extract x=103-3 -- 117+3
            #******************************
            a3tc1 = a3tc1[:,100:120+1,:]
            a3tc2 = a3tc2[:,100:120+1,:]


            #******************************
            #  Collect surroundings
            #******************************
            for idydx,[dy,dx] in enumerate(ldydx):
                #-- S1 ----
                a3tmp1 = shift_array(a3tc1, dy,dx,miss=-9999.)
                a3tmp2 = shift_array(a3tc2, dy,dx,miss=-9999.)
    
                #******************************
                # Extract x=103 - 117
                # Caution! Already extracted for x=101-119
                #******************************
                a3tmp1 = a3tmp1[:,3:-3,:]
                a3tmp2 = a3tmp2[:,3:-3,:]
    
                a1tmp1 = a3tmp1.reshape(-1,ntc1)
                a1tmp2 = a3tmp2.reshape(-1,ntc2)


                #print a1surftype.shape
                #print a2stop.flatten().shape
                #print a1tmp1.shape
                #print a2stop.shape 
                #******************************
                # Screen StormTop >=0 and SurfaceType
                #******************************
                for isurf in range(1,15+1):
                    a1flag= da1flag[isurf]
                    if type(a1flag)==np.bool_:
                        continue
                    else:
                        a1sc1 = a1tmp1[a1flag]
                        a1sc2 = a1tmp2[a1flag]
   
                        datc1[(dy,dx,isurf)].extend(a1sc1)
                        datc2[(dy,dx,isurf)].extend(a1sc2)
                #sys.exit()    




        #******** Save ******************
        for (dy,dx) in ldydx:
            for isurf in range(1,15+1):
                atc1 = array(datc1[(dy,dx,isurf)])
                atc2 = array(datc2[(dy,dx,isurf)])


                Year,Mon,Day = DTime.timetuple()[:3]
                #tcoutDir = '/work/hk01/utsumi/PMM/stop/data/Tc_p106-114/%04d/%02d/%02d'%(Year,Mon,Day)
                tcoutDir = '/mnt/j/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
                tcoutPath1= tcoutDir + '/Tc1.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
                tcoutPath2= tcoutDir + '/Tc2.%ddy.%ddx.%02dsurf.npy'%(dy,dx,isurf)
                util.mk_dir(tcoutDir)
                np.save(tcoutPath1, atc1)
                np.save(tcoutPath2, atc2)
        print tcoutPath1
            
            

 

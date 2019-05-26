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
verGMI = '05'
subverGMI = 'A'
fullverGMI = '%s%s'%(verGMI,subverGMI)
miss = -9999.
ldy   = [0]
ldx   = [0]
Shape = [len(ldy),len(ldx)]
ldydx = [[dy,dx] for dy in ldy for dx in ldx]
varNameFull= 'S1.ABp103-117.GMI.Latitude'
varName = varNameFull.split('.')[-1]
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

for (dy,dx) in ldydx:
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
        #lDTime = lDTime[:3]  # test
        for DTime in lDTime:
            Day = DTime.day
            try:
                lorbit = dlorbit[Day]
            except KeyError:
                continue
    
            #-- Initialize --
            dastop = {}
    
            for isurf in range(1,15+1):
                dastop[isurf] = deque([])
    
            #----------------
            #lorbit = lorbit[:2]  # test
            for orbinfo in lorbit:
                oid,Year,Mon,Day,itime,etime = orbinfo
        
                #-- Storm Top Height ----
                stopDir  = matchBaseDir + '/S1.ABp103-117.Ku.V06A.heightStormTop/%04d/%02d/%02d'%(Year,Mon,Day)
                stopPath = stopDir + '/heightStormTop.1.%06d.npy'%(oid)
                a2stop = np.load(stopPath)
                if a2stop.max()<=0: continue

                a1stop = a2stop.flatten() 

                #-- Variable to be extracted ----
                varDir  = matchBaseDir + '/%s/%04d/%02d/%02d'%(varNameFull,Year,Mon,Day)
                datPath = varDir + '/%s.%06d.npy'%(varName, oid)
                a2dat   = np.load(datPath) 
                #--- shift --------------
                nytmp,nxtmp = a2dat.shape 
                a2shift = shift_array(a2dat.reshape(nytmp,nxtmp,1),dy,dx,-9999.).reshape(nytmp,nxtmp)
 
                a1shift= a2shift.flatten() 
        
                #-- Surface Type Index --
                surftypeDir = matchBaseDir + '/S1.ABp103-117.GMI.surfaceTypeIndex/%04d/%02d/%02d'%(Year,Mon,Day)
                surftypePath= surftypeDir + '/surfaceTypeIndex.%06d.npy'%(oid)
                a1surftype = np.load(surftypePath).flatten()
    
                #-- Make flag array -----
                a1flagStop = ma.masked_greater(a1stop,0).mask   # Use a1stop, not a1shift.
                for isurf in range(1,15+1):
                    a1flagSurf = ma.masked_equal(a1surftype,isurf).mask
                    a1flag = a1flagStop * a1flagSurf
                    if a1flag.any()==False: continue
    
                    a1sc   = a1shift[a1flag]
                    dastop[isurf].extend(a1sc)
                    #print 'stop>0:',a1flagStop.sum()
                    #print 'isurf:',a1flagSurf.sum()
                    #print 'aft:',isurf,a1flag.sum(),len(a1sc)
                    
    
            print 'dastop[3].shape',len(dastop[3])
            #******** Save ******************
            for isurf in range(1,15+1):
                aout = array(dastop[isurf])
                Year,Mon,Day = DTime.timetuple()[:3]
                #outDir = '/work/hk01/utsumi/PMM/stop/data/%s/%04d/%02d/%02d'%(varName,Year,Mon,Day)
                outDir = '/mnt/j/PMM/stop/data/%s/%04d/%02d/%02d'%(varName,Year,Mon,Day)
                outPath= outDir + '/%s.%ddy.%ddx.%02dsurf.npy'%(varName,dy,dx,isurf)
                util.mk_dir(outDir)
                np.save(outPath, aout)
    
                if isurf==1:
                    print outPath

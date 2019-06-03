from numpy import *
from PIL import Image
import numpy as np
import shutil
import os, sys

lact = ['H','L','LT']
#lact = ['H']
lisurf = range(1,14+1)
coef_b = 5
latmin,latmax = [-90,90]
for i,act in enumerate(lact):

    da2dat = {}
    for i,isurf in enumerate(lisurf):
        expr    = 'act.%s.surf%d.b%d.lat.%d.%d'%(act,isurf,coef_b,latmin,latmax)
        figDir  = '/mnt/c/ubuntu/fig'
        figPath = figDir + '/train.%s.png'%(expr)
    
        a2png     = Image.open(figPath)
        a2array   = np.asarray(a2png)
        print ''
        print a2array.shape
        a2array   = a2array[730:-20,750:-100]
        print a2array.shape
        da2dat[i] = a2array

    a2dummy = ones(a2array.shape)*255
    a2line1 = hstack([da2dat[0], da2dat[1], da2dat[2], da2dat[3], da2dat[4]]) 
    a2line2 = hstack([da2dat[5], da2dat[6], da2dat[7], da2dat[8], da2dat[9]])
    a2line3 = hstack([da2dat[10], da2dat[11], da2dat[12], da2dat[13], a2dummy])

    a2oarray= vstack([a2line1, a2line2, a2line3])
    oimg    = Image.fromarray(uint8(a2oarray))

    oPath   = figDir + '/join.scatter.act.%s.png'%(act)
    oimg.save(oPath)
    print oPath




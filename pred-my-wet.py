import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec
from glob import glob
import tensorflow as tf
import numpy.ma as ma
import sys,os
from datetime import datetime, timedelta
import myfunc.util as util
from collections import deque
from sklearn.decomposition import PCA
#get_ipython().magic(u'matplotlib inline')

#******************************************
# Functions
#******************************************

def mk_daylist(days=None, rat_train=1.0):
    nall   = days
    ntrain = int(nall*rat_train)
    np.random.seed(0)

    a1idx = range(nall)
    a1idx = np.random.choice(a1idx, len(a1idx), replace=False)
    a1idx_train = a1idx[:ntrain]
    a1idx_valid = a1idx[ntrain:]

    return a1idx_train,a1idx_valid


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

def read_var_collect(varName=None, lDTime=None, ldydx=None, isurf=None):
    a2var = deque([])
    for DTime in lDTime:
        Year,Mon,Day = DTime.timetuple()[:3]
        a2varTmp = None
        for idydx,(dy,dx) in enumerate(ldydx):
            #srcDir = '/work/hk01/utsumi/PMM/stop/data/Tc/%04d/%02d/%02d'%(Year,Mon,Day)
            srcDir = '/mnt/j/PMM/stop/data/%s/%04d/%02d/%02d'%(varName,Year,Mon,Day)
            srcPath=srcDir + '/%s.%ddy.%ddx.%02dsurf.npy'%(varName,dy,dx,isurf)
            if not os.path.exists(srcPath): continue
            avar = np.load(srcPath)

            try:
                a2varTmp = np.c_[a2varTmp, avar]
            except ValueError:
                a2varTmp = avar

        if a2varTmp is None:
            continue
        else:
            a2varTmp = np.array(a2varTmp)
        #**********************
        a2var.extend(a2varTmp)
    return np.array(a2var)

def read_pc_coef(isurf):
    #*********************************
    # Read PC coefficient
    #*********************************
    #coefDir = '/work/hk01/utsumi/PMM/stop/data/coef'
    coefDir = '/mnt/j/PMM/stop/data/coef'
    egvecPath = coefDir + '/egvec.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    egvalPath = coefDir + '/egval.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    varratioPath = coefDir + '/varratio.%02dch.%03dpix.%02dsurf.npy'%(ntc1+ntc2, len(ldydx),isurf)
    
    a2egvec = np.load(egvecPath)  # (n-th, ncomb)
    a1varratio = np.load(varratioPath)
    a1cumvarratio= np.cumsum(a1varratio)
    return a2egvec, a1varratio, a1cumvarratio

def my_unit(x,Min,Max):
    return (x-Min)/(Max-Min)

def unit(x):
    return ( x - np.min(x,0) )/( np.max(x,0) - np.min(x,0) )


def rmse(x,y):
    x = x.flatten()
    y = y.flatten()
    return np.sqrt((((x-y))**2).mean())
def Rmse(x,y):
    Min,Max=MinStop,MaxStop
    return np.sqrt( ( ( ((Max-Min)*x+Min).flatten()-((Max-Min)*y+Min).flatten() )**2 ).mean() )

def cc(x,y):
    return np.corrcoef( x.flatten(), y.flatten() )[0,1]


print 'Define functions'
#******************************************
# Read data
#******************************************
ldy   = [-1,0,1]
ldx   = [-3,-2,-1,0,1,2,3]
#ldx   = [-2,-1,0,1,2]
imid  = int((len(ldy)*len(ldx)-1)/2)
ldydx = [[dy,dx] for dy in ldy for dx in ldx]
isurf = 1
ldays_train,ldays_valid = mk_daylist(days=31, rat_train=0.2)
lDTime_train = [datetime(2017,1,1)+timedelta(days=i) for i in ldays_train]
lDTime_valid = [datetime(2017,1,1)+timedelta(days=i) for i in ldays_valid]


trainTc   = read_Tc(lDTime_train, ldydx, isurf)
trainStop = read_var_collect('stop', lDTime_train, ldydx, isurf)
trainLat  = read_var_collect('Latitude', lDTime_train, [[0,0]], isurf)

print trainTc.shape, trainStop.shape, trainLat.shape

#******************************************
# Screen latitude
#******************************************
#latmin = -30
#latmax = 30
latmin=-90
latmax=-20

index_keep = []
for i in range(trainTc.shape[0]):
    lat = trainLat[i]
    if (latmin<=lat)and(lat<=latmax):
        index_keep.append(i)
print len(index_keep)
trainTc   = trainTc[index_keep]
trainStop = trainStop[index_keep]
trainLat  = trainLat[index_keep]
print trainTc.shape, trainStop.shape, trainLat.shape


#******************************************
# Screen by the number of storms
#******************************************
nstorm = 1
#nstorm = 21
nostorm = len(ldy)*len(ldx)-nstorm
if nstorm >1:
    a1flagstop = ((trainStop>=0).sum(axis=1) >=nstorm)
    trainTc   = trainTc[a1flagstop]
    trainStop = trainStop[a1flagstop]
trainTc.shape, trainStop.shape
trainStop = trainStop[:,imid].reshape(-1,1)
print trainTc.shape, trainStop


#******************************************
# Preprocess parameters 
#******************************************

# PC parameters
restriction = 10
#restriction = 20
#amean = trainTc.mean(axis=0)
#astd  = trainTc.std(axis=0)
#pca = PCA(n_components=restriction)
#pca.fit((trainTc-trainTc.mean(0))/trainTc.std(0))
#a2egvec=pca.components_
##a1egval=pca.explained_variance
#a1varratio=pca.explained_variance_ratio_

Mon=1
preptype = 'nynx.%dx%d.isurf.%d.Mon.%d.Lat.%d.%d'%(len(ldy),len(ldx),isurf,Mon,latmin,latmax)
paramDir = '/mnt/j/PMM/stop/prep-param/%s'%(preptype)

#*** Tc mean, std *******
meanPath = paramDir + '/mean.Tc.npy'
stdPath  = paramDir + '/std.Tc.npy'
ameanTc = np.load(meanPath)
astdTc  = np.load(stdPath)

#*** PC coefficient (eigen vector)
egvecPath = paramDir + '/egvec.npy'
varratioPath = paramDir + '/varratio.npy'
a2egvec   = np.load(egvecPath)
a1varratio= np.load(varratioPath)

#*** PC min, max ***************
minPath = paramDir + '/pc.min.npy'
maxPath = paramDir + '/pc.max.npy'
MinPC   = np.load(minPath)
MaxPC   = np.load(maxPath)

#******************************************
# Resample
#******************************************
resample_rate = 1.0
N = int(trainTc.shape[0]*resample_rate)
aidx = np.random.choice(np.arange(trainTc.shape[0]), N).astype('int32')
trainTc  = trainTc[aidx,:]
trainStop= trainStop[aidx,:]
print trainTc.shape, trainStop.shape


#******************************************
# PCA
#******************************************
trainTc = (trainTc-ameanTc)/astdTc
#reduction = np.dot(trainTc, a2egvec[:restriction,:].T)
reduction = np.dot(trainTc, a2egvec[:restriction,:].T)

#******************************************
# Divide training/test data
#******************************************
aidx = np.random.choice(np.arange(len(trainStop)), len(trainStop))
nTrain = int(len(aidx)*0.9)
aidxTrain = aidx[:nTrain]
aidxTest  = aidx[nTrain:]

print trainStop.shape
print nTrain
print reduction.shape
MinStop = 0
MaxStop = 32000

#trainX = unit(reduction)
#trainY = unit(trainStop)
trainX = my_unit(reduction,MinPC,MaxPC)
#trainX = unit(reduction)
trainY = my_unit(trainStop,MinStop,MaxStop)

testX = trainX[aidxTest]
testY = trainY[aidxTest]
trainX= trainX[aidxTrain]
trainY= trainY[aidxTrain]

print(trainX.shape, trainY.shape, testX.shape, testY.shape)


#******************************************
# Histogram
#******************************************
figPath = '/mnt/c/ubuntu/fig/fig1.png'
pl.subplot(111)
_,_,_ = pl.hist(trainY, 100)
pl.savefig(figPath)
pl.clf()
print figPath

#******************************************
# Figure function
#******************************************
def Figure(Label, Prediction, bins):
    Min,Max = MinStop, MaxStop
    recover_testY = (Max-Min)*Label.flatten()      + Min
    recover_pred  = (Max-Min)*Prediction.flatten() + Min

    pl.figure(figsize=(15,15))
    gs = gridspec.GridSpec(2,2, width_ratios=[1,1], height_ratios=[1,1])
    
    pl.subplot(gs[0,:])
    pl.plot(recover_testY/1000., c='r', label ='Observation')
    pl.plot(recover_pred /1000., c='b', label ='Prediction')
    pl.ylabel('height(km)')
    pl.ylim([0,18])
    pl.legend()
    pl.title('non-storm<=%d b=%d lat=[%.1f, %.1f] box=%dx%d'%(nostorm, bpara,latmin, latmax, len(ldy),len(ldx)))
    print('RMSE:'     , np.round(rmse(Label, Prediction) , 4))
    print('real RMSE:', np.round(Rmse(Label, Prediction) , 4))
    print('CC:'       , np.round(  cc(Label, Prediction) , 4))
    
    pl.subplot(gs[2]) # values prediction and testY are between -4 and 4
    aa = recover_pred
    bb = recover_testY
    interval           = np.array([ Min + (Max - Min)/bins*i for i in range(bins+1) ])
    interval1          = np.array([ Min + (Max - Min)/bins*i for i in range(bins+1) ])
    revised_interval   = interval[:-1]  + (Max - Min)/(2*bins)
    revised_interval1  = interval1[:-1] + (Max - Min)/(2*bins)
    cumulative_number  = []
    cumulative_number1 = []
    for i in range(bins):
        cumulative_number.append(  (aa < interval[i+1] ).sum() - (aa < interval[i] ).sum() )
        cumulative_number1.append( (bb < interval1[i+1]).sum() - (bb < interval1[i]).sum() )
    pl.plot(revised_interval/.1000          , cumulative_number   , color='green', alpha=0.5, label='Prediction')    
    pl.fill_between(revised_interval/.1000  , cumulative_number, 0, color='green', alpha=0.5)
    pl.plot(revised_interval1/.1000         , cumulative_number1  , color='red'  , alpha=0.5 ,label='Observation')    
    pl.fill_between(revised_interval1/.1000 ,cumulative_number1, 0, color='red'  , alpha=0.5)
    pl.ylabel('number of samples')
    pl.xlabel('height(km)')
    pl.legend() 
    pl.title('Distribution')
    pl.legend()

    #*** 2D histogram **********
    H,xbnd,ybnd = np.histogram2d(recover_testY/1000, recover_pred/1000, bins=[np.arange(0,20,0.5), np.arange(0,20,0.5)])
    H = ma.masked_equal(H,0)
    X,Y = np.meshgrid(xbnd,ybnd)
    pl.subplot(gs[3])
    pl.pcolormesh(X,Y,H.T, cmap='jet')
    pl.axis([0,18,0,18])
    pl.xticks([0,5,10,15])
    pl.yticks([0,5,10,15])
    pl.plot([0,35],[0,35],'k')
    pl.xlabel('Observation(km)')
    pl.ylabel('Prediction(km)')
    pl.title('Correlation')
    pl.grid()
    pl.colorbar()
    
    #pl.scatter(recover_testY/1000, recover_pred/1000,s=3)
    #pl.plot(np.arange(18000)/1000,np.arange(18000)/1000,c='black',linestyle = ':')
    #pl.axis([0,18,0,18])
    #pl.xticks([0,5,10,15])
    #pl.yticks([0,5,10,15])
    #pl.xlabel('Observation(km)')
    #pl.ylabel('Prediction(km)')
    #pl.title('Correlation')
    #pl.grid()

    figPath = '/mnt/c/ubuntu/fig/fig2.png'
    pl.savefig(figPath)


#******************************************
# Checkpoint
#******************************************
bpara  = 5
degree = 12
coef_b = bpara
#coef_poly  = mk_coef_polyfit(testY, degree, coef_b)

#**** Checkpoints **********************
expr= 'surf%d.b%d.lat.%d.%d'%(isurf,bpara,latmin,latmax)
#ckptDir = '/work/hk01/utsumi/PMM/stop/ml-param-%d'%(act)
ckptDir = '/mnt/j/PMM/stop/ml-param/%s'%(expr)
ckptPath= ckptDir + '/stop.%02d'%(isurf)
#******************************************
# Prediction
#******************************************
print testX.shape
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(ckptPath + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(ckptDir + '/'))
    graph = tf.get_default_graph()
    X      = graph.get_tensor_by_name('input:0')
    pred   = graph.get_tensor_by_name('pred:0')
    print X
    print pred
    out    = sess.run(pred, feed_dict={X:testX})
print out.shape
prediction = out
print 'testX.min, max', testX.min(), testX.max()
print 'prediction.min(),max()',prediction.min(),prediction.max()

#******************************************
# Figure
#******************************************

Figure(testY, prediction, 50)


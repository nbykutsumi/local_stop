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

#**********************************************
# Functions
#**********************************************
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
#****************************************************
# Set parameters and date list 
#****************************************************
print 'Define functions'

ldy   = [-1,0,1]
ldx   = [-3,-2,-1,0,1,2,3]
#ldx   = [-2,-1,0,1,2]
imid  = int((len(ldy)*len(ldx)-1)/2)
ldydx = [[dy,dx] for dy in ldy for dx in ldx]
isurf = 1
samplerate = 0.3
Year = 2017
Mon  = 1
saveparaflag = 1  # 0/1
MinStop = 0
MaxStop = 32000

if Mon == 1:
    #lDTime = util.ret_lDTime(datetime(Year,1,1), datetime(Year,2,28), timedelta(days=1))
    #lDTime = lDTime + util.ret_lDTime(datetime(Year,12,1), datetime(Year,12,31), timedelta(days=1))
    lDTime = util.ret_lDTime(datetime(Year,1,1), datetime(Year,1,31), timedelta(days=1))

else:
    eDay = calendar.monthrange(Year,Mon+1)[1]
    lDTime = util.ret_lDTime(datetime(Year,1,1), datetime(Year,2,eDay), timedelta(days=1))
np.random.seed(0)
np.random.shuffle(lDTime)

lDTime = lDTime[:int(len(lDTime)*samplerate)]
lDTime_train = lDTime[: int(len(lDTime)*0.7)]
lDTime_valid = lDTime[int(len(lDTime)*0.7):]

print 'Days (train, valid)',len(lDTime_train), len(lDTime_valid)

#****************************************************
# Read data
#****************************************************
trainTc   = read_Tc(lDTime_train, ldydx, isurf)
#trainStop = read_var_collect('stop', lDTime_train, ldydx, isurf)
trainStop = read_var_collect('stop', lDTime_train, [[0,0]], isurf)
trainLat  = read_var_collect('Latitude', lDTime_train, [[0,0]], isurf)

validTc   = read_Tc(lDTime_valid, ldydx, isurf)
#validStop = read_var_collect('stop', lDTime_valid, ldydx, isurf)
validStop = read_var_collect('stop', lDTime_valid, [[0,0]], isurf)
validLat  = read_var_collect('Latitude', lDTime_valid, [[0,0]], isurf)


print trainTc.shape, trainStop.shape, trainLat.shape
print validTc.shape, validStop.shape, validLat.shape

#****************************************************
# Screen latitude
#****************************************************
#latmin = -30
#latmax = 30
latmin=-90
latmax=-20

index_keep = []
for i in range(trainTc.shape[0]):
    lat = trainLat[i]
    if (latmin<=lat)and(lat<=latmax):
        index_keep.append(i)

trainTc   = trainTc[index_keep]
trainStop = trainStop[index_keep]
trainLat  = trainLat[index_keep]


index_keep = []
for i in range(validTc.shape[0]):
    lat = validLat[i]
    if (latmin<=lat)and(lat<=latmax):
        index_keep.append(i)

validTc   = validTc[index_keep]
validStop = validStop[index_keep]
validLat  = validLat[index_keep]


print trainTc.shape, trainStop.shape, trainLat.shape

#****************************************************
# Screen number of storms in each box
#****************************************************
nstorm = 1
#nstorm = 21
nostorm = len(ldy)*len(ldx)-nstorm
if nstorm >1:
    a1flagstop = ((trainStop>=0).sum(axis=1) >=nstorm)
    trainTc   = trainTc[a1flagstop]
    trainStop = trainStop[a1flagstop]
    trainStop = trainStop[:,imid].reshape(-1,1)

    a1flagstop = ((validStop>=0).sum(axis=1) >=nstorm)
    validTc   = validTc[a1flagstop]
    validStop = validStop[a1flagstop]
    validStop = validStop[:,imid].reshape(-1,1)


#-- test ------
aidx = range(trainTc.shape[0])
np.random.shuffle(aidx)
aidx = aidx[:10000]
validTc   = trainTc[aidx]
validStop = trainStop[aidx]
#--------------


trainStop = trainStop.reshape(-1,1)
validStop = validStop.reshape(-1,1)
print trainTc.shape, trainStop

#****************************************************
# PC parameters
#****************************************************
amean = trainTc.mean(axis=0)
astd  = trainTc.std(axis=0)
pca = PCA()
pca.fit((trainTc-amean)/astd)
a2egvec=pca.components_
#a1egval=pca.explained_variance
a1varratio=pca.explained_variance_ratio_
a1cumrat = np.cumsum(a1varratio)

#restriction = ma.masked_greater(a1cumrat,0.99).argmax()
restriction = 8
print 'restriction=',restriction

trainTc = (trainTc-amean)/astd
validTc = (validTc-amean)/astd
#reduction = np.dot(trainTc, a2egvec[:restriction,:].T)

trainReduction = np.dot(trainTc, a2egvec[:restriction,:].T)
validReduction = np.dot(validTc, a2egvec[:restriction,:].T)

#****************************************************
#*** Save preprocess parameters *******
#****************************************************
if saveparaflag ==1:
    Mon=1
    preptype = 'nynx.%dx%d.isurf.%d.Mon.%d.Lat.%d.%d'%(len(ldy),len(ldx),isurf,Mon,latmin,latmax)
    paramDir = '/mnt/j/PMM/stop/prep-param/%s'%(preptype)
    util.mk_dir(paramDir)
    
    #*** Save Tc mean, std *******
    meanPath = paramDir + '/mean.Tc.npy'
    stdPath  = paramDir + '/std.Tc.npy'
    np.save(meanPath, amean)
    np.save(stdPath,  astd)
    
    #*** PC coefficient (eigen vector)
    egvecPath = paramDir + '/egvec.npy'
    #egvalPath = paramDir + '/egval.npy'
    varratioPath = paramDir + '/varratio.npy'
    np.save(egvecPath, a2egvec.astype('float32'))
    np.save(varratioPath, a1varratio.astype('float32'))
    
    #*** PC min, max ***************
    minPath = paramDir + '/pc.min.npy'
    maxPath = paramDir + '/pc.max.npy'
    np.save(minPath, trainReduction.min(axis=0))
    np.save(maxPath, trainReduction.max(axis=0))
    print minPath
#****************************************************
print 'trainReduction.min, max=',trainReduction.min(),trainReduction.max()
MinRed = trainReduction.min()
MaxRed = trainReduction.max()

#trainX = unit(reduction)
#trainY = unit(trainStop)
print 'Before unit'
print 'trainReduction.min, max', trainReduction.min(), trainReduction.max()
print 'validReduction.min, max', validReduction.min(), validReduction.max()


trainX = my_unit(trainReduction,MinRed,MaxRed)
trainY = my_unit(trainStop,MinStop,MaxStop)
testX = my_unit(validReduction,MinRed,MaxRed)
testY = my_unit(validStop,MinStop,MaxStop)

print ''
print 'After unit'
print 'trainX.min, max', trainX.min(), trainX.max()
print 'testX.min, max', testX.min(), testX.max()


print(trainX.shape, trainY.shape, testX.shape, testY.shape)

print 'reduction.min, max',trainReduction.min(), trainReduction.max()
print 'trainX.min, max', trainX.min(), trainX.max()


def FFN(TraX, TraY, TesX, TesY, learning_rate, epochs, batch_size, dim, act): 
    fn1 = tf.nn.sigmoid
    fn2 = tf.nn.relu
    def fn3(x):
        return x/(1+np.abs(x))
    ac  = [fn1,fn3,fn1,fn3,fn1,fn1,fn1,fn1,fn1,fn1,fn1,fn1,fn1,fn1] # number of entry = len(dim) - 2
    total_batch = int(len(TraX)/batch_size) + 1
    Xdata = [ TraX[i*batch_size:(i+1)*batch_size] for i in range(total_batch) ]
    Ydata = [ TraY[i*batch_size:(i+1)*batch_size] for i in range(total_batch) ]
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, TraX.shape[1]], 'input')
    Y = tf.placeholder(tf.float32, [None, TraY.shape[1]])
        
    W = [ tf.Variable(tf.random_normal([dim[i], dim[i+1]]), name='w%d'%(i)) for i in range(len(dim) - 1) ]
    b = [ tf.Variable(tf.random_normal([dim[i+1]]), name='b%d'%(i))         for i in range(len(dim) - 1) ]
    A = [ X ]
    for i in range(len(dim) - 2):
        A.append(ac[i](tf.matmul(A[-1], W[i]) + b[i]))
    A.append(tf.matmul(A[-1], W[-1]) + b[-1])  
    if act == 0:
        cost = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(Y - A[-1]) ))) 
    elif act == 1:
        cost = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(Y - A[-1])*error_function(Y,label,100,12,1,bpara) ))) 
    elif act == 2:
        cost = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(Y - A[-1])*my_error_func(Y, coef_poly )))) 

    gogo = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    real = tf.placeholder(tf.float32, [None, TraY.shape[1]])
    pred = tf.placeholder(tf.float32, [None, TraY.shape[1]])
    rmse = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(real - pred))))
    prediction=tf.add(tf.matmul(A[-2], W[-1]), b[-1], name='pred')  # for Save
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #*** Saver ********
    saver = tf.train.Saver(max_to_keep=3)
    #******************

    for epoch in range(epochs):    
        for i in range(total_batch):
            feed1 = {X:Xdata[i], Y:Ydata[i]}
            sess.run(gogo, feed_dict = feed1)
            training_error = sess.run(cost, feed_dict = feed1)
            prediction     = sess.run(A[-1], feed_dict = {X:TesX})
            test_error     = sess.run(rmse, feed_dict = {real:TesY, pred:prediction})
        if epoch % int(epochs/5) == 0:    
            print('Training Error:',training_error,'and','Testing Error:', test_error)

            #**** Save **********************
            #ckptDir = '/work/hk01/utsumi/PMM/stop/ml-param-%d'%(act)
            ckptDir = '/mnt/j/PMM/stop/ml-param/%s'%(expr)
            util.mk_dir(ckptDir)
            ckptPath= ckptDir + '/stop.%02d'%(isurf)
            sv = saver.save(sess, ckptPath)
            #******************************** 
            
    return prediction


#def my_unit(x,Min,Max):
#    return (x-Min)/(Max-Min)

def mk_coef_polyfit(a1obs, degree, coef_b):
    # a1obs must be in range of [0,1]
    nbins = 100
    a1bnd = np.arange(nbins+1).astype('float32')/(nbins)
    frequency,_ = np.histogram(a1obs, bins=a1bnd)
    g = (coef_b - 1)*(-frequency/float(frequency.max()) +1)+1
    x = 0.5*(a1bnd[:-1]+a1bnd[1:])
    coef = np.polyfit(x,g,deg=degree)  # highest degree coef first.
    print 'coef.shape=',coef.shape
    return coef[::-1]

def my_error_func(a1obs, coef):
    degree = len(coef)-1
    for i in range(degree+1):
        if i==0:
            y = coef[i]*(a1obs**i)
        else:
            y = y + coef[i]*(a1obs**i)
    return y


    
def error_function(x, data, bins, degree, Min, Max):
    vec       = ((data - np.min(data,0))/float(np.max(data,0)-np.min(data,0)))
    interval  = [ i/float(bins) for i in range(bins + 1)]
    frequency = np.array([ ((vec<=interval[i+1]).sum() - (vec<interval[i]).sum())/float(len(vec)) for i in range(bins) ])
    xx        = np.arange(bins)/float(bins - 1)
    mat       = np.concatenate([(xx**i).reshape(-1,1) for i in range(degree)], axis=1)
    #print mat
    #print frequency
    coef      = np.dot(np.linalg.inv(np.dot(mat.T,mat)), np.dot(mat.T, frequency))
    poly      = 1 - sum([coef[i]*(x**i) for i in range(degree)])
    values    = 1 - sum([coef[i]*(xx**i) for i in range(degree)])
    M, N      = np.max(values), np.min(values)
    return (Max - Min)/float(M - N)*(poly - N) + Min 
def uniformize(reduction, label):
    mat = np.concatenate([reduction, label], axis=1)
    temporary = []
    for i in range(mat.shape[1]):
        a = np.arange(len(mat)).reshape(-1,1)
        b = np.concatenate([a,mat[:,[i]]], axis=1)
        c = b[b[:,1].argsort()]
        c[:,1] = np.arange(len(mat))/(len(mat)-1)
        d = c[c[:,0].argsort()]
        temporary.append(d[:,1])
    input_data = (np.array(temporary).T)[:, :-1]
    target     = (np.array(temporary).T)[:,[-1]]
    return input_data, target
def rmse(x,y):
    x = x.flatten()
    y = y.flatten()
    return np.sqrt((((x-y))**2).mean())
def Rmse(x,y):
    Min,Max=MinStop,MaxStop
    return np.sqrt( ( ( ((Max-Min)*x+Min).flatten()-((Max-Min)*y+Min).flatten() )**2 ).mean() )
def cc(x,y):
    return np.corrcoef( x.flatten(), y.flatten() )[0,1]
def sort(x):
    return np.sort(x.flatten())
def unit(x):
    return (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))
def f_act(x,label):
    degree = 12
    #y_val = np.sort(unit(label.flatten()))
    y_val = np.sort(my_unit(label.flatten(),MinStop,MaxStop))
    X     = (np.arange(len(y_val))/float(len(y_val)-1) )
    mat   = np.concatenate([(X**i).reshape(-1,1) for i in range(degree)], axis=1)
    print 'X.shape=',X.shape
    print 'mat.shape=',mat.shape
    coef  = np.dot(np.linalg.inv(np.dot(mat.T,mat)), np.dot(mat.T, y_val))
    #poly  = sum([coef[i]*(x**i) for i in range(degree)], axis=1)
    poly  = sum([coef[i]*(x**i) for i in range(degree)], axis=0)
    print 'poly.shape=',poly.shape
    return poly
#x = np.arange(100001)/float(100000)
#pl.figure(figsize=(14,4))
#pl.subplot(131)
#pl.plot(x,f_act(x,trainY))
##pl.plot(np.arange(len(trainY))/float(len(trainY)-1),np.sort(unit(trainY.flatten())))
#pl.plot(np.arange(len(trainY))/float(len(trainY)-1),np.sort(my_unit(trainY.flatten(),MinStop,MaxStop)))
#pl.subplot(132)
#pl.plot(x, error_function(x, trainY, 100, 12, 1, 10))

#pl.subplot(133)
pl.subplot(111)
_,_,_ = pl.hist(trainY, 100)

figPath = '/mnt/c/ubuntu/fig/fig1.png'
pl.savefig(figPath)
pl.clf()


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
    pl.title('non-storm<=%d act=%d b=%d lat=[%.1f, %.1f] box=%dx%d'%(nostorm,act, bpara,latmin, latmax, len(ldy),len(ldx)))
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
    pl.clf()

#****************************************************
#****************************************************
bpara  = 5
degree = 12
coef_b = bpara
coef_poly  = mk_coef_polyfit(testY, degree, coef_b)



# Apply error function
#Train, Label = unit(reduction), unit(label)
#ntrain       = int(0.7*len(reduction))
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
#dim = [trainX.shape[2], 30, 30, 30,10, trainY.shape[1]]
dim = [trainX.shape[1], 30, 30, 30,10, trainY.shape[1]]
act = 2
expr= 'surf%d.b%d.lat.%d.%d'%(isurf,bpara,latmin,latmax)
prediction   = FFN(trainX, trainY, testX, testY, 0.005, 50, 1024*4, dim, act)
print 'trainX.min, max', trainX.min(), trainX.max()
print 'trainY.min, max', trainY.min(), trainY.max()

print 'testX.min, max', testX.min(), testX.max()
print 'testY.min, max', testY.min(), testY.max()
print 'prediction.min, max',prediction.min(),prediction.max()


print prediction.min(),prediction.max()

Figure(testY, prediction, 50)


import numpy as np
#from extractFeat import featPull
import scipy.io as sio

frames = np.load('frames.npy')
labels = np.load('labels.npy')

feats = sio.loadmat('features.mat')
feats = feats['feat']

import numpy as np

keysX = [0,3,6,9,15,27,51,56]
keysY = np.add(keysX, 1)
keysZ = np.add(keysX, 2)

def featExtract(singleFeat,keys):
    mean = [singleFeat[keys[0]]]
    rms = [singleFeat[keys[1]]]
    axr = singleFeat[keys[2]:keys[3]]
    spp = singleFeat[keys[4]:keys[5]]
    pwr = singleFeat[keys[6]:keys[7]]
    return np.concatenate((mean, rms, axr, spp, pwr))

def featPull(singleFeat):
    featX = featExtract(singleFeat,keysX)
    featY = featExtract(singleFeat,keysY)
    featZ = featExtract(singleFeat,keysZ)
    return featX, featY, featZ

def fuseFrame(frame, feat):
    featX, featY, featZ = featPull(feat)
    for x,y,z in zip(featX, featY, featZ):
        frame = np.append(frame,(x,y,z))
    return frame.reshape(102,3)

fusedFrames = []

for frame, feat in zip(frames, feats):
    fusedFrames.append(fuseFrame(frame,feat))

fusedFrames = np.asarray(fusedFrames)

np.save('fusedFrames',fusedFrames)
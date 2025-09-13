import numpy as np
import random
import torch


def PFs(featureD,J):
    maxN = len(J)
    pos = J
    evenInds = pos%2 == 0
    oddInds = ~evenInds
    posM = pos[:, np.newaxis].repeat(featureD, axis=1)

    dimInd = np.arange(1, featureD + 1)
    dimM = dimInd[np.newaxis, :].repeat(maxN, axis=0)
    # even
    allPEs = np.zeros((maxN, featureD))
    allPEs[evenInds, :] = np.sin(posM[evenInds, :] / 1000 ** (2 * dimM[evenInds, :] / featureD))
    allPEs[oddInds, :] = np.cos(posM[oddInds, :] / 1000 ** (2 * dimM[oddInds, :] / featureD))

    return allPEs



def addEmpty(data,seed):
    vf = data['varFeatures']
    data['groupFeatures'] = torch.zeros(vf.shape[0], 32)

    return data



def randPEs(nTrial,nSample,maxN,featureD,seed):
    np.random.seed(seed)
    choice = np.arange(1, maxN + 1)

    # allPEs = getAllPEs(maxN,featureD)
    # inds = np.arange(0,maxN)
    allSamples = []
    for t in range(nTrial):
        # ts = np.random.choice(inds,nSample,replace=False)
        selectedJ = np.random.choice(choice, size=nSample, replace=False)
        allSamples = np.concatenate([allSamples,selectedJ])
    sampledPEs = PFs(featureD, allSamples)
    sampledPEs = sampledPEs.reshape((nTrial, nSample, -1))

    return sampledPEs





def addNoiseUniform(data,seed):

    vf = data['varFeatures']
    reorderInds = data['reorderInds'].long().reshape(-1)

    np.random.seed(seed)
    randV = np.random.rand(vf.shape[0])
    randV = (randV*1000000).round()
    groupFeatures = torch.Tensor(PFs(32,randV))
    data['groupFeatures'] = groupFeatures

    return data

def addBPNoiseOrbit(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(nElement - 7, nGroup, nGroup, 32, seed=seed)
    groupFeaturesM = randPEs(1, nGroup * 7, nGroup * 7, 32, seed=seed*2).reshape(7, nGroup, 32)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesM),torch.Tensor(groupFeaturesS)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1, 32)
    data['groupFeatures'] = groupFeatures

    return data


def addIPNoiseOrbit(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(nElement-5,nGroup,nGroup,32,seed=seed)
    groupFeaturesM = randPEs(1, nGroup*5, nGroup*5, 32, seed=seed*2).reshape(5,nGroup,32)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesS), torch.Tensor(groupFeaturesM)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1, 32)
    data['groupFeatures'] = groupFeatures

    return data

def addSMSPNoiseOrbit(data,seed):

    rowGroups = [
        [11,12],
        [23, 53],
        [87, 88],
        [4, 13, 20],
        [79, 80, 86],
        [0, 7, 16, 34],
        [3, 14, 18, 33, 70],
        [8, 9, 15, 17, 19],
        [52, 62, 76, 77, 78],
        [5,  6,  10,  21,  24, 110],
        [35,  40,  42,  43,  68,  74,  75,  82, 109],

    ]

    trivialRowGroup = [i for i in range(111) if i not in sum(rowGroups,[])]
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesX = torch.zeros(nElement,nGroup,32)

    # non-trivial rows
    seedInvRatio = 0.1
    for step,rg in enumerate(rowGroups):
        nR = len(rg)
        rgPE = randPEs(1,nR*nGroup,nR*nGroup,32,seed + int(seedInvRatio*seed)*(step+1) ).reshape(nR,nGroup,-1)
        groupFeaturesX[rg] = torch.Tensor(rgPE)

    # trivial rows
    groupFeaturesX[trivialRowGroup] = torch.Tensor(randPEs(len(trivialRowGroup),nGroup,nGroup,32,seed*2))
    # colPEs = torch.Tensor(randPEs(len(trivialRowGroup),nGroup,nGroup,32,seed*2))
    # rowPEs = torch.Tensor(randPEs(1, len(trivialRowGroup), len(trivialRowGroup), 32, seed * 2)).repeat(nGroup, 1,1).permute(1, 0,2)
    # groupFeaturesX[trivialRowGroup] = colPEs / 2 + rowPEs / 2


    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1, 32)
    data['groupFeatures'] = groupFeatures

    return data


def addSMSPNoiseGroup(data,seed):

    rowGroups = [
        [11,12],
        [23, 53],
        [87, 88],
        [4, 13, 20],
        [79, 80, 86],
        [0, 7, 16, 34],
        [3, 14, 18, 33, 70],
        [8, 9, 15, 17, 19],
        [52, 62, 76, 77, 78],
        [5,  6,  10,  21,  24, 110],
        [35,  40,  42,  43,  68,  74,  75,  82, 109],

    ]

    trivialRowGroup = [i for i in range(111) if i not in sum(rowGroups,[])]
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesX = torch.zeros(nElement,nGroup,32)

    # non-trivial rows
    seedInvRatio = 0.1
    for step,rg in enumerate(rowGroups):
        nR = len(rg)
        rgPE = randPEs(1,nR*nGroup,nR*nGroup,32,seed + int(seedInvRatio*seed)*(step+1)).reshape(nR,nGroup,-1)
        groupFeaturesX[rg] = torch.Tensor(rgPE)

    # trivial rows

    groupFeaturesX[trivialRowGroup] = torch.Tensor(randPEs(1,nGroup,nGroup,32,seed*2)).repeat(len(trivialRowGroup),1,1)
    colPEs = torch.Tensor(randPEs(1,nGroup,nGroup,32,seed*2)).repeat(len(trivialRowGroup),1,1)
    rowPEs = torch.Tensor(randPEs(1,len(trivialRowGroup),len(trivialRowGroup),32,seed*2)).repeat(nGroup,1,1).permute(1,0,2)
    groupFeaturesX[trivialRowGroup] = colPEs/2 + rowPEs/2

    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1, 32)
    data['groupFeatures'] = groupFeatures

    return data

def addBPNoiseGroup(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    fixedPE = randPEs(nTrial=1, nSample=nGroup, maxN=nGroup, featureD=32, seed=seed)
    groupFeaturesS = torch.Tensor(fixedPE.repeat(nElement - 7, axis=0))
    groupFeaturesM = randPEs(1, nGroup * 7, nGroup * 7, 32, seed=seed*2).reshape(7, nGroup, 32)
    groupFeaturesX = torch.cat([ torch.Tensor(groupFeaturesM),torch.Tensor(groupFeaturesS)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1, 32)
    data['groupFeatures'] = groupFeatures

    return data

def addIPNoiseGroup(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']


    fixedPE = randPEs(nTrial=1,nSample=nGroup,maxN=nGroup,featureD=32,seed=seed)
    groupFeaturesS = torch.Tensor(fixedPE.repeat(nElement-5,axis=0))
    groupFeaturesM = randPEs(1, nGroup * 5, nGroup * 5, 32, seed=seed*2).reshape(5, nGroup, 32)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesS), torch.Tensor(groupFeaturesM)], dim=0)


    groupFeatures = torch.zeros(vf.shape[0], 32)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1,32)
    data['groupFeatures'] = groupFeatures

    return data


def addNoisePos(data,seed):

    # biInds = data['biInds']
    vf = data['varFeatures']
    # reorderInds = data['reorderInds'].reshape(-1).long()

    # index
    # random.seed(0)
    fixedPE = randPEs(1, vf.shape[0],  vf.shape[0], 32,seed)
    data['groupFeatures'] = torch.Tensor(fixedPE[0])

    return data


if __name__ == '__main__':

    d = randPEs(nTrial=10,nSample=10,maxN=10,featureD=32,seed=0)
    print('done')
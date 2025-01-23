import numpy as np
import random
import torch


def addEmpty(data,seed):
    vf = data['varFeatures']
    groupFeatures = torch.zeros(vf.shape[0], 1)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)
    return data

def randPEs(nTrial,nSample,maxN,seed):
    np.random.seed(seed)
    choice = np.arange(1, maxN + 1)

    # allPEs = getAllPEs(maxN,featureD)
    # inds = np.arange(0,maxN)
    allSamples = []
    for t in range(nTrial):
        # ts = np.random.choice(inds,nSample,replace=False)
        selectedJ = np.random.choice(choice, size=nSample, replace=False)
        allSamples.append(selectedJ)
    sampledPEs = np.array(allSamples)/maxN

    return sampledPEs


def addNoiseUniform(data,seed):

    vf = data['varFeatures']

    np.random.seed(seed)
    randV = np.random.rand(vf.shape[0],1)
    groupFeatures = torch.Tensor(randV)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)

    return data

def addBPNoiseOrbit(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(nElement - 7, nGroup, nGroup, seed=seed)
    groupFeaturesM = randPEs(1, nGroup * 7, nGroup * 7, seed=seed).reshape(7, nGroup)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesM),torch.Tensor(groupFeaturesS)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 1)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1,1)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)

    return data


def addIPNoiseOrbit(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(nElement - 5, nGroup, nGroup, seed=seed)
    groupFeaturesM = randPEs(1, nGroup * 5, nGroup * 5, seed=seed).reshape(5, nGroup)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesS),torch.Tensor(groupFeaturesM)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 1)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1,1)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)

    return data

def addBPNoiseGroup(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(1, nGroup, nGroup, seed=seed).repeat(nElement - 7,axis=0)
    groupFeaturesM = randPEs(1, nGroup * 7, nGroup * 7, seed=seed).reshape(7, nGroup)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesM), torch.Tensor(groupFeaturesS)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 1)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1,1)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)

    return data

def addIPNoiseGroup(data,seed):

    # biInds = data['biInds']
    reorderInds = data['reorderInds'].long().reshape(-1)
    vf = data['varFeatures']
    nElement = data['nElement']
    nGroup = data['nGroup']

    groupFeaturesS = randPEs(1, nGroup, nGroup, seed=seed).repeat(nElement - 5,axis=0)
    groupFeaturesM = randPEs(1, nGroup * 5, nGroup * 5, seed=seed).reshape(5, nGroup)
    groupFeaturesX = torch.cat([torch.Tensor(groupFeaturesS), torch.Tensor(groupFeaturesM)], dim=0)

    groupFeatures = torch.zeros(vf.shape[0], 1)
    groupFeatures[reorderInds] = groupFeaturesX.reshape(-1,1)
    data['varFeatures'] = torch.cat([vf,groupFeatures],dim=-1)


    return data


def addNoisePos(data,seed):

    # biInds = data['biInds']
    vf = data['varFeatures']
    # reorderInds = data['reorderInds'].reshape(-1).long()

    # index
    # random.seed(0)
    groupFeatures = randPEs(1, vf.shape[0],  vf.shape[0],seed).transpose()
    data['varFeatures'] = torch.cat([vf,torch.Tensor(groupFeatures)],dim=-1)

    return data


if __name__ == '__main__':

    d = randPEs(nTrial=10,nSample=10,maxN=10,featureD=32,seed=0)
    print('done')
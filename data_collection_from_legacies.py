import multiprocessing

import pyscipopt as scip
import os
import numpy as np
import gzip,pickle
import argparse
from feature_extract import extract_features


def getSolObjs(m):
    solDicts = m.getSols()
    objs = [m.getSolObjVal(sol) for sol in solDicts]
    vars = [va for va in m.getVars()]
    varTypes = [va.vtype() for va in vars]
    sols = np.zeros((len(solDicts), len(vars)))
    for i in range(len(solDicts)):
        for j in range(len(vars)):
            sols[i, j] = solDicts[i].__getitem__(vars[j])
            if varTypes[j] in ['BINARY','INTEGER'] :
                sols[i,j] = round(sols[i,j]) # round integer values


    return sols,objs

def collect(filepath,solpath,targetDir):

    insName = os.path.basename(filepath)

    # extract MILP graph
    varNames, variableFeatures, constraintFeatures, edgeInds, edgeWeights = extract_features(filepath)
    bpSaveDir = os.path.join(targetDir, 'bipartites')
    filename = os.path.join(bpSaveDir, f'{insName}.bp')
    with gzip.open(filename, "wb") as f:
        pickle.dump({
            'varNames': varNames,
            'variableFeatures': variableFeatures,
            'constraintFeatures': constraintFeatures,
            'edgeInds': edgeInds,
            'edgeWeights': edgeWeights
        }, f)


    # collect solutions
    m = scip.Model()
    m.hideOutput()
    m.readProblem(filepath)

    vars = m.getVars()
    varNames = [va.name  for va in vars]
    # varTypes = [va.vtype()  for va in vars]

    # varNames = np.array(varNames)
    # varTypes = np.array(varTypes)


    # get solutions
    solData = pickle.load(open(solpath,'rb'))
    solVarNames = solData['varNames'] if 'varNames' in solData.keys() else solData['var_names']
    # match  varNames
    inds = [ solVarNames.index(varn) for varn in varNames]
    checkNames = [solVarNames[ind] for ind in inds]
    if ''.join(varNames) != ''.join(checkNames):
        raise NotImplementedError
    solVarTypes = ['Unknown' for i in range(len(varNames))]
    # biInds = solData['varInds']
    # for biInd in biInds:
    #     solVarTypes[biInd] = 'BINARY'

    sols = solData['sols']
    objs = solData['objs']

    filename = os.path.join(targetDir, 'solutions', f'{insName}.sol')

    with gzip.open(filename, "wb") as f:
        pickle.dump({
            'varNames':varNames,
            'varTypes':solVarTypes,
            'sols': sols[:,inds],
            'objs': objs,
            'objSense': m.getObjectiveSense()
        }, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sourceDir', type=str, default=r'F:\L2O_project\automorphism\ICLR2025\data\LB_old', help='root directory')
    parser.add_argument('--targetDir', type=str, default=r'F:\L2O_project\automorphism\ICLR2025\data\LB',
                        help='root directory')
    parser.add_argument('--nWorkers', type=int, default=1, help='number of processes')
    args = parser.parse_args()
    NWORKER = args.nWorkers
    sourceDir = args.sourceDir
    targetDir = args.targetDir
    os.makedirs(os.path.join(targetDir, 'solutions'), exist_ok=True)
    os.makedirs(os.path.join(targetDir, 'bipartites'), exist_ok=True)
    insDir = os.path.join(sourceDir,'ins')
    solDir = os.path.join(sourceDir,'sol')
    insNameList = os.listdir(insDir)
    insPathList = [os.path.join(insDir, insName) for insName in insNameList]
    solPathList = [os.path.join(solDir, insName+'.sol') for insName in insNameList]
    with multiprocessing.Pool(processes=NWORKER) as pool:
        for insPath,solPath in zip(insPathList,solPathList):
            pool.apply_async(collect, (insPath,solPath, targetDir))

        pool.close()
        pool.join()

    print('done')
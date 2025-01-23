import os
import numpy as np
import re
import scipy.io as io
import pandas

rootDir = r'F:\L2O_project\automorphism\ICLR2025\formal_res\exp_BP_IP_SMSP_1001_b4'
# F:\L2O_project\automorphism\ICLR2025\formal_res\exp_BP_IP_SMSP_1001_b4\dataset-BP-Aug-orbit-opt-opt-epoch-100-sampleTimes-8
sampleTimes = 8
epoch=100
augs = ['empty','uniform','pos','orbit','group']
datasets = ['BP','IP','SMSP']
opt='none'

expList = os.listdir(rootDir)

handisTable = pandas.DataFrame()
lossTable = pandas.DataFrame()
handisData = []
lossData = []

for dataset in datasets:

    for aug in augs:

        expInfo = [dataset,aug]

        expName = fr'dataset-{dataset}-Aug-{aug}-opt-{opt}-epoch-{epoch}-sampleTimes-{sampleTimes}'
        filepath = os.path.join(rootDir,expName,'loss_record.mat')
        data = io.loadmat(filepath)

        valid_loss = data['valid_loss'][0]
        bestInd = valid_loss.argmin()
        valid_handis = list(data['valid_handis'][bestInd])

        handisData.append(expInfo + valid_handis)


handisTable = pandas.DataFrame(handisData)


handisTable.to_excel(os.path.join(rootDir,'handisTable_valid_none.xlsx'))

print('done')

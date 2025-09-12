import os
import numpy as np
import re
import scipy.io as io
import pandas

rootDir = r'./exp'
sampleTimes = 8
epoch=100
augs = ['empty','uniform','pos','orbit','group']
datasets = ['BPP','BIP','SMSP']
opt='opt'

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


handisTable = pandas.DataFrame(handisData, columns=["Dataset", "Method", "Top-10%", "Top-20%", "Top-30%", "Top-40%", "Top-50%", "Top-60%", "Top-70%", "Top-80%", "Top-90%", "Top-100%"])


handisTable.to_excel(os.path.join(rootDir,'handisTable_valid.xlsx'))

print('done')

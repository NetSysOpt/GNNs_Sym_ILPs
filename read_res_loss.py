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
opt='opt'

expList = os.listdir(rootDir)

lossTable = pandas.DataFrame()

lossData = []

for dataset in datasets:

    for aug in augs:

        expInfo = [dataset,aug]

        expName = fr'dataset-{dataset}-Aug-{aug}-opt-{opt}-epoch-{epoch}-sampleTimes-{sampleTimes}'
        filepath = os.path.join(rootDir,expName,'loss_record.mat')
        data = io.loadmat(filepath)


        train_loss = data['train_loss'][0].min()
        valid_loss = data['valid_loss'][0].min()


        lossData.append(expInfo + [train_loss,valid_loss])


lossTable = pandas.DataFrame(lossData)


lossTable.to_excel(os.path.join(rootDir,'lossTable_opt.xlsx'))

print('done')

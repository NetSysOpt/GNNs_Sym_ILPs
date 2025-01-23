from utils import *
from feature_aug import *
import os

confInfo = {
'BP': {
    'name':'BP',
    'trainDir':r'data/BPP/nItem20_s500/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderBP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addBPNoiseOrbit,
        'group':addBPNoiseGroup
    }
},

'SMSP':{
    'name':'SMSP',
    'trainDir':r'data/SMSP/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderSMSP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addSMSPNoiseOrbit,
        'group':addSMSPNoiseGroup
    }
},


'IP':{
    'name':'IP',
    'trainDir':r'data/IP/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderIP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addIPNoiseOrbit,
        'group':addIPNoiseGroup
    }

},

'BP10': {
    'name':'BP10',
    'trainDir':r'data/BPP/nItem20_s10/train',
    'testDir':r'',
    'nGroup':32,
    'reorder':reorderBP,
    'featureAugFuncs':{
        'empty':addEmpty,
        'uniform':addNoiseUniform,
        'pos':addNoisePos,
        'orbit':addBPNoiseOrbit,
        'group':addBPNoiseGroup
    }
},

}




#
# info = ipTuneInfo
#
# DIR_INS = os.path.join(info['trainDir'],'ins')
# DIR_SOL = os.path.join(info['trainDir'],'sol')
# DIR_BG = os.path.join(info['trainDir'],'bg')
# NGROUP = info['nGroup']
#
# TEST_INS = os.path.join(info['testDir'],'ins')
# TEST_BG = os.path.join(info['testDir'],'bg')
#
# ADDPOS = info['addPosFeature']
# REORDER = info['reorder']
#
#
# DATA_NAME = info['name']
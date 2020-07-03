import tensorflow as tf

# state-of-the-arte semantic segmentation approaches

from ICNet import ICNet
from ICNet_BN import ICNet_BN

# my models for ICRA 2019 (LSTM_ICNet_version*)

from LSTM_ICNet_v1 import LSTM_ICNet_v1
from LSTM_ICNet_v2 import LSTM_ICNet_v2
from LSTM_ICNet_v3 import LSTM_ICNet_v3
from LSTM_ICNet_v4 import LSTM_ICNet_v4
from LSTM_ICNet_v5 import LSTM_ICNet_v5
from LSTM_ICNet_v6 import LSTM_ICNet_v6

# skip models

from Faster_LSTM_ICNet_v2 import Faster_LSTM_ICNet_v2
from Faster_LSTM_ICNet_v5 import Faster_LSTM_ICNet_v5
from Faster_LSTM_ICNet_v6 import Faster_LSTM_ICNet_v6



###################################################################################
###                         model config                                        ###
###################################################################################

model_config = {
    # state-of-the-arte semantic segmentation approaches
    'ICNet': ICNet, 
    'ICNet_BN': ICNet_BN,

    # LSTM-ICNet network-architectures of IV 2019 (LSTM_ICNet_v*)
    'LSTM_ICNet_v1': LSTM_ICNet_v1,
    'LSTM_ICNet_v2': LSTM_ICNet_v2,
    'LSTM_ICNet_v3': LSTM_ICNet_v3,
    'LSTM_ICNet_v4': LSTM_ICNet_v4,
    'LSTM_ICNet_v5': LSTM_ICNet_v5,
    'LSTM_ICNet_v6': LSTM_ICNet_v6,

    # Faster-LSTM-ICNet network-architectures of ITSC 2020 (Faster_LSTM_ICNet_v*)
    'Faster_LSTM_ICNet_v2': Faster_LSTM_ICNet_v2,
    'Faster_LSTM_ICNet_v5': Faster_LSTM_ICNet_v5,
    'Faster_LSTM_ICNet_v6': Faster_LSTM_ICNet_v6,

    } 


def get_model(model_name):

    model = model_config[model_name]
    
    return model

from train import run_banchmark
from dotmap import DotMap
from utils import AttnLabelConverter

def_params = {
    "batch_size": 128,
    "train_data": "data_lmdb_release/training",
    "select_data": "MJ-ST",
    "batch_ratio": "0.5-0.5",
    "workers": 28,
    "batch_max_length": 25,
    "total_data_usage_ratio": 0.01,
    "Transformation": "TPS",
    "FeatureExtraction": "ResNet",
    "SequenceModeling": "BiLSTM",
    "Prediction": "Attn",
    "adam": True,
    # train_data
    # valid_data
    "beta1": 0.009,
    "grad_clip": 5,
    "imgH": 32,
    "imgW": 100,
    "rgb": False,
    "character": "0123456789abcdefghijklmnopqrstuvwxyz",
    "sensitive": False,
    "PAD": True,
    "data_filtering_off": False,
    'rho': 0.95,
    'FT': False,
    'valid_data': 'data_lmdb_release/validation',
    'num_iter': 300000,
    'saved_model': '',
    'eps': 1e-08,
    "input_channel": 1,
    'valInterval': 2000,
    'exp_name': None

}

params = [
    {
        "lr": 0.001584893192461114,
        "num_fiducial": 20,
        "output_channel": 512,
        "hidden_size": 256,
        "manualSeed": 1111
    },
    {
        "lr": 0.001584893192461114,
        "num_fiducial": 40,
        "output_channel": 256,
        "hidden_size": 512,
        "manualSeed": 1111
    }

]


for i, hparam in enumerate(params):
    merged = {**hparam, **def_params}
    print(f"----- RUN #{i} START ------")
    print(merged)
    dotted = DotMap(merged)
    converter = AttnLabelConverter(dotted.character)

    dotted.num_class = len(converter.character)
    dotted.select_data = dotted.select_data.split('-')
    dotted.batch_ratio = dotted.batch_ratio.split('-')
    run_banchmark(dotted)
    print(f"----- RUN #{i} END ------")

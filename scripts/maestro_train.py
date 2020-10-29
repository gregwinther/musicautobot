import sys

sys.path.insert(0, "..")

from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.utils.midifile import *
from musicautobot.config import *
from musicautobot.music_transformer import *


midi_path = Path("../data/midi/maestro_all")
data_path = Path("../data/numpy")
data_save_name = "maestro_all_data_save.pkl"

lr = 1e-4
epochs = 8
bptt = 512
batch_size = 16
encode_position = True
dl_tfms = [batch_position_tfm] if encode_position else []

data = load_data(
    data_path,
    data_save_name,
    bs=batch_size,
    bptt=bptt, 
    ecode_position=encode_position,
    dl_tfms=dl_tfms,
)

config = default_config()
config["encode_position"] = encode_position
learn = music_model_learner(data, config=config.copy())

for i, guy in enumerate(data.train_ds):
    if guy[0].data.max() >= len(learn.data.vocab):
        print(f"Vocab error: {i:3} {guy[0].data.max():4}") 

import warnings

warnings.simplefilter("ignore", UserWarning)

learn.fit_one_cycle(epochs, lr)

learn.save("maestro_all_model")

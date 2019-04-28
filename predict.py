import keras
from keras.models import load_model
from keras import backend as K
from Generators import KerasBatchGenerator
import os, json
import numpy as np
from util import getPseudoHash

model_path = "/Users/paulbiberstein/Desktop/FormattedChordDatasets/model/final_model.hdf5"
default_reversed_dictionary_path = "/Users/paulbiberstein/Desktop/FormattedChordDatasets/defaultReverseDict.json"
hashed_reversed_dictionary_path = "/Users/paulbiberstein/Desktop/FormattedChordDatasets/hashedReverseDict.json"

with open(default_reversed_dictionary_path, "r") as f:
    default_reversed_dictionary = json.load(f)
with open(hashed_reversed_dictionary_path, "r") as f:
    hashed_reversed_dictionary = json.load(f)

train_path = "/Users/paulbiberstein/Desktop/FormattedChordDatasets/RealBookTrainDataset.txt"
with open(train_path, "r") as f:
    train_data = json.load(f)


model = load_model(model_path)
# print(model.summary())

dummy_iters = 0
num_steps = 5
batch_size = 20
features = 12
example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, features,
                                                 skip_step=1)

for i in range(dummy_iters):
    dummy = next(example_training_generator.generate())
num_predict = 1
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "

# print(data)
# for chord in data:
#     myVar = getPseudoHash(chord)
#     print(hashed_reversed_dictionary[myVar])

for i in range(num_predict):
    data = next(example_training_generator.generate())[0]

    print(list( map( (lambda x : hashed_reversed_dictionary.get(getPseudoHash(x), "<unk>") ) , data[0]) ) ) 
    print("rawdata: " + str(data[0][0]))

    prediction = model.predict(data)
    predNoteArrays = (prediction > 0.5).astype(np.int)[0]
  
    for noteArray in predNoteArrays:
        try:
            chord = reversed_dictionary[str(list(noteArray))]
            pred_print_out += " " + str(chord)
        except KeyError:
            print("Could not find chord that corresponds to: " + str(list(noteArray)))
            pred_print_out += " <unk>"

    trueNoteArrays = train_data[dummy_iters + i : num_steps + dummy_iters + i]
    for noteArray in trueNoteArrays:
        try:
            chord = reversed_dictionary[str(list(noteArray))]
            true_print_out += " " + str(chord)
        except KeyError:
            print("Could not find chord that corresponds to\n" + str(list(noteArray)))
            true_print_out += " <unk>"
  
    # true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
    # pred_print_out += reversed_dictionary[predict_word] + " "
print(true_print_out)
print(pred_print_out)

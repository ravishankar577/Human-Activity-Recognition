import os
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

#configuration to hide warnings in terminal
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#load saved label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

#load saved scaler for normalizing data
scaler = pickle.load(open('scaler.pkl', 'rb'))

#load saved model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights("model.h5")

#read input data file
input_data = pd.read_csv("input_data.csv")
input_data = input_data.to_numpy()

#scale the input data
input_data = scaler.transform(input_data)

#make prediction using the loaded neural network model
prediction = encoder.inverse_transform([np.argmax(loaded_model.predict(input_data),axis=1)[0]])
print(prediction[0])

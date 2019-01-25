import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataframe = pd.read_csv("logs/norm_flight_data.csv", sep=";")
dataset = dataframe.values
# split into input (X) and output (Y) variables
Y = dataset[:,0:4]
X = dataset[:,4:7]

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(16, input_dim=3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(8, kernel_initializer='normal', activation='relu'))
	model.add(Dense(4, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = baseline_model()
model.fit(X, Y, epochs=50, batch_size=10, verbose=1)
print("ANN trained")

model.save("model_from_pid.h5")
print("Saved model to disk")
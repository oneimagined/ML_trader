import os
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import random
from keras.callbacks import TensorBoard


class VolumeProfileTrainer:
    def __init__(self):
        self.model = None

    def readTrainingData(self, file_path):
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            date = None
            label = None
            price = None
            profiles = []
            gradient = []

            row = next(reader)
            while row:
                if row[0] == 'Date':
                    row = next(reader)
                    date = row[0]
                    label = row[1]
                    price = float(row[2])
                elif row[0].startswith('Volume Profile VolumeProfile['):
                    profile = []
                    profiles.append(profile)
                    row = next(reader)
                    while row and not row[0].startswith('Volume'):
                        if len(row)>1:
                            profile.append([float(val) for val in row])
                        row = next(reader)
                    continue
                elif row[0] == 'Volume Gradient':
                    row = next(reader)
                    while row:
                        gradient.append([float(row[0]), float(row[1])])
                        try:
                            row = next(reader)
                        except StopIteration:
                            break
                else:
                    pass
                try:
                    row = next(reader)
                except StopIteration:
                    break


        return label, price, profiles, gradient    # (the original readTrainingData function code goes here)

    def prepare_vpdata_for_ann(self, instruction, trading, base):
        max_modes = 5
        vp_instruction = []
        for idx in range(0,max_modes):
            if idx < len(instruction):
                vp_instruction.append(instruction[idx])
            else:
                vp_instruction.append([0,0,0,0])
        vp_trading = []
        for idx in range(0,max_modes):
            if idx < len(trading):
                vp_trading.append(trading[idx])
            else:
                vp_trading.append([0,0,0,0])
        vp_base = []
        for idx in range(0,max_modes):
            if idx < len(base):
                vp_base.append(base[idx])
            else:
                vp_base.append([0,0,0,0])
        return vp_instruction, vp_trading, vp_base

    def prepare_gradient_statistics_for_ann(self, gradient):
        bid_gradient = np.array(gradient)[:, 0]
        offer_gradient = np.array(gradient)[:, 1]
        bid_average = np.average(bid_gradient)
        offer_average = np.average(offer_gradient)
        bid_stddev = np.std(bid_gradient)
        offer_stddev = np.std(offer_gradient)
        return [bid_average, bid_stddev, offer_average, offer_stddev]

    def extract_labels_and_data(self, trade_direction, label, price, profiles, gradient):

        if trade_direction>0:
            if label == 'long':
                label = [1, 0]   # Trade long
            else:
                label = [0, 1]  # Dont Trade long
        else:
            if label == 'short':
                label = [1,0] # Trade short
            else:
                label = [0,1] # Dont Trade short


        gradient_stats = self.prepare_gradient_statistics_for_ann(gradient)
        vp_instruction, vp_trading, vp_base = self.prepare_vpdata_for_ann(profiles[0], profiles[1], profiles[2])

        return label, price, [vp_instruction, vp_trading, vp_base], gradient_stats

    def prep_data_for_processing(self,trade_direction,  train_data_dir='../Training/'):
        # Get a list of all the training data files in the directory
        train_data_paths = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.csv')]
        random.shuffle(train_data_paths)
        # label, features = readTrainingData('../Training/long-{fc12f2f0-c19d-11ed-9736-8e6e30f8fc30}.csv')
        X_train_price = []
        X_train_vp = []
        X_train_gradient=[]
        Y_train = []

        for path in train_data_paths:
            label, price, profiles, gradient = self.readTrainingData(path)

            label, price, vp_data, gradient_stats = self.extract_labels_and_data(trade_direction, label, price, profiles, gradient)

            Y_train.append(label)
            X_train_vp.append(vp_data)
            X_train_gradient.append(gradient_stats)
            X_train_price.append(price)

        X_train = self.normalise(X_train_price, X_train_vp, X_train_gradient)
        return X_train, np.array(Y_train)

    def normalise(self,X_train_price, X_train_vp, X_train_gradient):
        # normalise the price across all volme profiles
        X_train = []

        X_train_vp=np.array(X_train_vp)
        min_val = np.min(X_train_vp[:, :, :, 0][X_train_vp[:, :, :, 0] > 0])
        max_val = np.max(X_train_vp[:, :, :, 0])

        range_val = max_val - min_val
        if range_val>0:
            X_train_vp[:, :, :, 0] = X_train_vp[:, :, :, 0] - min_val
            X_train_vp[:, :, :, 0] = X_train_vp[:, :, :, 0] / range_val
            X_train_vp[:, :, :, 0][X_train_vp[:, :, :, 0] < 0] = 0

            X_train_price = np.array(X_train_price)
            X_train_price = X_train_price - min_val
            X_train_price = X_train_price / range_val
            X_train_price[X_train_price < 0] = 0

        else:
            X_train_vp[:, :, :, 0] = 0
            X_train_price = 0

        # normalise the standard distribution and weight per trading profile

        for idx in range(0, X_train_vp.shape[1]):
            for n in range(1,4):
                min_val = np.min(X_train_vp[:, idx, :, n][X_train_vp[:, idx, :, n] > 0])
                max_val = np.max(X_train_vp[:, idx, :, n])
                range_val = max_val - min_val
                if range_val>0:
                    X_train_vp[:, idx, :, n] = X_train_vp[:, idx, :, n] - min_val
                    X_train_vp[:, idx, :, n] = X_train_vp[:, idx, :, n] / range_val
                    X_train_vp[:, idx, :, n][X_train_vp[:, idx, :, n] < 0] = 0
                else:
                    X_train_vp[:, idx, :, n] = 0


        # normalise gradient per feature
        X_train_gradient = np.array(X_train_gradient)
        for idx in range(0, X_train_gradient.shape[0]):
            max_val = np.max([X_train_gradient[idx, 0],X_train_gradient[idx, 2]])
            min_val = np.min([X_train_gradient[idx, 0],X_train_gradient[idx, 2]])
            range_val = max_val - min_val
            if range_val>0:
                X_train_gradient[idx, 0] = X_train_gradient[idx, 0] - min_val/ range_val
                X_train_gradient[idx, 2] = X_train_gradient[idx, 2] - min_val/ range_val
                max_val = np.max([X_train_gradient[idx, 1],X_train_gradient[idx, 3]])
                min_val = np.min([X_train_gradient[idx, 0],X_train_gradient[idx, 2]])
                range_val = max_val - min_val
                X_train_gradient[idx, 1] = X_train_gradient[idx, 1] - min_val/ range_val
                X_train_gradient[idx, 3] = X_train_gradient[idx, 3] - min_val/ range_val
            else:
                X_train_gradient[idx]=0

        for idx in range(0, X_train_price.shape[0]):
            feature = []
            feature.append(X_train_price[idx])
            for vp_idx in range(0, X_train_vp.shape[1]):
                for vp_dist_idx in range(0, X_train_vp.shape[2]):
                    for vp_dist_stats_idx in range(0, X_train_vp.shape[3]):
                        feature.append(X_train_vp[idx,vp_idx,vp_dist_idx,vp_dist_stats_idx])
            #for grad_idx in range(0, 4):
            #    feature.append(X_train_gradient[idx][grad_idx])
            X_train.append(feature)


        X_train = np.array(X_train)
        return X_train


    def train(self, trade_direction, X_train, Y_train, hidden_layers=(30, 20, 8), epochs=50, batch_size=32, validation_split=0.1):
        num_inputs = len(X_train[0])
        num_outputs = len(Y_train[0])

        # Define the neural network architecture
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=num_inputs, activation='relu'))
        for layer in range(1, len(hidden_layers)):
            self.model.add(Dense(hidden_layers[layer], activation='relu'))
        self.model.add(Dense(num_outputs, activation='softmax'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        if trade_direction > 0:
            self.model.save(f'buy-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')
        else:
            self.model.save(f'sell-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')

    def evaluate(self, trade_direction, X_test, Y_test, hidden_layers=(30, 20, 8)):
        if trade_direction > 0:
            self.model = load_model(f'buy-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')
        else:
            self.model = load_model(f'sell-{hidden_layers[0]}-{hidden_layers[1]}-{hidden_layers[2]}.h5')

        loss, accuracy = self.model.evaluate(X_test, Y_test, batch_size=32)
        return loss, accuracy

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)


trainer = VolumeProfileTrainer()
trade_direction = 1
X_train, Y_train = trainer.prep_data_for_processing(trade_direction, '../training/')

hidden_layers= (10, 40, 12)
trainer.train(trade_direction, X_train, Y_train, hidden_layers)

X_test, Y_test = trainer.prep_data_for_processing(trade_direction, '../training_set2/')
loss, accuracy = trainer.evaluate(trade_direction, X_test, Y_test, hidden_layers)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


print('Bla')

failcount=0
successcount = 0
threshold = 0.9
for idx in range(0, len(X_test)):
    output = trainer.predict(np.array([X_test[idx]]))
    output = np.array(output)
    output_binary = (output > threshold).astype(int)
    if np.all(np.logical_xor(Y_test[idx], output_binary[0])):
        print(Y_test[idx], output_binary, output)
        failcount+=1
    if np.array_equal(Y_test[idx], output_binary[0]):
        successcount+=1

print('Long: number of correct classifications ', successcount, ' and misclassifications', failcount, 'of ', len(X_test))

trade_direction = -1
X_train, Y_train = trainer.prep_data_for_processing(trade_direction, '../training/')

trainer.train(trade_direction, X_train, Y_train, hidden_layers)

X_test, Y_test = trainer.prep_data_for_processing(trade_direction, '../training_set2/')
loss, accuracy = trainer.evaluate(trade_direction, X_test, Y_test, hidden_layers)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


failcount=0
successcount = 0
for idx in range(0, len(X_test)):
    output = trainer.predict(np.array([X_test[idx]]))
    output = np.array(output)
    output_binary = (output > threshold).astype(int)
    if np.all(np.logical_xor(Y_test[idx], output_binary[0])):
        print(Y_test[idx], output_binary, output)
        failcount+=1
    if np.array_equal(Y_test[idx], output_binary[0]):
        successcount+=1

print('Short: number of correct classifications ', successcount, ' and misclassifications', failcount, 'of ', len(X_test))

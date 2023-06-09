from VolumeProfileANN import VolumeProfileANN
import numpy as np


trainer = VolumeProfileANN()
trade_directions = [-1, 1]
trade_name = ['sell','buy']

for idx in range(0, 2):
    trade_direction = trade_directions[idx]
    trainer.load_features(trade_direction, '../training/', 'Train')


    X_train = np.array(trainer.generate_features())
    Y_train = np.array(trainer.Y_train)

    hidden_layers = (10, 40, 8)

    trainer.train(trade_direction, X_train, trainer.Y_train, hidden_layers, epochs=400)

    trainer.load_features(trade_direction, '../testing/')

    X_test = np.array(trainer.generate_features())
    Y_test = np.array(trainer.Y_train)

    missed_count = 0
    losing_count = 0
    success_count = 0
    threshold = 0.9
    for i in range(0, len(X_test)):
        output = trainer.predict(np.array([X_test[i]]))
        output = np.array(output)

        output_binary = (output > threshold).astype(int)
        if output[0][0] > output[0][1] and Y_test[i][0] == 0 and output[0][0] > threshold:
            print('Losing Classifications ', Y_test[i], output)
            losing_count += 1
        if output[0][0] < output[0][1] and Y_test[i][0] == 1 and output[0][0] > threshold:
            print('Missed Classifications ', Y_test[i], output)
            missed_count += 1
        if output[0][0] > output[0][1] and Y_test[i][0] == 1 and output[0][0] > threshold:
            print('Success Classifications ', Y_test[i], output)
            success_count += 1

    print(trade_name[idx], ': number of losing classifications ', losing_count,
          ' and missed classifications', missed_count,
          'and success classifications ', success_count,
          'of ', len(X_test))

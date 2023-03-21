from CommonClasses import VolumeProfileANN
import numpy as np


trainer = VolumeProfileANN()
trade_directions = [-1, 1]

for idx in range(0, 2):
    trade_direction = trade_directions[idx]
    trainer.load_features(trade_direction, '../training/')

    X_train, Y_train = trainer.create_normalised_features(trainer.X_train_price, trainer.X_train_vp, trainer.X_train_gradient, trainer.Y_train)

    hidden_layers = (10, 40, 10)
    hidden_layers = (40, 20, 10)
    trainer.train(trade_direction, X_train, Y_train, hidden_layers)

    trainer.load_features(trade_direction, '../testing/')
    X_test, Y_test = trainer.create_normalised_features(trainer.X_train_price, trainer.X_train_vp, trainer.X_train_gradient, trainer.Y_train)

    loss, accuracy = trainer.evaluate(trade_direction, X_test, Y_test, hidden_layers)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    failcount = 0
    successcount = 0
    threshold = 0.7
    for i in range(0, len(X_test)):
        output = trainer.predict(np.array([X_test[i]]))
        output = np.array(output)
        output_binary = (output > threshold).astype(int)
        if np.all(np.logical_xor(Y_test[i], output_binary[0])):
            print(Y_test[i], output_binary, output)
            failcount += 1
        if np.array_equal(Y_test[i], output_binary[0]):
            successcount += 1

    print(str(trade_direction), ': number of correct classifications ', successcount, ' and misclassifications', failcount, 'of ', len(X_test))

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small

if __name__ == '__main__':
    go_board_rows, go_board_cols = 19,19
    num_classes = go_board_rows * go_board_cols
    #num_games = 20000
    num_games = 100
    #num_games = 8000
    #num_games = 1000

    encoder = OnePlaneEncoder((go_board_rows,go_board_cols))

    processor = GoDataProcessor(encoder=encoder.name())

    generator = processor.load_go_data('train',num_games,use_generator=True)
    test_generator = processor.load_go_data('test',num_games,use_generator=True)

    # g = generator.generate()
    # count = 0
    # for row in g:
    #     # row[0].shape (128,19,19,1) NWHC, channels last
    #     # this is input to model, it retreives via the generator
    #     # row[1].shape (128,361) categorical output, 1 move on the 19x19=361 space board
    #     # this is the ground truth label output for the given input
    #     #so 19x19x1 input (in 128 batches) with an output of 361
    #     # 19x19x1 -> 361, one-hot
    #     print("type(row): {}, type(row[0]): {}, type(row[1]): {}".format(type(row),type(row[0]),type(row[1])))
    #     #print("row: {}".format(row))
    #     print("shape 0: {}, shape 1: {}".format((row[0]).shape,(row[1]).shape))
    #     count += 1
    #     if count > 2:
    #         break

    #input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    epochs = 5
    batch_size = 128
    print("num samples: {}, batch_size: {}, num/batch: {}".format(generator.get_num_samples(),batch_size,generator.get_num_samples() / batch_size))
    #model.compile("sgd","mean_squared_error")
    model.compile(loss="categorical_crossentropy",optimizer="sgd",
                    metrics=['accuracy'])
    model.fit(generator.generate(batch_size, num_classes),
                    epochs=epochs,
                    steps_per_epoch=generator.get_num_samples() / batch_size,
                    #steps_per_epoch=12288,
                    validation_data=test_generator.generate(batch_size, num_classes),
                    validation_steps=test_generator.get_num_samples() / batch_size,
                    callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')])
    model.evaluate(test_generator.generate(batch_size, num_classes),
                    steps=test_generator.get_num_samples() / batch_size)
                    #steps=12288)

    # model.fit_generator(generator=generator.generate(batch_size, num_classes),
    #                 epochs=epochs,
    #                 steps_per_epoch=generator.get_num_samples() / batch_size,
    #                 validation_data=test_generator.generate(batch_size, num_classes),
    #                 validation_steps=test_generator.get_num_samples() / batch_size,
    #                 callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')])
    # model.evaluate_generator(generator=test_generator.generate(batch_size, num_classes),
    #                 steps=test_generator.get_num_samples() / batch_size)

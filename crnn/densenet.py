from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.layers import Bidirectional, LSTM, GRU

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal',
               padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)

    '''rnnunit = 256
    x = Bidirectional(GRU(rnnunit,return_sequences=True), name='blstm1')(x)
    x = Dense(rnnunit, name='blstm1_out', activation='linear')(x)
    x = Bidirectional(GRU(rnnunit,return_sequences=True), name='blstm2')(x)'''
    '''rnnunit = 256
    x = GRU(rnnunit,return_sequences=True)(x)'''

    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred


def dense_blstm(input):
    pass


'''input = Input(shape=(72, 272, 1), name='the_input')
dense_cnn(input, 5000)'''


def get_Model_noLSTM(training=False, shape=(80, 320, 1)):
    # crnn_weight = os.path.join('./crnn/models/crnn.h5')
    input = Input(shape=shape, name='the_input')
    y_pred = dense_cnn(input, 34)
    # crnn_model = Model(inputs=input, outputs=y_pred)
    # crnn_graph = tf.get_default_graph()
    # with crnn_graph.as_default():
    #     crnn_model.load_weights(crnn_weight)
    # output = cr

    # labels = Input(name='the_labels', shape=[9], dtype='float32')  # (None ,8)
    # input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
    # label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)
    #
    # # Keras doesn't currently support loss funcs with extra parameters
    # # so CTC loss is implemented in a lambda layer
    # loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    #     [y_pred, labels, input_length, label_length])  # (None, 1)
    #
    # if training:
    #     return Model(inputs=[input, labels, input_length, label_length],
    #                  outputs=loss_out)
    # else:
    return Model(inputs=[input], outputs=y_pred)


if __name__ == '__main__':
    model_t = get_Model_noLSTM(False, shape=(80, 320, 1))
    # model = get_Model_noLSTM()
    # crnn_weight = './crnn/models/crnn.h5'
    crnn_weight = './models/CNN--40--5.692.hdf5'
    # model_t.layers[112].name = 'out34'
    model_t.load_weights(crnn_weight, by_name=True)
    print(model_t)

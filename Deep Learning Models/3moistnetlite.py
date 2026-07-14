# Optimum model from hyperparameter tuning
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomTranslation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def MoistNetLite(learning_rate, num_filters, num_layers, dropout_rate, dense_layer_size):
    input_shape = (224, 224, 3)

    # Define input layer
    inputs = Input(shape=input_shape, name='input_1')

    # Normalize inputs
    #x = Normalization(mean=0.0, variance=255.0, name='normalization')(inputs)

    # Add random translation to inputs
    x = RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect', name='random_translation')(inputs)

    for i in range(num_layers):
        if i==2:
            x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name=f'conv2d_{i}_0')(x)
            #x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name=f'conv2d_{i}_1')(x)
        else:
            x = Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', name=f'conv2d_{i}_0')(x)
            #x = Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', name=f'conv2d_{i}_1')(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f'max_pooling2d_{i}')(x)
        x = Dropout(rate=dropout_rate, name=f'dropout_{i}')(x)

    # Global average pooling layer
    x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)

    # Dense layer
    x = Dense(units=dense_layer_size, activation='relu', name='dense')(x)

    # Output layer
    # outputs = Dense(units=3, activation=softmax, name='classification_head_1')(x)
    outputs = Dense(units=3, activation=softmax, name='classification_head_2')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs, name='model')

    # Define optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', precision_m, recall_m, f1])

    return model

# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam


def fit(model, X_train, y_train, loss='sparse_categorical_crossentropy', optimizer=SGD, lr=0.001, batch_size=100,
        lr_scheduler=True, early_stopping=True, patience=7, no_epochs=30, validation_split=0.2):
    '''
        Compile and Train Model
        
        Param:
            model: Model that should be trained
            X_train: Training Data
            y_train: Train labels
            loss: Loss function (default categorical_crossentropy)
            optimizer: Optimizer for the training (default SGD (other option is Adam))
            lr: Learning Rate (default 0.001)
            lr_scheduler: Boolean if a Learning Rate Scheduler should be used (default True)
            early_stopping: Boolean if early stopping should be used (default True)
            patience: If early_stopping is true, than patience is the number of epochs that should be waited before stopping (default 7)
            no_epochs: Number of epochs
            class_weights: Weights for the classes
        
        Return:
            history: model_loss and model_accuracy for the train and val data
            model: Trained model
    '''
    # Compile Model
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=['accuracy'])

    if lr_scheduler:
        # Initialize Learning Rate Scheduler
        lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        if early_stopping:
            # Initialize Early Stopping
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

            # Train Model with lr_scheduler und early stopping
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=no_epochs,
                                verbose=1,
                                validation_split=validation_split,
                                callbacks=[early_stopping_cb, lr_callback])

        else:
            # Train Model with lr_scheduler
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=no_epochs,
                                verbose=1,
                                validation_split=validation_split,
                                callbacks=[lr_callback])
    else:
        # Train Model
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=no_epochs,
                            verbose=1,
                            validation_split=validation_split)

    return history, model

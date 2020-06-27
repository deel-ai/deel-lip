Demo 3: HKR classifier on MNIST dataset
=======================================

This notebook will demonstrate learning a binary task on the MNIST0-8
dataset.

.. code:: ipython3

    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import ReLU
    from tensorflow.keras.optimizers import Adam

    from deel.lip.layers import SpectralConv2D, SpectralDense, FrobeniusDense
    from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
    from deel.lip.utils import load_model
    from deel.lip.losses import HKR_loss, KR_loss, hinge_margin_loss

    from model_samples.model_samples import get_lipMLP, get_lipVGG_model

data preparation
----------------

For this task we will select two classes: 0 and 8. Labels are changed to
{-1,1}, wich is compatible with the Hinge term used in the loss.

.. code:: ipython3

    from tensorflow.keras.datasets import mnist

    # first we select the two classes
    selected_classes = [0, 8]  # must be two classes as we perform binary classification

    def prepare_data(x, y, class_a=0, class_b=8):
        """
        This function convert the MNIST data to make it suitable for our binary classification
        setup.
        """
        # select items from the two selected classes
        mask = (y==class_a)+(y==class_b)  # mask to select only items from class_a or class_b
        x=x[mask]
        y=y[mask]
        x=x.astype('float32')
        y=y.astype('float32')
        # convert from range int[0,255] to float32[-1,1]
        x/=255
        x=x.reshape((-1,28,28,1))
        # change label to binary classification {-1,1}
        y[y==class_a] = 1.0
        y[y==class_b] = -1.0
        return x, y

    # now we load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # prepare the data
    x_train, y_train = prepare_data(x_train, y_train, selected_classes[0], selected_classes[1])
    x_test, y_test = prepare_data(x_test, y_test, selected_classes[0], selected_classes[1])

    # display infos about dataset
    print("train set size: %i samples, classes proportions: %.3f percent" %
          (y_train.shape[0], 100*y_train[y_train==1].sum()/y_train.shape[0]))
    print("test set size: %i samples, classes proportions: %.3f percent" %
          (y_test.shape[0], 100*y_test[y_test==1].sum()/y_test.shape[0]))


.. parsed-literal::

    train set size: 11774 samples, classes proportions: 50.306 percent
    test set size: 1954 samples, classes proportions: 50.154 percent


Build lipschitz Model
---------------------

Let’s first explicit the paremeters of this experiment

.. code:: ipython3

    # training parameters
    epochs=5
    batch_size=128

    # network parameters
    hidden_layers_size = [128,64,32]
    activation = GroupSort #ReLU, MaxMin, GroupSort2

    # loss parameters
    min_margin=1
    alpha = 10

Now we can build the network. Here the experiment is done with a MLP.
But ``Deel-lip`` also provide state of the art 1-Lipschitz convolutions.

.. code:: ipython3

    K.clear_session()
    # helper function to build the 1-lipschitz MLP
    wass=get_lipMLP((28,28,1), hidden_layers_size = hidden_layers_size ,activation=activation, nb_classes = 1,kCoefLip=1.0)
    # an other helper function exist to build a VGG model
    # wass=get_lipVGG_model((28,28,1),layers_conv=[32,64],layers_dense=[128],activation_conv=GroupSort2,activation_dense=FullSort,use_bias=True , nb_classes = 1, last_activ = None)
    wass.summary()


.. parsed-literal::

    128
    64
    32
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 1)]       0
    _________________________________________________________________
    flatten (Flatten)            (None, 784)               0
    _________________________________________________________________
    spectral_dense (SpectralDens (None, 128)               100609
    _________________________________________________________________
    group_sort (GroupSort)       (None, 128)               0
    _________________________________________________________________
    spectral_dense_1 (SpectralDe (None, 64)                8321
    _________________________________________________________________
    group_sort_1 (GroupSort)     (None, 64)                0
    _________________________________________________________________
    spectral_dense_2 (SpectralDe (None, 32)                2113
    _________________________________________________________________
    group_sort_2 (GroupSort)     (None, 32)                0
    _________________________________________________________________
    frobenius_dense (FrobeniusDe (None, 1)                 33
    =================================================================
    Total params: 111,076
    Trainable params: 110,849
    Non-trainable params: 227
    _________________________________________________________________


.. code:: ipython3

    optimizer = Adam(lr=0.01)

.. code:: ipython3

    # as the output of our classifier is in the real range [-1, 1], binary accuracy must be redefined
    def HKR_binary_accuracy(y_true, y_pred):
        S_true= tf.dtypes.cast(tf.greater_equal(y_true[:,0], 0),dtype=tf.float32)
        S_pred= tf.dtypes.cast(tf.greater_equal(y_pred[:,0], 0),dtype=tf.float32)
        return binary_accuracy(S_true,S_pred)

.. code:: ipython3

    wass.compile(
        loss=HKR_loss(alpha=alpha,min_margin=min_margin),  # HKR stands for the hinge regularized KR loss
        metrics=[
            KR_loss((-1,1)),  # shows the KR term of the loss
            hinge_margin_loss(min_margin=min_margin),  # shows the hinge term of the loss
            HKR_binary_accuracy  # shows the classification accuracy
        ],
        optimizer=optimizer
    )

Learn classification on MNIST
-----------------------------

Now the model is build, we can learn the task.

.. code:: ipython3

    wass.fit(
        x=x_train, y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        verbose=1
    )


.. parsed-literal::

    Train on 11774 samples, validate on 1954 samples
    Epoch 1/5
    11774/11774 [==============================] - 5s 426us/sample - loss: -3.8264 - KR_loss_fct: -5.2401 - hinge_margin_fct: 0.1413 - HKR_binary_accuracy: 0.9546 - val_loss: -6.3826 - val_KR_loss_fct: -6.6289 - val_hinge_margin_fct: 0.0269 - val_HKR_binary_accuracy: 0.9889
    Epoch 2/5
    11774/11774 [==============================] - 2s 194us/sample - loss: -6.5813 - KR_loss_fct: -6.8297 - hinge_margin_fct: 0.0248 - HKR_binary_accuracy: 0.9906 - val_loss: -6.8006 - val_KR_loss_fct: -6.9829 - val_hinge_margin_fct: 0.0202 - val_HKR_binary_accuracy: 0.9908
    Epoch 3/5
    11774/11774 [==============================] - 2s 206us/sample - loss: -6.8227 - KR_loss_fct: -7.0366 - hinge_margin_fct: 0.0214 - HKR_binary_accuracy: 0.9929 - val_loss: -6.8027 - val_KR_loss_fct: -7.0636 - val_hinge_margin_fct: 0.0270 - val_HKR_binary_accuracy: 0.9893
    Epoch 4/5
    11774/11774 [==============================] - 2s 206us/sample - loss: -6.9042 - KR_loss_fct: -7.1081 - hinge_margin_fct: 0.0204 - HKR_binary_accuracy: 0.9929 - val_loss: -6.9615 - val_KR_loss_fct: -7.1755 - val_hinge_margin_fct: 0.0233 - val_HKR_binary_accuracy: 0.9913
    Epoch 5/5
    11774/11774 [==============================] - 2s 207us/sample - loss: -6.9774 - KR_loss_fct: -7.1707 - hinge_margin_fct: 0.0193 - HKR_binary_accuracy: 0.9927 - val_loss: -6.9884 - val_KR_loss_fct: -7.1752 - val_hinge_margin_fct: 0.0215 - val_HKR_binary_accuracy: 0.9918




.. parsed-literal::

    <tensorflow.python.keras.callbacks.History at 0x1fd64b2a048>



As we can see the model reach a very decent accuracy on this task.

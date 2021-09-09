HKR classifier on MNIST dataset
-------------------------------

|Open In Colab|

This notebook will demonstrate learning a binary task on the MNIST0-8
dataset.

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/doc/notebooks/demo3.ipynb

.. code:: ipython3

    # pip install deel-lip -qqq

.. code:: ipython3

    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.python.keras.layers import Input, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import binary_accuracy
    from tensorflow.keras.models import Sequential
    
    from deel.lip.layers import (
        SpectralConv2D,
        SpectralDense,
        FrobeniusDense,
        ScaledL2NormPooling2D,
    )
    from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
    from deel.lip.losses import HKR, KR, HingeMargin


.. parsed-literal::

    2021-09-09 15:23:07.702005: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0


data preparation
~~~~~~~~~~~~~~~~

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
        mask = (y == class_a) + (
            y == class_b
        )  # mask to select only items from class_a or class_b
        x = x[mask]
        y = y[mask]
        x = x.astype("float32")
        y = y.astype("float32")
        # convert from range int[0,255] to float32[-1,1]
        x /= 255
        x = x.reshape((-1, 28, 28, 1))
        # change label to binary classification {-1,1}
        y[y == class_a] = 1.0
        y[y == class_b] = -1.0
        return x, y
    
    
    # now we load the dataset
    (x_train, y_train_ord), (x_test, y_test_ord) = mnist.load_data()
    
    # prepare the data
    x_train, y_train = prepare_data(
        x_train, y_train_ord, selected_classes[0], selected_classes[1]
    )
    x_test, y_test = prepare_data(
        x_test, y_test_ord, selected_classes[0], selected_classes[1]
    )
    
    # display infos about dataset
    print(
        "train set size: %i samples, classes proportions: %.3f percent"
        % (y_train.shape[0], 100 * y_train[y_train == 1].sum() / y_train.shape[0])
    )
    print(
        "test set size: %i samples, classes proportions: %.3f percent"
        % (y_test.shape[0], 100 * y_test[y_test == 1].sum() / y_test.shape[0])
    )



.. parsed-literal::

    train set size: 11774 samples, classes proportions: 50.306 percent
    test set size: 1954 samples, classes proportions: 50.154 percent


Build lipschitz Model
~~~~~~~~~~~~~~~~~~~~~

Let’s first explicit the paremeters of this experiment

.. code:: ipython3

    # training parameters
    epochs = 10
    batch_size = 128
    
    # network parameters
    activation = GroupSort  # ReLU, MaxMin, GroupSort2
    
    # loss parameters
    min_margin = 1.0
    alpha = 10.0


Now we can build the network. Here the experiment is done with a MLP.
But ``Deel-lip`` also provide state of the art 1-Lipschitz convolutions.

.. code:: ipython3

    K.clear_session()
    # helper function to build the 1-lipschitz MLP
    wass = Sequential(
        layers=[
            Input((28, 28, 1)),
            Flatten(),
            SpectralDense(32, GroupSort2(), use_bias=True),
            SpectralDense(16, GroupSort2(), use_bias=True),
            FrobeniusDense(1, activation=None, use_bias=False),
        ],
        name="lipModel",
    )
    wass.summary()


.. parsed-literal::

    2021-09-09 15:23:10.579084: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 15:23:10.579563: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
    2021-09-09 15:23:10.611186: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.611438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 15:23:10.611451: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:23:10.612507: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:23:10.612532: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 15:23:10.613099: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 15:23:10.613223: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 15:23:10.614505: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 15:23:10.614807: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 15:23:10.614871: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 15:23:10.614925: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.615184: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.615402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 15:23:10.616085: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 15:23:10.616254: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.616489: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 15:23:10.616500: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:23:10.616512: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:23:10.616520: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 15:23:10.616527: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 15:23:10.616534: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 15:23:10.616541: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 15:23:10.616548: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 15:23:10.616555: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 15:23:10.616593: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.616889: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:10.617101: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 15:23:10.617118: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:23:11.064873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
    2021-09-09 15:23:11.064896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
    2021-09-09 15:23:11.064900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
    2021-09-09 15:23:11.065029: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:11.065277: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:11.065498: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:23:11.065704: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7250 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5)
    2021-09-09 15:23:11.125200: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:23:11.344450: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11


.. parsed-literal::

    Model: "lipModel"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    spectral_dense (SpectralDens (None, 32)                50241     
    _________________________________________________________________
    spectral_dense_1 (SpectralDe (None, 16)                1057      
    _________________________________________________________________
    frobenius_dense (FrobeniusDe (None, 1)                 32        
    =================================================================
    Total params: 51,330
    Trainable params: 25,664
    Non-trainable params: 25,666
    _________________________________________________________________


.. code:: ipython3

    optimizer = Adam(lr=0.001)

.. code:: ipython3

    # as the output of our classifier is in the real range [-1, 1], binary accuracy must be redefined
    def HKR_binary_accuracy(y_true, y_pred):
        S_true = tf.dtypes.cast(tf.greater_equal(y_true[:, 0], 0), dtype=tf.float32)
        S_pred = tf.dtypes.cast(tf.greater_equal(y_pred[:, 0], 0), dtype=tf.float32)
        return binary_accuracy(S_true, S_pred)


.. code:: ipython3

    wass.compile(
        loss=HKR(
            alpha=alpha, min_margin=min_margin
        ),  # HKR stands for the hinge regularized KR loss
        metrics=[
            KR,  # shows the KR term of the loss
            HingeMargin(min_margin=min_margin),  # shows the hinge term of the loss
            HKR_binary_accuracy,  # shows the classification accuracy
        ],
        optimizer=optimizer,
    )

Learn classification on MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now the model is build, we can learn the task.

.. code:: ipython3

    wass.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        verbose=1,
    )


.. parsed-literal::

    Epoch 1/10


.. parsed-literal::

    2021-09-09 15:23:11.892060: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
    2021-09-09 15:23:11.909014: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3600000000 Hz


.. parsed-literal::

    92/92 [==============================] - 3s 19ms/step - loss: -1.1506 - KR: 3.5147 - HingeMargin: 0.2364 - HKR_binary_accuracy: 0.9176 - val_loss: -5.0700 - val_KR: 5.5769 - val_HingeMargin: 0.0520 - val_HKR_binary_accuracy: 0.9777
    Epoch 2/10
    92/92 [==============================] - 1s 11ms/step - loss: -4.9765 - KR: 5.5275 - HingeMargin: 0.0551 - HKR_binary_accuracy: 0.9783 - val_loss: -5.4217 - val_KR: 5.7943 - val_HingeMargin: 0.0397 - val_HKR_binary_accuracy: 0.9830
    Epoch 3/10
    92/92 [==============================] - 1s 11ms/step - loss: -5.3703 - KR: 5.7864 - HingeMargin: 0.0416 - HKR_binary_accuracy: 0.9843 - val_loss: -5.4753 - val_KR: 5.8651 - val_HingeMargin: 0.0405 - val_HKR_binary_accuracy: 0.9825
    Epoch 4/10
    92/92 [==============================] - 2s 17ms/step - loss: -5.4480 - KR: 5.8239 - HingeMargin: 0.0376 - HKR_binary_accuracy: 0.9875 - val_loss: -5.5214 - val_KR: 5.9311 - val_HingeMargin: 0.0418 - val_HKR_binary_accuracy: 0.9816
    Epoch 5/10
    92/92 [==============================] - 2s 24ms/step - loss: -5.5125 - KR: 5.8967 - HingeMargin: 0.0384 - HKR_binary_accuracy: 0.9866 - val_loss: -5.5819 - val_KR: 5.9777 - val_HingeMargin: 0.0406 - val_HKR_binary_accuracy: 0.9820
    Epoch 6/10
    92/92 [==============================] - 1s 11ms/step - loss: -5.5839 - KR: 5.9638 - HingeMargin: 0.0380 - HKR_binary_accuracy: 0.9858 - val_loss: -5.6179 - val_KR: 6.0573 - val_HingeMargin: 0.0448 - val_HKR_binary_accuracy: 0.9820
    Epoch 7/10
    92/92 [==============================] - 1s 15ms/step - loss: -5.6282 - KR: 6.0265 - HingeMargin: 0.0398 - HKR_binary_accuracy: 0.9841 - val_loss: -5.7544 - val_KR: 6.0605 - val_HingeMargin: 0.0329 - val_HKR_binary_accuracy: 0.9855
    Epoch 8/10
    92/92 [==============================] - 2s 23ms/step - loss: -5.6031 - KR: 6.0081 - HingeMargin: 0.0405 - HKR_binary_accuracy: 0.9855 - val_loss: -5.7766 - val_KR: 6.0768 - val_HingeMargin: 0.0329 - val_HKR_binary_accuracy: 0.9860
    Epoch 9/10
    92/92 [==============================] - 1s 16ms/step - loss: -5.6996 - KR: 6.0569 - HingeMargin: 0.0357 - HKR_binary_accuracy: 0.9857 - val_loss: -5.7625 - val_KR: 6.1506 - val_HingeMargin: 0.0400 - val_HKR_binary_accuracy: 0.9820
    Epoch 10/10
    92/92 [==============================] - 1s 7ms/step - loss: -5.6771 - KR: 6.0637 - HingeMargin: 0.0387 - HKR_binary_accuracy: 0.9848 - val_loss: -5.8074 - val_KR: 6.1306 - val_HingeMargin: 0.0341 - val_HKR_binary_accuracy: 0.9850




.. parsed-literal::

    <tensorflow.python.keras.callbacks.History at 0x7f2dcc52d710>



As we can see the model reach a very decent accuracy on this task.

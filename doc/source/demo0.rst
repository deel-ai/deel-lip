Example and usage
-----------------

|Open In Colab|

In order to make things simple the following rules have been followed
during development:

-  ``deel-lip`` follows the ``keras`` package structure.
-  All elements (layers, activations, initializers, …) are compatible
   with standard the ``keras`` elements.
-  When a k-Lipschitz layer overrides a standard keras layer, it uses
   the same interface and the same parameters. The only difference is a
   new parameter to control the Lipschitz constant of a layer.

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/doc/notebooks/demo0.ipynb

Which layers are safe to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table indicates which layers are safe to use in a Lipshitz
network, and which are not.

+--------------+---+------------------+---------------------------------+
| layer        | 1 | deel-lip         | comments                        |
|              | - | equivalent       |                                 |
|              | l |                  |                                 |
|              | i |                  |                                 |
|              | p |                  |                                 |
|              | ? |                  |                                 |
+==============+===+==================+=================================+
| ``Dense``    | n | ``SpectralDense` | ``SpectralDense`` and           |
|              | o | `\ \ ``Frobenius | \ ``FrobeniusDense`` are        |
|              |   | Dense``          | similarwhen there is a single   |
|              |   |                  | output                          |
+--------------+---+------------------+---------------------------------+
| ``Conv2D``   | n | ``SpectralConv2D | ``SpectralConv2D`` also         |
|              | o | ``\ \ ``Frobeniu | implement Björck normalization  |
|              |   | sConv2D``        |                                 |
+--------------+---+------------------+---------------------------------+
| ``MaxPooling | y | na.              |                                 |
| ``\ \ ``Glob | e |                  |                                 |
| alMaxPooling | s |                  |                                 |
| ``           |   |                  |                                 |
+--------------+---+------------------+---------------------------------+
| ``AveragePoo | n | ``ScaledAverageP | The lipschitz constant is       |
| ling``\ \ `` | o | ooling``\ \ ``Sc | bounded by sqrt(pool_h*pool_h)  |
| GlobalAverag |   | aledGlobalAverag |                                 |
| ePooling``   |   | ePooling``       |                                 |
+--------------+---+------------------+---------------------------------+
| ``Flatten``  | y | na.              |                                 |
|              | e |                  |                                 |
|              | s |                  |                                 |
+--------------+---+------------------+---------------------------------+
| ``Dropout``  | n | None             | The lipschitz constant is       |
|              | o |                  | bounded by the dropout factor   |
+--------------+---+------------------+---------------------------------+
| ``BatchNorm` | n | None             | It is suspected that layer      |
| `            | o |                  | normalization alreadylimits     |
|              |   |                  | internal covariateshift         |
+--------------+---+------------------+---------------------------------+

Design tips
-----------

Designing lipschitz networks require a careful design in order to avoid
vanishing/exploding gradient problem.

Choosing pooling layers:

+-------------+------------------------+------------------------------+
| layer       | advantages             | disadvantages                |
+=============+========================+==============================+
| ``ScaledAve | very similar to        | not norm preserving nor      |
| ragePooling | original               | gradient norm preserving.    |
| 2D``        | implementation (just   |                              |
| and         | add a scaling factor   |                              |
| ``MaxPoolin | for avg).              |                              |
| g2D``       |                        |                              |
+-------------+------------------------+------------------------------+
| ``Invertibl | norm preserving and    | increases the number of      |
| eDownSampli | gradient norm          | channels (and the number of  |
| ng``        | preserving.            | parameters of the next       |
|             |                        | layer).                      |
+-------------+------------------------+------------------------------+
| ``ScaledL2N | norm preserving.       | lower numerical stability of |
| ormPooling2 |                        | the gradient when inputs are |
| D``         |                        | close to zero.               |
| (           |                        |                              |
| ``sqrt(avgp |                        |                              |
| ool(x**2))` |                        |                              |
| `           |                        |                              |
| )           |                        |                              |
+-------------+------------------------+------------------------------+

Choosing activations:

+-------------+------------------------+------------------------------+
| layer       | advantages             | disadvantages                |
+=============+========================+==============================+
| ReLU\ ``| w |                        |                              |
| idely used  |                        |                              |
| | create a  |                        |                              |
| strong vani |                        |                              |
| shing gradi |                        |                              |
| ent effect. |                        |                              |
|  | |``\ Max |                        |                              |
| Min\ ``(``\ |                        |                              |
|  stack([ReL |                        |                              |
| U(x),       |                        |                              |
| ReLU(-x)])\ |                        |                              |
|  ``) | have |                        |                              |
|  similar pr |                        |                              |
| operties to |                        |                              |
|  ReLU, but  |                        |                              |
| is norm and |                        |                              |
|  gradient n |                        |                              |
| orm preserv |                        |                              |
| ing | doubl |                        |                              |
| e the numbe |                        |                              |
| r of output |                        |                              |
| s | |``\ Gr |                        |                              |
| oupSort\ `` |                        |                              |
| | Input and |                        |                              |
|  GradientNo |                        |                              |
| rm preservi |                        |                              |
| ng. Also li |                        |                              |
| mit the nee |                        |                              |
| d of biases |                        |                              |
|  (as it is  |                        |                              |
| shift invar |                        |                              |
| iant). | mo |                        |                              |
| re computat |                        |                              |
| ionally exp |                        |                              |
| ensive, (wh |                        |                              |
| en it's par |                        |                              |
| ameter``\ n |                        |                              |
| \`          |                        |                              |
| is large)   |                        |                              |
+-------------+------------------------+------------------------------+

Please note that when learning with the :class:``.HKR`` and
:class:``.MulticlassHKR``, no activation is required on the last layer.

.. code:: ipython3

    from deel.lip.layers import (
        SpectralDense,
        SpectralConv2D,
        ScaledL2NormPooling2D,
        FrobeniusDense,
    )
    from deel.lip.model import Sequential
    from deel.lip.activations import GroupSort
    from deel.lip.losses import MulticlassHKR, MulticlassKR
    from tensorflow.keras.layers import Input, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    import numpy as np
    
    # Sequential (resp Model) from deel.model has the same properties as any lipschitz model.
    # It act only as a container, with features specific to lipschitz
    # functions (condensation, vanilla_exportation...)
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            # Lipschitz layers preserve the API of their superclass ( here Conv2D )
            # an optional param is available: k_coef_lip which control the lipschitz
            # constant of the layer
            SpectralConv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            # usual pooling layer are implemented (avg, max...), but new layers are also available
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            SpectralConv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            # our layers are fully interoperable with existing keras layers
            Flatten(),
            SpectralDense(
                32,
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            FrobeniusDense(
                10, activation=None, use_bias=False, kernel_initializer="orthogonal"
            ),
        ],
        # similary model has a parameter to set the lipschitz constant
        # to set automatically the constant of each layer
        k_coef_lip=1.0,
        name="hkr_model",
    )
    
    # HKR (Hinge-Krantorovich-Rubinstein) optimize robustness along with accuracy
    model.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=MulticlassHKR(alpha=50, min_margin=0.05),
        optimizer=Adam(1e-3),
        metrics=["accuracy", MulticlassKR()],
    )
    
    model.summary()
    
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # standardize and reshape the data
    x_train = np.expand_dims(x_train, -1)
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = np.expand_dims(x_test, -1)
    x_test = (x_test - mean) / std
    # one hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # fit the model
    model.fit(
        x_train,
        y_train,
        batch_size=2048,
        epochs=30,
        validation_data=(x_test, y_test),
        shuffle=True,
    )
    
    # once training is finished you can convert
    # SpectralDense layers into Dense layers and SpectralConv2D into Conv2D
    # which optimize performance for inference
    vanilla_model = model.vanilla_export()


.. parsed-literal::

    2021-09-09 15:16:01.159175: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:16:02.890348: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 15:16:02.890831: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
    2021-09-09 15:16:02.922959: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.923206: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 15:16:02.923219: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:16:02.924425: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:16:02.924448: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 15:16:02.924976: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 15:16:02.925093: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 15:16:02.926353: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 15:16:02.926644: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 15:16:02.926707: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 15:16:02.926755: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.927005: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.927218: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 15:16:02.927730: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 15:16:02.927803: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.928022: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 15:16:02.928033: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:16:02.928042: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:16:02.928050: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 15:16:02.928057: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 15:16:02.928064: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 15:16:02.928071: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 15:16:02.928079: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 15:16:02.928086: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 15:16:02.928114: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.928343: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:02.928551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 15:16:02.928567: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 15:16:03.382533: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
    2021-09-09 15:16:03.382554: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
    2021-09-09 15:16:03.382558: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
    2021-09-09 15:16:03.382680: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:03.382924: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:03.383140: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 15:16:03.383346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7250 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5)
    /home/thibaut.boissin/projects/repo_github/deel-lip/deel/lip/model.py:56: UserWarning: Sequential model contains a layer wich is not a Lipschitz layer: flatten
      layer.name


.. parsed-literal::

    Model: "hkr_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    spectral_conv2d (SpectralCon (None, 28, 28, 16)        321       
    _________________________________________________________________
    scaled_l2norm_pooling2d (Sca (None, 14, 14, 16)        0         
    _________________________________________________________________
    spectral_conv2d_1 (SpectralC (None, 14, 14, 16)        4641      
    _________________________________________________________________
    scaled_l2norm_pooling2d_1 (S (None, 7, 7, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    spectral_dense (SpectralDens (None, 32)                50241     
    _________________________________________________________________
    frobenius_dense (FrobeniusDe (None, 10)                640       
    =================================================================
    Total params: 55,843
    Trainable params: 27,920
    Non-trainable params: 27,923
    _________________________________________________________________


.. parsed-literal::

    2021-09-09 15:16:04.662404: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
    2021-09-09 15:16:04.680914: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3600000000 Hz


.. parsed-literal::

    Epoch 1/30


.. parsed-literal::

    2021-09-09 15:16:06.629816: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 15:16:06.849059: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 15:16:06.859078: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8


.. parsed-literal::

    30/30 [==============================] - 4s 49ms/step - loss: 6.6052 - accuracy: 0.2772 - MulticlassKR: 0.1076 - val_loss: 0.8885 - val_accuracy: 0.7818 - val_MulticlassKR: 0.3007
    Epoch 2/30
    30/30 [==============================] - 1s 34ms/step - loss: 0.7030 - accuracy: 0.8191 - MulticlassKR: 0.3289 - val_loss: 0.2899 - val_accuracy: 0.8886 - val_MulticlassKR: 0.4163
    Epoch 3/30
    30/30 [==============================] - 1s 44ms/step - loss: 0.2362 - accuracy: 0.8910 - MulticlassKR: 0.4383 - val_loss: 0.0432 - val_accuracy: 0.9129 - val_MulticlassKR: 0.5269
    Epoch 4/30
    30/30 [==============================] - 1s 40ms/step - loss: 0.0201 - accuracy: 0.9141 - MulticlassKR: 0.5554 - val_loss: -0.1618 - val_accuracy: 0.9284 - val_MulticlassKR: 0.6757
    Epoch 5/30
    30/30 [==============================] - 1s 38ms/step - loss: -0.1892 - accuracy: 0.9269 - MulticlassKR: 0.7151 - val_loss: -0.3834 - val_accuracy: 0.9376 - val_MulticlassKR: 0.8826
    Epoch 6/30
    30/30 [==============================] - 1s 38ms/step - loss: -0.3982 - accuracy: 0.9318 - MulticlassKR: 0.9375 - val_loss: -0.6242 - val_accuracy: 0.9383 - val_MulticlassKR: 1.1741
    Epoch 7/30
    30/30 [==============================] - 1s 35ms/step - loss: -0.6932 - accuracy: 0.9369 - MulticlassKR: 1.2578 - val_loss: -0.9868 - val_accuracy: 0.9412 - val_MulticlassKR: 1.5829
    Epoch 8/30
    30/30 [==============================] - 1s 34ms/step - loss: -1.0436 - accuracy: 0.9388 - MulticlassKR: 1.7083 - val_loss: -1.4178 - val_accuracy: 0.9458 - val_MulticlassKR: 2.1600
    Epoch 9/30
    30/30 [==============================] - 1s 38ms/step - loss: -1.4851 - accuracy: 0.9356 - MulticlassKR: 2.3177 - val_loss: -1.9688 - val_accuracy: 0.9402 - val_MulticlassKR: 2.8508
    Epoch 10/30
    30/30 [==============================] - 1s 46ms/step - loss: -2.0808 - accuracy: 0.9380 - MulticlassKR: 3.0365 - val_loss: -2.5709 - val_accuracy: 0.9406 - val_MulticlassKR: 3.6565
    Epoch 11/30
    30/30 [==============================] - 1s 35ms/step - loss: -2.6181 - accuracy: 0.9396 - MulticlassKR: 3.7817 - val_loss: -3.1971 - val_accuracy: 0.9460 - val_MulticlassKR: 4.3753
    Epoch 12/30
    30/30 [==============================] - 1s 35ms/step - loss: -3.2213 - accuracy: 0.9419 - MulticlassKR: 4.4936 - val_loss: -3.7071 - val_accuracy: 0.9461 - val_MulticlassKR: 5.0001
    Epoch 13/30
    30/30 [==============================] - 1s 45ms/step - loss: -3.6440 - accuracy: 0.9407 - MulticlassKR: 5.0669 - val_loss: -4.1264 - val_accuracy: 0.9487 - val_MulticlassKR: 5.5340
    Epoch 14/30
    30/30 [==============================] - 1s 36ms/step - loss: -4.0550 - accuracy: 0.9429 - MulticlassKR: 5.5451 - val_loss: -4.4434 - val_accuracy: 0.9459 - val_MulticlassKR: 5.8784
    Epoch 15/30
    30/30 [==============================] - 1s 36ms/step - loss: -4.2643 - accuracy: 0.9441 - MulticlassKR: 5.8402 - val_loss: -4.7133 - val_accuracy: 0.9532 - val_MulticlassKR: 6.1418
    Epoch 16/30
    30/30 [==============================] - 1s 36ms/step - loss: -4.5803 - accuracy: 0.9482 - MulticlassKR: 6.1061 - val_loss: -4.8904 - val_accuracy: 0.9512 - val_MulticlassKR: 6.3929
    Epoch 17/30
    30/30 [==============================] - 1s 36ms/step - loss: -4.7383 - accuracy: 0.9452 - MulticlassKR: 6.3495 - val_loss: -5.0702 - val_accuracy: 0.9538 - val_MulticlassKR: 6.5704
    Epoch 18/30
    30/30 [==============================] - 1s 38ms/step - loss: -4.9243 - accuracy: 0.9498 - MulticlassKR: 6.5062 - val_loss: -5.1903 - val_accuracy: 0.9541 - val_MulticlassKR: 6.7263
    Epoch 19/30
    30/30 [==============================] - 1s 41ms/step - loss: -5.1261 - accuracy: 0.9522 - MulticlassKR: 6.6473 - val_loss: -5.2914 - val_accuracy: 0.9531 - val_MulticlassKR: 6.8514
    Epoch 20/30
    30/30 [==============================] - 1s 36ms/step - loss: -5.1228 - accuracy: 0.9497 - MulticlassKR: 6.7646 - val_loss: -5.4142 - val_accuracy: 0.9560 - val_MulticlassKR: 6.9832
    Epoch 21/30
    30/30 [==============================] - 1s 37ms/step - loss: -5.1569 - accuracy: 0.9514 - MulticlassKR: 6.8529 - val_loss: -5.5265 - val_accuracy: 0.9566 - val_MulticlassKR: 7.0741
    Epoch 22/30
    30/30 [==============================] - 1s 36ms/step - loss: -5.3783 - accuracy: 0.9545 - MulticlassKR: 6.9732 - val_loss: -5.6297 - val_accuracy: 0.9577 - val_MulticlassKR: 7.1469
    Epoch 23/30
    30/30 [==============================] - 1s 47ms/step - loss: -5.4597 - accuracy: 0.9546 - MulticlassKR: 7.0535 - val_loss: -5.6792 - val_accuracy: 0.9583 - val_MulticlassKR: 7.2324
    Epoch 24/30
    30/30 [==============================] - 1s 36ms/step - loss: -5.5423 - accuracy: 0.9544 - MulticlassKR: 7.1429 - val_loss: -5.7364 - val_accuracy: 0.9593 - val_MulticlassKR: 7.2954
    Epoch 25/30
    30/30 [==============================] - 1s 44ms/step - loss: -5.6056 - accuracy: 0.9562 - MulticlassKR: 7.1997 - val_loss: -5.7986 - val_accuracy: 0.9588 - val_MulticlassKR: 7.3697
    Epoch 26/30
    30/30 [==============================] - 1s 40ms/step - loss: -5.6212 - accuracy: 0.9541 - MulticlassKR: 7.2370 - val_loss: -5.8694 - val_accuracy: 0.9618 - val_MulticlassKR: 7.3834
    Epoch 27/30
    30/30 [==============================] - 1s 35ms/step - loss: -5.7346 - accuracy: 0.9554 - MulticlassKR: 7.3269 - val_loss: -5.9372 - val_accuracy: 0.9622 - val_MulticlassKR: 7.4778
    Epoch 28/30
    30/30 [==============================] - 1s 43ms/step - loss: -5.6740 - accuracy: 0.9550 - MulticlassKR: 7.3377 - val_loss: -5.9493 - val_accuracy: 0.9590 - val_MulticlassKR: 7.5607
    Epoch 29/30
    30/30 [==============================] - 1s 42ms/step - loss: -5.7000 - accuracy: 0.9557 - MulticlassKR: 7.3908 - val_loss: -6.0002 - val_accuracy: 0.9601 - val_MulticlassKR: 7.5647
    Epoch 30/30
    30/30 [==============================] - 1s 36ms/step - loss: -5.8546 - accuracy: 0.9571 - MulticlassKR: 7.4730 - val_loss: -6.0826 - val_accuracy: 0.9607 - val_MulticlassKR: 7.6294


How to use it ?
~~~~~~~~~~~~~~~

|Open In Colab|

Here is an example of 1-lipschitz network trained on MNIST:

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/doc/notebooks/demo0.ipynb

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
    # functions (condensation, vanilla_exportation...) but The layers are fully compatible
    # with the tf.keras.model.Sequential/Model
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

    2021-09-09 18:20:38.651881: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 18:20:41.859471: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 18:20:41.859959: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
    2021-09-09 18:20:41.887947: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.888196: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 18:20:41.888209: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 18:20:41.889435: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 18:20:41.889461: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 18:20:41.889997: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 18:20:41.890121: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 18:20:41.891391: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 18:20:41.891695: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 18:20:41.891762: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 18:20:41.891814: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.892071: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.892288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 18:20:41.892775: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
    2021-09-09 18:20:41.892838: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.893060: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
    pciBusID: 0000:01:00.0 name: GeForce RTX 2070 SUPER computeCapability: 7.5
    coreClock: 1.785GHz coreCount: 40 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
    2021-09-09 18:20:41.893071: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 18:20:41.893079: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 18:20:41.893086: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 18:20:41.893094: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
    2021-09-09 18:20:41.893101: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
    2021-09-09 18:20:41.893107: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
    2021-09-09 18:20:41.893115: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
    2021-09-09 18:20:41.893122: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
    2021-09-09 18:20:41.893153: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.893390: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:41.893601: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
    2021-09-09 18:20:41.893617: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
    2021-09-09 18:20:42.348799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
    2021-09-09 18:20:42.348820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
    2021-09-09 18:20:42.348824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
    2021-09-09 18:20:42.348955: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:42.349207: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:42.349427: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2021-09-09 18:20:42.349634: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7250 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5)
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

    2021-09-09 18:20:43.638117: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
    2021-09-09 18:20:43.656873: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 3600000000 Hz


.. parsed-literal::

    Epoch 1/30


.. parsed-literal::

    2021-09-09 18:20:45.586440: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
    2021-09-09 18:20:45.805767: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
    2021-09-09 18:20:45.815934: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8


.. parsed-literal::

    30/30 [==============================] - 4s 52ms/step - loss: 5.7859 - accuracy: 0.3059 - MulticlassKR: 0.0994 - val_loss: 0.7743 - val_accuracy: 0.8195 - val_MulticlassKR: 0.3336
    Epoch 2/30
    30/30 [==============================] - 1s 35ms/step - loss: 0.5617 - accuracy: 0.8488 - MulticlassKR: 0.3664 - val_loss: 0.2028 - val_accuracy: 0.8998 - val_MulticlassKR: 0.4562
    Epoch 3/30
    30/30 [==============================] - 1s 40ms/step - loss: 0.1443 - accuracy: 0.9037 - MulticlassKR: 0.4800 - val_loss: -0.0439 - val_accuracy: 0.9243 - val_MulticlassKR: 0.5668
    Epoch 4/30
    30/30 [==============================] - 1s 39ms/step - loss: -0.0865 - accuracy: 0.9233 - MulticlassKR: 0.6017 - val_loss: -0.2614 - val_accuracy: 0.9352 - val_MulticlassKR: 0.7281
    Epoch 5/30
    30/30 [==============================] - 1s 44ms/step - loss: -0.3090 - accuracy: 0.9345 - MulticlassKR: 0.7771 - val_loss: -0.5085 - val_accuracy: 0.9448 - val_MulticlassKR: 0.9635
    Epoch 6/30
    30/30 [==============================] - 1s 35ms/step - loss: -0.5742 - accuracy: 0.9418 - MulticlassKR: 1.0413 - val_loss: -0.8245 - val_accuracy: 0.9469 - val_MulticlassKR: 1.3165
    Epoch 7/30
    30/30 [==============================] - 1s 36ms/step - loss: -0.8896 - accuracy: 0.9426 - MulticlassKR: 1.4164 - val_loss: -1.2121 - val_accuracy: 0.9464 - val_MulticlassKR: 1.7998
    Epoch 8/30
    30/30 [==============================] - 1s 35ms/step - loss: -1.3101 - accuracy: 0.9430 - MulticlassKR: 1.9421 - val_loss: -1.7661 - val_accuracy: 0.9515 - val_MulticlassKR: 2.4609
    Epoch 9/30
    30/30 [==============================] - 1s 47ms/step - loss: -1.8807 - accuracy: 0.9425 - MulticlassKR: 2.6451 - val_loss: -2.4294 - val_accuracy: 0.9480 - val_MulticlassKR: 3.2977
    Epoch 10/30
    30/30 [==============================] - 1s 43ms/step - loss: -2.5482 - accuracy: 0.9444 - MulticlassKR: 3.4797 - val_loss: -3.0506 - val_accuracy: 0.9478 - val_MulticlassKR: 4.1679
    Epoch 11/30
    30/30 [==============================] - 1s 38ms/step - loss: -3.1723 - accuracy: 0.9439 - MulticlassKR: 4.3124 - val_loss: -3.6976 - val_accuracy: 0.9475 - val_MulticlassKR: 4.9445
    Epoch 12/30
    30/30 [==============================] - 1s 34ms/step - loss: -3.7133 - accuracy: 0.9441 - MulticlassKR: 5.0248 - val_loss: -4.2211 - val_accuracy: 0.9525 - val_MulticlassKR: 5.5240
    Epoch 13/30
    30/30 [==============================] - 1s 37ms/step - loss: -4.1847 - accuracy: 0.9456 - MulticlassKR: 5.5629 - val_loss: -4.5868 - val_accuracy: 0.9538 - val_MulticlassKR: 5.9152
    Epoch 14/30
    30/30 [==============================] - 1s 46ms/step - loss: -4.4194 - accuracy: 0.9447 - MulticlassKR: 5.9083 - val_loss: -4.8092 - val_accuracy: 0.9530 - val_MulticlassKR: 6.2309
    Epoch 15/30
    30/30 [==============================] - 1s 42ms/step - loss: -4.6380 - accuracy: 0.9473 - MulticlassKR: 6.1855 - val_loss: -4.9103 - val_accuracy: 0.9499 - val_MulticlassKR: 6.4634
    Epoch 16/30
    30/30 [==============================] - 1s 36ms/step - loss: -4.8019 - accuracy: 0.9476 - MulticlassKR: 6.3995 - val_loss: -5.1251 - val_accuracy: 0.9541 - val_MulticlassKR: 6.6381
    Epoch 17/30
    30/30 [==============================] - 1s 40ms/step - loss: -4.9292 - accuracy: 0.9503 - MulticlassKR: 6.5580 - val_loss: -5.2763 - val_accuracy: 0.9563 - val_MulticlassKR: 6.7558
    Epoch 18/30
    30/30 [==============================] - 1s 35ms/step - loss: -5.0473 - accuracy: 0.9504 - MulticlassKR: 6.6735 - val_loss: -5.3574 - val_accuracy: 0.9554 - val_MulticlassKR: 6.8654
    Epoch 19/30
    30/30 [==============================] - 1s 41ms/step - loss: -5.1484 - accuracy: 0.9503 - MulticlassKR: 6.7765 - val_loss: -5.4485 - val_accuracy: 0.9561 - val_MulticlassKR: 6.9638
    Epoch 20/30
    30/30 [==============================] - 1s 47ms/step - loss: -5.2245 - accuracy: 0.9506 - MulticlassKR: 6.8670 - val_loss: -5.5184 - val_accuracy: 0.9558 - val_MulticlassKR: 7.0767
    Epoch 21/30
    30/30 [==============================] - 1s 35ms/step - loss: -5.3259 - accuracy: 0.9507 - MulticlassKR: 6.9613 - val_loss: -5.5777 - val_accuracy: 0.9573 - val_MulticlassKR: 7.1658
    Epoch 22/30
    30/30 [==============================] - 1s 35ms/step - loss: -5.4587 - accuracy: 0.9519 - MulticlassKR: 7.0682 - val_loss: -5.7211 - val_accuracy: 0.9595 - val_MulticlassKR: 7.2207
    Epoch 23/30
    30/30 [==============================] - 1s 37ms/step - loss: -5.5685 - accuracy: 0.9534 - MulticlassKR: 7.1410 - val_loss: -5.7894 - val_accuracy: 0.9618 - val_MulticlassKR: 7.2921
    Epoch 24/30
    30/30 [==============================] - 1s 35ms/step - loss: -5.4871 - accuracy: 0.9533 - MulticlassKR: 7.1789 - val_loss: -5.8136 - val_accuracy: 0.9606 - val_MulticlassKR: 7.3730
    Epoch 25/30
    30/30 [==============================] - 1s 46ms/step - loss: -5.6827 - accuracy: 0.9551 - MulticlassKR: 7.2730 - val_loss: -5.9069 - val_accuracy: 0.9588 - val_MulticlassKR: 7.4427
    Epoch 26/30
    30/30 [==============================] - 1s 34ms/step - loss: -5.7042 - accuracy: 0.9556 - MulticlassKR: 7.3001 - val_loss: -5.9921 - val_accuracy: 0.9606 - val_MulticlassKR: 7.4756
    Epoch 27/30
    30/30 [==============================] - 1s 48ms/step - loss: -5.7871 - accuracy: 0.9549 - MulticlassKR: 7.3868 - val_loss: -6.0014 - val_accuracy: 0.9609 - val_MulticlassKR: 7.5259
    Epoch 28/30
    30/30 [==============================] - 1s 38ms/step - loss: -5.8166 - accuracy: 0.9548 - MulticlassKR: 7.3946 - val_loss: -5.9561 - val_accuracy: 0.9573 - val_MulticlassKR: 7.5932
    Epoch 29/30
    30/30 [==============================] - 1s 36ms/step - loss: -5.8229 - accuracy: 0.9551 - MulticlassKR: 7.4779 - val_loss: -6.1211 - val_accuracy: 0.9593 - val_MulticlassKR: 7.6141
    Epoch 30/30
    30/30 [==============================] - 1s 34ms/step - loss: -5.9549 - accuracy: 0.9559 - MulticlassKR: 7.5246 - val_loss: -6.2155 - val_accuracy: 0.9606 - val_MulticlassKR: 7.6790


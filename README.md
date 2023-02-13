# MSTransformer

## Overview

MSTransformer is a deep learning transformer for music source separation. Since the introduction of Deep Learning, source separation has been dominated by RNNs. However, RNNs scale poorly due to parallelization and exploding/vanishing gradients. MSTransformer attempts to implement a scalable model for music source separation, by focusing on local context provided by a transformer. Prior implementations of transformers for audio source separation have underperformed due to smaller datasets. To remedy this, we use the random sampling method from Open-Unmix to generate new mixtures as input data.

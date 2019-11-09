# Neural ODE

## Implementation of Latent ODE, Augmented ODE and Neural ODE Topologies
## Original implementation via PyTorch can be found here: https://github.com/rtqichen/torchdiffeq (1)
## ANODEs implementation using PyTorch located here: https://github.com/EmilienDupont/augmented-neural-odes/ (2)

## This repo contains modifications of reimplemented (1) for usage with TensorFlow: https://github.com/titu1994/tfdiffeq/
## i.e.
### -Refactoring of the existing code to make it PEP8-friendly
### -Adding new layer types
### -Reimplementing existing methods for usage with vanilla Keras, not tf.Keras
### -Implementing (2) via tf/theano backend for Keras

# MSTransformer
This project introduces two transformer-based architectures for music separation, MSTransformer and MSTU. MSTransformer is a purely transformer-based architecture, following the method from "Attention Is All You Need" (Vaswani et al., 2017). MSTU utilizes both the transformer encoder and the UNet architecture to encode additional information and improve performance.

## Layout
### Transformer
Transformer directory includes all sublayers and model implementations:
- ``decoder.py``: Individual transformer decoder block. Decoder class lays consecutive decoder blocks.
- ``encoder.py``: Individual transformer encoder block. Encoder class lays consecutive encoder blocks.
- ``mstransformer.py``: MSTransformer implementation using preprocessing layer, encoder, decoder, and postprocessing layer.
- ``mstu.py``: MSTU implementation using downsampling blocks, transformer encoder, bottleneck, and upsampling blocks.
- ``position.py``: Positional encodings for transformer encoder.
- ``preprocess.py``: Both preprocessing and postprocessing layers for MSTransformer.
- ``sampling.py``: Implementation of upsampling and downsampling blocks for MSTU. 

Utils directory includes all additional utilities for training, preprocessing, and analysis:
- ``analysis.py``: SDR evaluation. Calculation for number of parameters in a model.
- ``dataset.py``: Implementation of torch dataset wrapper class for MUSDB18 dataset.
- ``fourier.py``: Short-Time Fourier Transform and Inverse Short-Time Fourier Transform.

## Diagrams
We provide diagrams of both MSTransformer and MSTU architectures for clarity.
### MSTransformer
![diagram of MSTransformer architecture](https://github.com/ezackr/MSTransformer/blob/main/images/MSTransformer.png?raw=true)
### MSTU
![diagram of MSTU architecture](https://github.com/ezackr/MSTransformer/blob/main/images/MSTU.png?raw=true)

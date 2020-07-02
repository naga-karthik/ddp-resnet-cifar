### Training Memory-Intensive Deep Learning Models with PyTorch’s Distributed Data Parallel

This is a mini-repository for running a ResNet101 model on CIFAR10 dataset using distributed training. Link to the 
main article can be found [here][1].

### Getting Started

#### Prerequisites

1. Linux (only tested on Linux)
2. PyTorch
3. NVIDIA GPU and CuDNN
    
#### Installation
1. Clone this repository:
    ```py
    git clone https://github.com/naga-karthik/ddp-resnet-cifar
    cd ddp-resnet-cifar
    ```
2. Download the necessary packages:
    ```py
    pip install requirements.txt
    ```
3. If you will be running it on a remote server, then it is probably better to pre-download the dataset than actually 
doing it on-the-fly.
    
    * [CIFAR10 Dataset][2]
    
    * Create a folder named "data" and move the downloaded dataset into the folder.  
    
#### Running the model
From the terminal use the following commands to run the model.
    
1. With default settings:
    ```py
    python mainCIFAR10.py
    ```
2. With other options:
    ```py
    python mainCIFAR10.py --n_epochs=100 --lr=0.001 --batch_size=32
    ```

[1]: https://naga-karthik.github.io/posts/2020/07/pytorch-ddp/
[2]: https://www.cs.toronto.edu/~kriz/cifar.html

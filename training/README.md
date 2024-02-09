# AST-T5: Pretraining

## Dependencies

Tested environment: CUDA 11.3, Python 3.8, PyTorch 1.12.1+cu113, Apex (custom build), Flash Attention (custom build)

Scripts used to set up the environment for reference:

```bash
conda install \
    numpy=1.23.5 pandas=1.5.3 scipy=1.10.0 scikit-learn=1.2.1 matplotlib=3.7.1 seaborn=0.12.2 \
    ipython=8.10.0 nbclassic=0.5.2 notebook=6.5.2 tensorboardx=2.2 qtconsole=5.4.0 \
    jupyter jupyter_client=7.4.9 jupyter_console=6.6.2 jupyter_core=5.2.0 jupyter_server=1.23.4 \
    jupyterlab=3.5.3 jupyterlab_server=2.19.0 \
    requests=2.28.1 beautifulsoup4=4.11.1 lxml=4.9.1 urllib3=1.26.14 \
    mkl=2021.4.0 mkl-service=2.4.0 mkl_fft=1.3.1 mkl_random=1.2.2 gmp=6.2.1 \
    pybind11=2.10.1 cython=0.29.33 \
    chardet=4.0.0 gensim=4.3.0 nltk=3.7 editdistance=0.6.1 \
    black=22.6.0 isort=5.9.3 pylint=2.16.2 pytest=7.1.2 coverage=6.3.2 \
    pyyaml=6.0 tqdm=4.64.1 absl-py=1.3.0 toolz=0.12.0 filelock=3.9.0 bzip2=1.0.8 zstd=1.5.2  \
    ffmpeg=4.2.2 lame=3.100 openh264=2.1.1 x264 \
    gnutls=3.6.15 nettle=3.7.3 bitarray=2.5.1 \
    libiconv=1.16 libidn2=2.3.2 libtasn1=4.16.0 libunistring=0.9.10 libuv=1.44.2
conda install \
    gputil=1.4.0 gpustat=1.0.0 py3nvml=0.2.6 cupy=11.5.0 cymem=2.0.7 thinc=8.0.17 \
    fire=0.4.0 yacs=0.1.8 omegaconf=2.0.6 \
    sacrebleu=2.3.1 sacremoses=0.0.53 fastbpe=0.1.0 spacy=3.3.0 \
    pysnooper=1.1.1 pyarrow=8.0.0 murmurhash=1.0.9 blessed=1.19.1 \
    boost-cpp=1.70.0 libgcc-ng=12.2.0 libstdcxx-ng=12.2.0 cudatoolkit=11.3 -c conda-forge
python -m pip install \
transformers==4.16.2 tokenizers==0.13.2 \
sentencepiece==0.1.97 openpyxl==3.1.2 tensorboard==2.12.0 hydra-core==1.0.7 \
grpcio==1.51.3 tensorboard-data-server==0.7.0 \
datasets==2.9.0

# Check NVCC version
which nvcc  # expected: /usr/local/cuda/bin/nvcc
nvcc --version  # expected: 11.3

# Build and install Apex
rm -rf apex
mkdir -p apex
cp apex.zip apex/
cd apex
unzip apex.zip
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export CUDA_HOME=/usr/local/cuda
python -m pip install -v --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./
cd ..

# Build and install Flash Attention
rm -rf flash-attention
mkdir -p flash-attention
cp flash-attention.zip flash-attention/
cd flash-attention
unzip flash-attention.zip
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export CUDA_HOME=/usr/local/cuda
python -m pip install -v --no-cache-dir ./
```

## Setup Environment

Install Fairseq

```bash
cd training || exit
python -m pip uninstall --yes fairseq  # uninstall if there is any installed version of fairseq and fused_ops
python -m pip uninstall --yes fused_ops
python -m pip install -e .  # install fairseq from source
CUDA_HOME=/usr/local/cuda python -m pip install -e ./fused_ops --global-option="--fused_softmax_dropout"  # install fused ops; tested on CUDA 11.3
```

## Run Pretraining

Sanity check on single node (assume 8 GPUs):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Assume 8 GPUs
export HYDRA_FULL_ERROR=1  # show more error message
export NCCL_DEBUG=INFO  # show more error message
fairseq-hydra-train -m --config-dir examples/t5/config/pl \
--config-name base1m_rpe_mask25_10_2e-4_8gpus_bs8192 \
task.data=/path/to/data/githubcorpus/data-bin \  # data path
hydra.sweep.dir=/path/to/outputs/out_001  # output path
```

If you see something like:

```
{"epoch": 1, "update": 0.003, "loss": "15.686", "nll_loss": "15.686", "total": "233312", "n_correct": "7013.27", "ppl": "52720.5", "accuracy": "3.006", "wps": "1.91693e+06", "ups": "2.07", "wpb": "925887", "bsz": "1288.9", "num_updates": "200", "lr": "4e-06", "gnorm": "2", "clip": "49.5", "loss_scale": "32", "train_wall": "102", "gb_free": "26.2", "wall": "867"}
```

It means that pretraining can be successfully run on a single node.

To pretrain a Base scale model on 8 nodes, we need to run this one each node:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Assume 8 GPUs
export HYDRA_FULL_ERROR=1  # show more error message
export NCCL_DEBUG=INFO  # show more error message
fairseq-hydra-train -m --config-dir examples/t5/config/pl \
--config-name base1m_rpe_mask25_ast10_lb5_ub100_op50_or50_2e-4_64gpus_bs8192 \
distributed_training.distributed_world_size=$(( WORLD_SIZE * 8 )) \
distributed_training.distributed_rank=$(( RANK * 8 )) \
distributed_training.distributed_init_method="tcp://${MASTER_IP}:${MASTER_PORT}" \
task.data=/path/to/data/githubcorpus/data-bin \  # data path
hydra.sweep.dir=/path/to/outputs/out_002  # output path
```

We need to set the following environment variables:

```
WORLD_SIZE: the total number of nodes; "8" for 8 nodes
RANK: the rank of the current node. Should be "0" on node 0, "1" on node 1, ..., and "7" on node 7
MASTER_IP: the IP address or host name of node 0
MASTER_PORT: any available port of node 0; please do not use port number less than 10000 to avoid permission issues
```

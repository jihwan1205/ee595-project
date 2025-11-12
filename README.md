## Environment Setup on slurm compute node

```shell
git clone
cd ee595-project
conda create --prefix /tmp/$USER/.conda/envs/ee595-env python=3.10 -y
conda activate ee595-env
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
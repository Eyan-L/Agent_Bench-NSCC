curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER


curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-latest-Linux-x86_64.sh
source $HOME/miniconda/bin/activate
conda init --all

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda config --add channels conda-forge

conda create -n agent-bench python=3.10 pip -y
conda activate agent-bench
## install dependencies from pyproject.toml 
pip install .

cd ~/agent-bench/agent_bench/harnesses/openclaw/docker
bash build.sh
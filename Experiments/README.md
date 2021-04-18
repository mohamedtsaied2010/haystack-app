# Haystack App

## Installation steps:

### Step1. Create Conda Environment
```
conda create --name haystack
conda activate haystack
conda install -c anaconda pip
pip install git+https://github.com/deepset-ai/haystack.git
pip install urllib3==1.25.4
```

### Step2. Elastic Search Setup
If you are using linux, first [setup docker](https://docs.docker.com/engine/install/ubuntu/)
```
sudo chmod 666 /var/run/docker.sock
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
sudo yum install poppler-utils
pip install pdf2image byaldi
```

There is something to note here.
if set os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' in the code, the excution will be failed.
while set it in the .bashrc file, it can be executed successfully.

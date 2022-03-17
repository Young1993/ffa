# FFA-for-Punctuation-Restoration

## our model
+ bart-large
+ funnel-transformer-xlarge

## Methods
+ parallel_enc_dec.py
+ modeling_funnel.py(talk_matrix)

## Language Models
+ facebook/bart-large
+ funnel-transformer/xlarge

## Directory
+ **main** - Source Code
+ **main/train.py** - Training Process
+ **main/config.py** - Training Configurations
+ **main/res/data/raw** - IWSLT Source Data
+ **main/src/models** - Models
+ **main/src/utils** - Helper Function

## Dependencies
+ python >= 3.8.5
+ jupyterlab >= 3.1.4
+ flair >= 0.8.
+ scikit_learn >= 0.24.1
+ torch >= 1.7.1
+ tqdm >= 4.57.0
+ transformers >= 4.3.2
+ ipywidgets >= 7.6.3

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd FFA_punc_restore
$ cd main
$ pip install pip --upgrade
$ pip install -r requirements.txt
Looking in indexes: http://mirrors.cloud.aliyuncs.com/pypi/simple/
Collecting flair==0.8
  Downloading http://mirrors.cloud.aliyuncs.com/pypi/packages/16/a9/02ab3594958a89c5477f2820a19158187e095763ab6d5d6c0aa5a896087c/flair-0.8-py3-none-any.whl (277 kB)
     |████████████████████████████████| 277 kB 23.4 MB/s
...
...
...
Installing collected packages: urllib3, numpy, idna, chardet, zipp, tqdm, smart-open, six, scipy, requests, regex, PySocks, pyparsing, joblib, decorator, click, wrapt, wcwidth, typing-extensions, tokenizers, threadpoolctl, sentencepiece, sacremoses, python-dateutil, pillow, packaging, overrides, networkx, kiwisolver, importlib-metadata, gensim, future, filelock, cycler, cloudpickle, transformers, torch, tabulate, sqlitedict, segtok, scikit-learn, mpld3, matplotlib, lxml, langdetect, konoha, janome, hyperopt, huggingface-hub, gdown, ftfy, deprecated, bpemb, flair
Successfully installed PySocks-1.7.1 bpemb-0.3.2 chardet-4.0.0 click-7.1.2 cloudpickle-1.6.0 cycler-0.10.0 decorator-4.4.2 deprecated-1.2.12 filelock-3.0.12 flair-0.8 ftfy-5.9 future-0.18.2 gdown-3.12.2 gensim-3.8.3 huggingface-hub-0.0.7 hyperopt-0.2.5 idna-2.10 importlib-metadata-3.7.3 janome-0.4.1 joblib-1.0.1 kiwisolver-1.3.1 konoha-4.6.4 langdetect-1.0.8 lxml-4.6.3 matplotlib-3.4.0 mpld3-0.3 networkx-2.5 numpy-1.19.5 overrides-3.1.0 packaging-20.9 pillow-8.1.2 pyparsing-2.4.7 python-dateutil-2.8.1 regex-2021.3.17 requests-2.25.1 sacremoses-0.0.43 scikit-learn-0.24.1 scipy-1.6.2 segtok-1.5.10 sentencepiece-0.1.95 six-1.15.0 smart-open-4.2.0 sqlitedict-1.7.0 tabulate-0.8.9 threadpoolctl-2.1.0 tokenizers-0.10.1 torch-1.7.1 tqdm-4.57.0 transformers-4.3.2 typing-extensions-3.7.4.3 urllib3-1.26.4 wcwidth-0.2.5 wrapt-1.12.1 zipp-3.4.1
```

## Run
Before training, please take a look at the **config.py** to ensure training configurations.
```
$ cd main
$ vim config.py
$ python train.py
```

## Output
If everything goes well, you should see a similar progressing shown as below.
```
*Configuration*
*Configuration*
model: parallelendecoder
language model: funnel-transformer/xlarge
freeze language model: False
sequence boundary sampling: random
mask loss: False
trainable parameters: 907,906,052
model:
encode_layer.embeddings.word_embeddings.weight	torch.Size([30522, 1024])
encode_layer.embeddings.layer_norm.weight	torch.Size([1024])
encode_layer.embeddings.layer_norm.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.attention.r_w_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.0.0.attention.r_r_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.0.0.attention.r_kernel	torch.Size([1024, 16, 64])
encode_layer.encoder.blocks.0.0.attention.r_s_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.0.0.attention.seg_embed	torch.Size([2, 16, 64])
encode_layer.encoder.blocks.0.0.attention.talk_matrix	torch.Size([16, 16])
encode_layer.encoder.blocks.0.0.attention.q_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.0.0.attention.k_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.0.0.attention.k_head.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.attention.v_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.0.0.attention.v_head.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.attention.post_proj.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.0.0.attention.post_proj.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.attention.layer_norm.weight	torch.Size([1024])
encode_layer.encoder.blocks.0.0.attention.layer_norm.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.ffn.linear_1.weight	torch.Size([4096, 1024])
encode_layer.encoder.blocks.0.0.ffn.linear_1.bias	torch.Size([4096])
encode_layer.encoder.blocks.0.0.ffn.linear_2.weight	torch.Size([1024, 4096])
encode_layer.encoder.blocks.0.0.ffn.linear_2.bias	torch.Size([1024])
encode_layer.encoder.blocks.0.0.ffn.layer_norm.weight	torch.Size([1024])
encode_layer.encoder.blocks.0.0.ffn.layer_norm.bias	torch.Size([1024])
...
...
...
encode_layer.encoder.blocks.2.9.attention.r_w_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.2.9.attention.r_r_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.2.9.attention.r_kernel	torch.Size([1024, 16, 64])
encode_layer.encoder.blocks.2.9.attention.r_s_bias	torch.Size([16, 64])
encode_layer.encoder.blocks.2.9.attention.seg_embed	torch.Size([2, 16, 64])
encode_layer.encoder.blocks.2.9.attention.talk_matrix	torch.Size([16, 16])
encode_layer.encoder.blocks.2.9.attention.q_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.2.9.attention.k_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.2.9.attention.k_head.bias	torch.Size([1024])
encode_layer.encoder.blocks.2.9.attention.v_head.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.2.9.attention.v_head.bias	torch.Size([1024])
encode_layer.encoder.blocks.2.9.attention.post_proj.weight	torch.Size([1024, 1024])
encode_layer.encoder.blocks.2.9.attention.post_proj.bias	torch.Size([1024])
encode_layer.encoder.blocks.2.9.attention.layer_norm.weight	torch.Size([1024])
encode_layer.encoder.blocks.2.9.attention.layer_norm.bias	torch.Size([1024])
encode_layer.encoder.blocks.2.9.ffn.linear_1.weight	torch.Size([4096, 1024])
encode_layer.encoder.blocks.2.9.ffn.linear_1.bias	torch.Size([4096])
encode_layer.encoder.blocks.2.9.ffn.linear_2.weight	torch.Size([1024, 4096])
encode_layer.encoder.blocks.2.9.ffn.linear_2.bias	torch.Size([1024])
encode_layer.encoder.blocks.2.9.ffn.layer_norm.weight	torch.Size([1024])
encode_layer.encoder.blocks.2.9.ffn.layer_norm.bias	torch.Size([1024])
encode_layer.decoder.layers.0.attention.r_w_bias	torch.Size([16, 64])
encode_layer.decoder.layers.0.attention.r_r_bias	torch.Size([16, 64])
encode_layer.decoder.layers.0.attention.r_kernel	torch.Size([1024, 16, 64])
encode_layer.decoder.layers.0.attention.r_s_bias	torch.Size([16, 64])
encode_layer.decoder.layers.0.attention.seg_embed	torch.Size([2, 16, 64])
encode_layer.decoder.layers.0.attention.talk_matrix	torch.Size([16, 16])
encode_layer.decoder.layers.0.attention.q_head.weight	torch.Size([1024, 1024])
encode_layer.decoder.layers.0.attention.k_head.weight	torch.Size([1024, 1024])
encode_layer.decoder.layers.0.attention.k_head.bias	torch.Size([1024])
encode_layer.decoder.layers.0.attention.v_head.weight	torch.Size([1024, 1024])
encode_layer.decoder.layers.0.attention.v_head.bias	torch.Size([1024])
encode_layer.decoder.layers.0.attention.post_proj.weight	torch.Size([1024, 1024])
encode_layer.decoder.layers.0.attention.post_proj.bias	torch.Size([1024])
encode_layer.decoder.layers.0.attention.layer_norm.weight	torch.Size([1024])
encode_layer.decoder.layers.0.attention.layer_norm.bias	torch.Size([1024])
encode_layer.decoder.layers.0.ffn.linear_1.weight	torch.Size([4096, 1024])
encode_layer.decoder.layers.0.ffn.linear_1.bias	torch.Size([4096])
encode_layer.decoder.layers.0.ffn.linear_2.weight	torch.Size([1024, 4096])
encode_layer.decoder.layers.0.ffn.linear_2.bias	torch.Size([1024])
encode_layer.decoder.layers.0.ffn.layer_norm.weight	torch.Size([1024])
encode_layer.decoder.layers.0.ffn.layer_norm.bias	torch.Size([1024])
...
...
...
decode_layer.encoder.layers.11.self_attn.k_proj.weight	torch.Size([1024, 1024])
decode_layer.encoder.layers.11.self_attn.k_proj.bias	torch.Size([1024])
decode_layer.encoder.layers.11.self_attn.v_proj.weight	torch.Size([1024, 1024])
decode_layer.encoder.layers.11.self_attn.v_proj.bias	torch.Size([1024])
decode_layer.encoder.layers.11.self_attn.q_proj.weight	torch.Size([1024, 1024])
decode_layer.encoder.layers.11.self_attn.q_proj.bias	torch.Size([1024])
decode_layer.encoder.layers.11.self_attn.out_proj.weight	torch.Size([1024, 1024])
decode_layer.encoder.layers.11.self_attn.out_proj.bias	torch.Size([1024])
decode_layer.encoder.layers.11.self_attn_layer_norm.weight	torch.Size([1024])
decode_layer.encoder.layers.11.self_attn_layer_norm.bias	torch.Size([1024])
decode_layer.encoder.layers.11.fc1.weight	torch.Size([4096, 1024])
decode_layer.encoder.layers.11.fc1.bias	torch.Size([4096])
decode_layer.encoder.layers.11.fc2.weight	torch.Size([1024, 4096])
decode_layer.encoder.layers.11.fc2.bias	torch.Size([1024])
decode_layer.encoder.layers.11.final_layer_norm.weight	torch.Size([1024])
decode_layer.encoder.layers.11.final_layer_norm.bias	torch.Size([1024])
decode_layer.encoder.layernorm_embedding.weight	torch.Size([1024])
decode_layer.encoder.layernorm_embedding.bias	torch.Size([1024])
decode_layer.decoder.embed_tokens.weight	torch.Size([50265, 1024])
decode_layer.decoder.embed_positions.weight	torch.Size([1026, 1024])
decode_layer.decoder.layers.0.self_attn.k_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.self_attn.k_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.self_attn.v_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.self_attn.v_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.self_attn.q_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.self_attn.q_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.self_attn.out_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.self_attn.out_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.self_attn_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.0.self_attn_layer_norm.bias	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn.k_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.encoder_attn.k_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn.v_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.encoder_attn.v_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn.q_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.encoder_attn.q_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn.out_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.0.encoder_attn.out_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.0.encoder_attn_layer_norm.bias	torch.Size([1024])
decode_layer.decoder.layers.0.fc1.weight	torch.Size([4096, 1024])
decode_layer.decoder.layers.0.fc1.bias	torch.Size([4096])
decode_layer.decoder.layers.0.fc2.weight	torch.Size([1024, 4096])
decode_layer.decoder.layers.0.fc2.bias	torch.Size([1024])
decode_layer.decoder.layers.0.final_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.0.final_layer_norm.bias	torch.Size([1024])
...
...
...
decode_layer.decoder.layers.11.self_attn.k_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.self_attn.k_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.self_attn.v_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.self_attn.v_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.self_attn.q_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.self_attn.q_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.self_attn.out_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.self_attn.out_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.self_attn_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.11.self_attn_layer_norm.bias	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn.k_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.encoder_attn.k_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn.v_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.encoder_attn.v_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn.q_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.encoder_attn.q_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn.out_proj.weight	torch.Size([1024, 1024])
decode_layer.decoder.layers.11.encoder_attn.out_proj.bias	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.11.encoder_attn_layer_norm.bias	torch.Size([1024])
decode_layer.decoder.layers.11.fc1.weight	torch.Size([4096, 1024])
decode_layer.decoder.layers.11.fc1.bias	torch.Size([4096])
decode_layer.decoder.layers.11.fc2.weight	torch.Size([1024, 4096])
decode_layer.decoder.layers.11.fc2.bias	torch.Size([1024])
decode_layer.decoder.layers.11.final_layer_norm.weight	torch.Size([1024])
decode_layer.decoder.layers.11.final_layer_norm.bias	torch.Size([1024])
decode_layer.decoder.layernorm_embedding.weight	torch.Size([1024])
decode_layer.decoder.layernorm_embedding.bias	torch.Size([1024])
fusion_layer.self_attn.in_proj_weight	torch.Size([6144, 2048])
fusion_layer.self_attn.in_proj_bias	torch.Size([6144])
fusion_layer.self_attn.out_proj.weight	torch.Size([2048, 2048])
fusion_layer.self_attn.out_proj.bias	torch.Size([2048])
fusion_layer.linear1.weight	torch.Size([4096, 2048])
fusion_layer.linear1.bias	torch.Size([4096])
fusion_layer.linear2.weight	torch.Size([2048, 4096])
fusion_layer.linear2.bias	torch.Size([2048])
fusion_layer.norm1.weight	torch.Size([2048])
fusion_layer.norm1.bias	torch.Size([2048])
fusion_layer.norm2.weight	torch.Size([2048])
fusion_layer.norm2.bias	torch.Size([2048])
out_layer.weight	torch.Size([4, 2048])
out_layer.bias	torch.Size([4])
device: cuda
train size: 8950
val size: 1273
ref test size: 54
asr test size: 54
batch size: 8
train batch: 1118
val batch: 160
ref test batch: 7
asr test batch: 7
valid win size: 8
if load check point: False

Training...
Loss:0.9846:   1%|█▊                                                                                                                                                                                                    | 5/559 [00:02<03:30,  2.63it/s]
```

## Note
1. 可以在 ./main/src/modes/parallel_enc_dec.py 修改预训练模型

## Authors
* **Kebin Fang** -fkb@zjuici.com

## BibTex
```

```

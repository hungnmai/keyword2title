### GENERATE TITLE FROM KEYWORDS

---

### 1. Install libraries
Before installing libraries, ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installed

* Create environment with conda
```shell
conda create -n keyword2title python=3.8
```
* **k2t** (stand for keyword to title) is name of environment
* **python=3.8** to define python version is using

* Activate environment:
```shell
conda activate k2t
```

* cd to app:
```shell
cd /path/to/app
```
* Install libraries
```shell
pip install -r requirements.txt
```

### 2. Run application
We provide 2 models that is T5 and BART to generate title from keywords. 
Before running application you have to download them at here:
* [BART](https://drive.google.com/file/d/1vscEhytL3_cfi963tpIXbnN6NyBTucBB/view?usp=share_link)
* [T5](https://drive.google.com/file/d/1i8xrv9FkqDqP12YwYulp4hcj9YMtvWK1/view?usp=share_link)

Then, you need to unzip and place 2 models in [models](models) folder. For example:
- models
  - [checkpoint-7048](checkpoint-7048)
  - [checkpoint-2024](checkpoint-2024)
#####  2.1 To run BART model:


```shell
sh run_1.sh
```

After running success, You call API from below URL:
http://localhost:8098/generate_title_from_kw_v1?spans=keywords

* keywords are list of words separated by comma
* For example: You can generate title from '**woman, man, love**'

```
http://localhost:8098/generate_title_from_kw_v1?spans=woman, man, love
```

#####  2.2. To run T5 model

```shell
sh run_2.sh
```

After running success, You call API from below URL:

http://localhost:8098/generate_title_from_kw_v2?spans=keywords

* keywords are list of words separated by comma
* For example: You can generate title from '**woman, man, love**'

```
http://localhost:8098/generate_title_from_kw_v1?spans=woman, man, love
```
This repository is a clone of the synthyverse library, containing the implementation and experiments of XGenBoost.

We use Python 3.10.

# Usage:
- Easiest and cleanest way to reproduce results is to run the bash script main.sh and modify the generator and dataset variables inside:
```bash
bash main.sh
```

- Otherwise run for the different generators and datasets. In this case it works best to use separate conda environments for each generator.
```bash
pip install -r requirements/generators/base.txt
pip install -r requirements/generators/{generator}.txt
pip install -r requirements/evaluation/eval.txt
python main.py --generator {generator} --dataset {dataset} 
```

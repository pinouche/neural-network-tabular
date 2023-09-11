# Preliminary experiments for training neural networks on tabular data

## Set-up

* Use Poetry to manage the dependencies 
* Use a Python 3.10 (or above) virtual environment (it's recommended to use conda to create the environment)
* Activate the virtual environment
* Install the dependencies using Poetry


```bash
conda create -n <my_env> python=3.10
conda activate <my_env>
poetry install
```

To run the script, run the following commands:


```
python3 main.py --config ../config_feed_forward.yml
```

for the simple MLP network, or:

```
python3 main.py --config ../config_transformer.yml
```

for the transformer model (encoder only).

# cc-tweets

Use conda

```
conda create -n <env_name> python=3.8
conda activate <env_name>
conda develop .
conda install <packages_as_errors_are_rased>  # sorry, will fix this soon
```

go to `config.py` to set up the necessary data paths. 

run numbered scripts in numbered directories (skip `0.data_retrieval` if you already have access to the raw data) in order, observe result saved to `WORKING_DIR` as specified.

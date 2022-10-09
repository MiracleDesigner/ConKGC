## Quick Start

### Process data
Run the following command to preprocess the datasets.

```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`<dataset>` is the name of any dataset folder in the `./data` directory. In our experiments, the three datasets used are: `fb15k-237`, `yago3-10`  and `wn18rr`. 
`<gpu-ID>` is a non-negative integer number representing the GPU index.

### Train the model
Then the following commands can be used to train the proposed model in the paper. By default, dev set evaluation results will be printed when training terminates.

```
./experiment.sh configs/<dataset>.sh --train <gpu-ID>
```

### Evaluate the model
To generate the evaluation results of the model, simply change the `--train` flag in the commands above to `--inference`. 

```
./experiment.sh configs/<dataset>.sh --inference <gpu-ID>
```

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).


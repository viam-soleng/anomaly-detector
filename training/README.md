# Viam Custom Training Script Workflow

This folder contains a ML model training script which is used to train an isolation forest model with data stored in the Viam cloud backend.
The model is then uploaded into the Viam Registry and is ready to be deployed onto a Viam machine running `viam-server`.

## Create Custom Training Script Bundle (tar.gz)

The Viam custom training script workflow requires a Python source distribution.
To create this tarball change into the `training` folder and run the python command:

```shell
cd training
python setup.py sdist
```

## Upload Custom Training Script

The tarball can then be uploaded into the Viam Registry using the following viam CLI commmand (You will have to be authenticated: `viam login`):

```shell
viam training-script upload --path=dist/train_if-0.1.tar.gz --org-id=< YOUR ORG ID > --script-name="IsolationForest"
```

## Submit a Custom Training Job

Once the training script is uploaded to the Viam Registry, you can run the training process with the following command:

```shell
viam train submit custom from-registry --dataset-id=< DATASET ID - NOT USED BUT REQUIRED > --org-id=< ORG ID > --model-name="isolation_forest" --script-name="< ORG ID >":IsolationForest" --version=< SCRIPT VERSION FROM REGISTRY >
```

## Get Training Job Logs

Once you have successfully submitted the custom training job, you can access log information with the following command:

```shell
viam train get --job-id=< JOB ID >
```

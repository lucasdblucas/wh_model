# wh_model
Algorithm used in the development of a method for estimating weight and height from multi-view images of an individual. The code supports both training and inference based on provided examples

## Tools

* Python 3.10
* Cuda 11.7
* Conda 23.3

### How to use

* First, download the Anaconda tool from the distributor's website.
```
https://www.anaconda.com/about-us
```

* Create a virtual environment from the file according to the project specifications. 
The YAML file is at the root of the project.

```
conda env create -f environment.yml
```
* Then activate the conda environment.

```
conda activate env_name
```

* To download the project, clone it directly from git by executing the following command:

```
https://github.com/lucasdblucas/wh_model.git
```

* Then, go to the "src/bashs" folder and run the bash file to execute an experiment. The configuration files are located in the "src/config" folder, and they pass the parameters to run the experiment.
WideResnet Experiment:
```
bash execute_wh_21.sh
```
Scale-Equivariant WideResnet Experiment:
```
bash execute_wh_22.sh
```

Please, feel free to contact us for any questions: lucas.daniel.bp@gmail.com

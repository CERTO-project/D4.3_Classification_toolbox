# CERTO - D4.3 Classification Toolbox


## Installation
To install the toolbox we recommend using a conda environment. To clone the repository and install the dependencies:
```
git clone git@github.com:CERTO-project/D4.3_Classification_toolbox.git
cd D4.3_Classification_toolbox
conda create -f environment.yml
conda activate certo
python setup.py install
```

## Worked Example
The repository includes a worked example for the Curonian Lagoon to demonstrate how to build a training datset, run the
toolbox to identify the best model, and then apply this model to product a set of optical water type classes. 

To start the Jupyter notebook, run:
```
jupyter-notebook --notebook-dir=notebooks --ip=*
```

### Support 
This toolbox was created by PML as part of the CERTO Project (https://certo-project.org/). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 870349.

If you need support in using this toolbox please email contact@certo-project.org or tweet @CERTO_project




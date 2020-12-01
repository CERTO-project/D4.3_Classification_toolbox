# Ocean water spectra classifier

This will become a python module for the fuzzy clustering of optical spectra.

## __Specification__: _Solution_

* __Easy to use__: _Scikit-learn syntax_

Enables integration into [scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Useful for grid searching for the optimal parameters and joining up pre and post processing steps.

```
import WaterClassifier
wc = WaterClassifier(method='cmeans')

# generate a set of clusters to fit the data
# cluster configuration is held within the WaterClassifier instance
wc.fit(X, **kwargs)

# classify a set of data
Y = wc.transform(X)

# access model
wc.model
>> model.instance....
```

* __Self consistent repo__: _conda environment `.yml` file_

* __Flexible__:_method kwarg `method='cmeans'`_

## Plan of action

1) build a framework for clusting data and scoring the cluster set
2) use framework to test out different clustering algorithms with different hyper parameters
3) find the best hyper parameters to use as default
4) generate a best effort class set for different sensors
4) 

overview google doc: https://docs.google.com/spreadsheets/d/1uoGp7u3A-hUjuZtvD6z0T5eiK2pyHwqqYJemsZEgY-g/edit?ts=5fc0d0df#gid=0

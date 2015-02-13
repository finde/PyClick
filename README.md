Information Retrieval 2 - Evluation of Click Models
====

This project is a fork of [PyClick](http://www.github.com/markovi/PyClick). All information about PyClick can be found there.

## How to Use

#### Installation
To install PyClick, run the following command from the project's root:

 ```python setup.py install```

It currently additionally to PyClick implements the TCM Model
* **TCM**: Zhang, Yuchen and Chen, Weizhu and Wang, Dong and Yang, Qiang. **User-click Modeling for Understanding and Predicting Search-behavior.**  In *Proceedings of KDD'11*, pages 1388 - 1396, 2011.


#### Example
To compare TCM with the other implemented models use:
 
 ```python example/YandexExample.py data_set number_of_sessions```

This will compare the models on a Yandex Click log, for which the description is found [here](http://imat-relpred.yandex.ru/en/datasets). This dataset is not included here. The number_of_sessions variable will default to 1000


## Acknowledgements
* This project is supervised by [Ilya Markovi](http://github.com/markovi) and [Artem Grotov](http://github.com/agrotov)

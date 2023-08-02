# fl-stacking-personalization

Code that accompanies my undergrad thesis *Personalized Federated Learning via Model Stacking* (*Aprendizaje Federado Personalizado Mediante Ensamblaje De Modelos*).

To run the experiments you will have to download the datasets from the links included in `datasets.py` and run `partition_loader.py`.

The dataset summary table and example partition graphs from Section 3.1 were generated in `dataset_info.ipynb`.
Data pooling experiments (Section 3.2) were run and analyzed with `data_experiments.py` and `data_pooling_experiments.ipynb`. Model sharing experiments (Section 3.3) with `model_experiments.py` and `model_experiments.ipynb`. The client-server implementation from Chapter 4 is in `example_implementation/`.
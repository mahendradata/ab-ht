
# AB-HT

This repository is a proof of concept of algorithms described in [AB-HT: An Ensemble Incremental Learning Algorithm for Network Intrusion Detection Systems](https://doi.org/10.1109/ICoDSA55874.2022.9862833) paper.

## Installation

Clone repo and install requirements.txt in a $ python>=3.8.0 environment.

```console
$ git clone https://github.com/mahendradata/ab-ht.git # clone
$ cd ab-ht
$ pip install -r requirements.txt  # install
```

## Prepare the dataset

Before running the main program, we need to prepare the CICIDS2017 dataset.
Run the following commands to prepare the dataset.

```console
$ cd datasets/
$ bash step1-download.sh
$ python step2-preprocessing.py
$ python step3-merging-monday.py
```

## Running other programs

After the dataset is ready, you can run other programs in this repository. But first, let create a folder for the output file.

```console
$ cd ..  #back to the root folder of the repository
$ mkdir outputs
```

Then run the experimental programs, for example:

Running the `AB-HT` program:

```console
$ python AB-HT.py conf/AB-HT.json outputs/AB-HT.log
```

Running the `HT` program:

```console
$ python HT.py conf/HT.json outputs/HT.log
```

Running the `HATT` program:

```console
$ python HATT.py conf/HATT.json outputs/HATT.log
```

Running the `DT` program:

```console
$ python DT.py conf/DT.json outputs/DT.log
```

Running the `AB-DT` program:

```console
$ python AB-DT.py conf/AB-DT.json outputs/AB-DT.log
```
 

## Folder Structure 

- `datasets`: contains the preprocessed dataset. 
- `conf`: contains configuration files in json format.
- `lib`:
    - `dataset`: contains libraries to preprocess and manipulate the dataset.
    - `model`: contains libraries to create the models.
    - `util`: contains utility libraries.

These are additional folders. You need to create this folder on your own.
- `outputs`: a folder to save the experimental data.

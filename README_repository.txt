This is program repository for "Computational analyses of linguistic features with schizophrenic and autistic traits along with formal thought disorders" from ICMI2023.

###
### Prerequisite
###

- Anaconda
- Python3.7

Please construct the environment using env/py37_FTD.yaml file.
For example, run the following command:
1. conda create -f=py37_FTD.yaml
2. conda activate py37_FTD

###
### Step0: dataset download
###

The dataset is only available for academic research purpose.
Please submit the following application form to get access.

[Dataset application URL]

Once you get the dataset, place the dataset in "dataset" directory as follows:

- dataset
--- data_audio
--- data_audio_separated180
--- data_label
--- data_transcript
--- data_transcript_separated180

###
### Step1: data preparation
###

Run program from number 10 to 14 one-by-one.


###
### Step2: correlation calculation
###

20: run correlation calculation program in 30, 60, 180 seconds
21: run correlation calculation program separated in each 60 seconds


###
### Step3: ablation study
###

30: create feature-set for ablation study
31: run ablation study with data in 30, 60, 180 seconds
32: run ablation study with data separated in each 60 seconds


###
### If you find this repository is helpful, please cite the following paper.
###

Takeshi Saga, Hiroki Tanaka, Satoshi Nakamura, "Computational analyses of linguistic features with schizophrenic and autistic traits along with formal thought disorders", ICMI2023
# REI-Net-MI
This project propose a framework called Reference Electrode Standardization Interpolation Network (REI-Net) for for motor imagery (MI) analysis.
## Code Execution Instructions

### - Data Generation

Run EEGML/Prep_time_domain.py ./MI/model/EEGML_XXX.yml

Notes:
To generate different datasets, modify the corresponding .yml file.
Data generation is divided into preprocessing and training data folders. The save paths need to be modified in prep_time_domain.
(1): Preprocessed data is used for operations such as data filtering, downsampling, and rereferencing, and then the subject_dependent_setting_spilt() function is used to divide the data. The data will be saved in the ProcessData folder.
(2): Training data is divided within or across subjects through related functions. The divided data will be saved in the TrainData folder.

**- Data Training**

Run EEGML/MI/Train/train.py ../model/EEGML_XXX.yml

Notes:
Set the path for the training data in the .py file.
The corresponding function for REI-Net in MI/EEGML.py is DADLNet_noAttention().

**- Data Testing**

Run EEGML/MI/Train/train.py ../model/EEGML_XXX.yml

Notes:
Set the path for the training data in the .py file.
Set the model training results folder in the .yml file.
Virtual Environment
Chinese Medicine Hospital: mi
Dongfang Polytechnic: mi3 mirror, Joe's training project

**- Transfer Learning Code Execution**

1. Transfer Learning Data Generation
Run EEGML/MI/Transfer/DataGenerate.py, with the configuration file being Transfer_dataset_name_medical.yml in the yml folder.
Note that the Chinese Medicine Hospital server already has generated data, so this step can be skipped.

2. Running Transfer Learning Code
Run EEGML/MI/Transfer/DataGenerate.py, with the corresponding configuration file being Transfer_BCI_medical.yml. Relevant parameters are already set, but the name of the result table to be saved needs to be modified.

Notes
Transfer learning does not save model parameters.

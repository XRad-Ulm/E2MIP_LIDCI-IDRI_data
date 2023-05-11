# [E2MIP Challenge](https://e2mip.github.io/) on the LIDC-IDRI dataset

This repository  contains:
1. information about the submission for the Challenge
2. the structure of the training and testing data folders that will be used for your submission
3. sample code that creates data folders with random train/test split for the classification and segmentation task

### 1. Challenge submission:
**The following information is preliminary. Final submission information will be available soon.**

* Please provide your complete algorithm for training and predicting in a docker script.
  Further information about how the script should look like will be published here soon.
  * script takes as input the path to the "training_data" and "testing_data_classification" or "testing_data_segmentation" folders
  * script outputs predicted segmentation in a newly created folder "testing_data_prediction_classification" or "testing_data_prediction_segmentation". 
The predictions need to be filed in the folder in a certain folder structure (see 2. Data for Challenge)
* Besides the performance metric also the energy consumption during training and evaluation is being measured
  and both determine the Challenge ranking.
### 2. Data for Challenge:
The folder structure of the training and testing data used for evaluating your code will look like the following:
The "training_data" folder contains various information about the scans and nodules.
```bash
training_data
├── scan_0
│   ├── image_total.nii
│   ├── segmentation_total.nii
│   ├── nodule_0
│       ├── annotation_0
│           ├── mask.nii
│           ├── bbox.txt
│           ├── centroid.txt
│           ├── mal.txt
│           └── attri.txt
│       ├── annotation_1
│       ...
│   ├── nodule_1
│   
├── scan_2
    ...
...
```
For classification the folder of the testing data looks like the following:
```bash
testing_data_classification
├── scan_1
│   ├── nodule_0.nii
│   ├── nodule_1.nii
│   ├── nodule_2.nii
│   └── nodule_3.nii
├── scan_5
│   ├── nodule_0.nii
    ...
...
```
For segmentation the folder of the testing data looks like the following:
```bash
testing_data_segmentation
├── scan_1
│   └── image_total.nii
├── scan_5
│   └── image_total.nii
...
```
The folder structure of the classification predictions that your script should create from  "testing_data_classification" should have the following structure:
```bash
testing_data_prediction_classification
├── scan_1
│   ├── nodule_0.txt
│   ├── nodule_1.txt
│   ├── nodule_2.txt
│   └── nodule_3.txt
├── scan_5
│   ├── nodule_0.txt
    ...
...
```
The folder structure of the segmentation predictions that your script should create from  "testing_data_segmentation" should have the following structure:
```bash
testing_data_prediction_segmentation
├── scan_1
│   └── prediction_total.nii
├── scan_5
│   └── prediction_total.nii
...
```

#### 3. Sample Data
This repository contains code that creates folders named "training_data", "testing_data_classification", "testing_data_solution_classification"
, "testing_data_segmentation", "testing_data_solution_segmentation" for a random train/test split.
This data extractor works with the [pylidc](https://pylidc.github.io/install.html) framework. 
For installation, please follow the [instructions](https://pylidc.github.io/install.html), 
which includes [downloading](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) _Images (DICOM, 125GB)_.

### Debugging:
- ``RuntimeError: Could not establish path to dicom files. Have you specified the `path` option in the configuration file /home/[user]/.pylidcrc?``
  This error occurs during data loading, if you did not specify the path to the LIDC-IDRI dicom files.
  To solve the error
  1. Download _Images (DICOM, 125GB)_ from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)
  2. Add the path /[...]/LIDC-IDRI-classic/manifest-1600709154662/LIDC-IDRI to the configuration file /home/[user]/.pylidcrc

For further questions about this code, please contact luisa.gallee@uni-ulm.de

Find this [repository](https://github.com/LuisaGallee/E2MIP_LIDCI-IDRI_classification) as starting point for classification and this [repository](https://github.com/LuisaGallee/E2MIP_LIDCI-IDRI_segmentation) for starting point for segmentation.
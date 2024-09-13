import os
import sys
import random
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import pylidc as pl
from tqdm import tqdm
import psutil


def get_pylidc_directory():
    '''
    Gets the directory path for the pylidc configuration file.

    This function looks for the .pylidcrc file on Linux and Mac or pylidc.conf 
    on Windows to determine the path where DICOM data is stored. On Linux and 
    Mac, the file should be located at /home/[user]/.pylidcrc, while on Windows,
    it should be located at C:\\Users\\[User]\\pylidc.conf.

    Users should ensure that the correct path is specified in their 
    configuration file.

    Returns:
        str: The directory path if found, otherwise None.
    '''
    # Define the path for the configuration file based on the operating system.
    # In this case for linux.
    config_path = os.path.expanduser('~/.pylidcrc')

    if not os.path.exists(config_path):
        print('Configuration file .pylidcrc not found.')
        return None

    # Read the configuration file and search for the 'path' entry
    with open(config_path, 'r') as config_file:
        for line in config_file:
            if line.startswith('path'):
                return line.split('=')[1].strip()

    print('No directory path found in the configuration file.')
    return None


def create_traintest_folder(base_dir=None, pylidc_dir=None):
    '''
    Creates folders for training and testing datasets.

    This function checks if there is sufficient disk space in the base directory 
    and then creates the necessary folders for storing training and testing data. 
    It processes scans from the LIDC dataset, splits them into training and testing 
    sets, and saves the respective data in the corresponding folders.

    Args:
        base_dir (str): The base directory where folders will be created.
        pylidc_dir (str): The directory containing the pylidc DICOM data.

    Raises:
        FileExistsError: If any of the target directories already exist.
    '''
    base_path = Path(base_dir)

    required_space_gb = calculate_required_space(pylidc_dir)

    if not check_disk_space(base_path, required_space_gb):
        print(f'Not enough free disk space in chosen dir : {base_dir}. '
              'Estimated required space: {required_space_gb:.2f} GB.')
        return

    directories = [
        'training_data',
        'testing_data_segmentation',
        'testing_data_solution_segmentation',
        'testing_data_classification',
        'testing_data_solution_classification'
    ]

    for directory in directories:
        path = base_path / directory
        if path.exists():
            raise FileExistsError(f'Path "{path}" exists. '
                                  'Please choose another base path.')
        else:
            print('Added directory: ', path)
            path.mkdir(parents=True, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    scan_idx = list(range(len(scans)))
    random.shuffle(scan_idx)
    testing_idx = scan_idx[:int(0.1 * len(scan_idx))]
    current_sample_idx = 0
    running_idx = 0

    with tqdm(total=len(scans), desc='Processing scans', unit='scan') as pbar:
        while running_idx < len(scans):
            scan_vol, scan_segm = process_scan(scans[scan_idx[running_idx]])

            if np.max(scan_segm) < 1:
                print(f'Scan {running_idx} has max segmentation value: '
                      '{np.max(scan_segm)}')
                running_idx += 1
                pbar.update(1)
                continue

            if current_sample_idx in testing_idx:
                save_test_data(base_path, current_sample_idx, scan_vol, scan_segm,
                               scans[scan_idx[running_idx]].cluster_annotations())
            else:
                save_training_data(base_path, current_sample_idx, scan_vol,
                                   scan_segm, scans[scan_idx[running_idx]].cluster_annotations())

            current_sample_idx += 1
            running_idx += 1
            pbar.update(1)

    print('\n---------------------------------------------------')
    print('Successfully processed and split data from LIDC-IDRI!')
    print('---------------------------------------------------')


def calculate_required_space(directory):
    '''
    Calculates the required disk space based on the size of files in the specified directory.

    This function iterates through all files in the given directory and sums their sizes to estimate 
    the total storage needed. It assumes that the target directories will require approximately twice 
    the space of the original data.

    Args:
        directory (str): The directory to calculate the required space for.

    Returns:
        float: The estimated required space in gigabytes (GB).
    '''
    print('Calculating required space in pylidc directory:', directory)
    total_size = 0

    # Get the total number of files to process
    total_files = sum(len(files) for _, _, files in os.walk(directory))

    # Progress bar for file size calculation
    with tqdm(total=total_files, desc='Calculating file sizes', unit='file') as pbar:
        # Iterate through all files in the directory and add their sizes
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                # File size in bytes
                total_size += os.path.getsize(file_path)
                pbar.update(1)

    total_size_gb = total_size / (1024 ** 3)  # Convert to gigabytes
    # Assume target directories require about twice the space
    required_space_gb = total_size_gb * 2
    print(f'Estimated required space: {required_space_gb:.2f} GB')
    return required_space_gb


def check_disk_space(directory, required_space_gb):
    '''
    Checks if there is enough disk space available in the specified directory.

    This function calculates the available disk space in the given directory 
    and compares it with the required space to determine if there is sufficient 
    storage available.

    Args:
        directory (str): The directory to check for available disk space.
        required_space_gb (float): The required space in gigabytes (GB).

    Returns:
        bool: True if there is enough space available, otherwise False.
    '''
    usage = shutil.disk_usage(directory)
    free_space_gb = usage.free / (1024 ** 3)  # Umrechnung in Gigabyte
    print(f'Free disk space: {free_space_gb:.2f} GB')
    return free_space_gb >= required_space_gb


def process_scan(scan):
    '''
    Processes a scan to generate volume and segmentation data.

    This function converts the scan data to a volume, clusters the annotations, 
    and creates a segmentation mask for the nodules detected in the scan.

    Args:
        scan (pylidc.Scan): The scan object to process.

    Returns:
        tuple: A tuple containing the volume array and the segmentation array.
    '''
    scan_vol = scan.to_volume()
    nodules = scan.cluster_annotations()
    scan_segm = np.zeros_like(scan_vol)

    for nod in nodules:
        if 3 <= len(nod) <= 4:
            for nod_anni in nod:
                nod_anni_mask = nod_anni.boolean_mask()
                nod_anni_bbox = nod_anni.bbox()
                scan_segm[nod_anni_bbox] += nod_anni_mask

    scan_segm[np.where(scan_segm > 0)] = 1
    return scan_vol, scan_segm


def save_test_data(base_path, sample_idx, scan_vol, scan_segm, nodules):
    '''
    Saves the test data for segmentation and classification tasks.

    This function creates directories for storing segmentation and classification 
    data for testing purposes. It saves the scan volume and segmentation data 
    as well as the classification data for each nodule.

    Args:
        base_path (Path): The base directory path.
        sample_idx (int): The sample index.
        scan_vol (ndarray): The scan volume array.
        scan_segm (ndarray): The scan segmentation array.
        nodules (list): List of nodule annotations.
    '''
    test_seg_dir = base_path / f'testing_data_segmentation/scan_{sample_idx}'
    test_sol_seg_dir = base_path / f'testing_data_solution_segmentation/scan_{sample_idx}'
    test_class_dir = base_path / f'testing_data_classification/scan_{sample_idx}'
    test_sol_class_dir = base_path / f'testing_data_solution_classification/scan_{sample_idx}'

    for dir_path in [test_seg_dir, test_sol_seg_dir, test_class_dir, test_sol_class_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Save segmentation data
    nib.save(nib.Nifti1Image(scan_vol, affine=np.eye(4)),
             test_seg_dir / 'image_total.nii')
    nib.save(nib.Nifti1Image(scan_segm, affine=np.eye(4)),
             test_sol_seg_dir / 'segmentation_total.nii')

    # Save classification data
    for i, nod in enumerate(nodules):
        if 3 <= len(nod) <= 4:
            save_classification_data(
                nod, test_class_dir, test_sol_class_dir, sample_idx, i)


def save_training_data(base_path, sample_idx, scan_vol, scan_segm, nodules):
    '''
    Saves the training data for segmentation tasks.

    This function creates a directory for storing training data and saves the 
    scan volume, segmentation data, and nodule information for each sample.

    Args:
        base_path (Path): The base directory path where the data will be stored.
        sample_idx (int): The sample index.
        scan_vol (ndarray): The scan volume array.
        scan_segm (ndarray): The scan segmentation array.
        nodules (list): List of nodule annotations.
    '''
    train_dir = base_path / f'training_data/scan_{sample_idx}'
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save segmentation data
    nib.save(nib.Nifti1Image(scan_vol, affine=np.eye(4)),
             train_dir / 'image_total.nii')
    nib.save(nib.Nifti1Image(scan_segm, affine=np.eye(4)),
             train_dir / 'segmentation_total.nii')

    # Save nodule data
    for i, nod in enumerate(nodules):
        if 3 <= len(nod) <= 4:
            nod_dir = train_dir / f'nodule_{i}'
            nod_dir.mkdir(parents=True, exist_ok=True)
            for nod_anni in range(len(nod)):
                save_nodule_data(nod[nod_anni], nod_dir, nod_anni)


def save_classification_data(nod, class_dir, sol_class_dir, sample_idx, i):
    '''
    Saves the classification data for nodules.

    This function saves the cropped-out cube of the nodule, the bounding box, and 
    the corresponding malignancy labels for classification tasks.

    Args:
        nod (list): The list of nodule annotations.
        class_dir (Path): The directory for classification data.
        sol_class_dir (Path): The directory for solution classification data.
        sample_idx (int): The sample index.
        i (int): The nodule index.
    '''
    nod_mal = [nod_anni.malignancy for nod_anni in nod]
    target_label = [1, 0] if np.mean(nod_mal) < 3 else [0, 1]

    for nod_anni in range(len(nod)):
        mask = nod[nod_anni].boolean_mask()
        bbox = nod[nod_anni].bbox()
        vol = nod[nod_anni].scan.to_volume()
        centroid_ijk = nod[nod_anni].centroid
        cropout_cube_size = 48
        cropout_bor = np.array(
            [[0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]]])

        # Berechnung der Zuschneidegrenzen
        for d in range(3):
            cropout_bor[d, 0] = max(
                0, int(centroid_ijk[d] - cropout_cube_size // 2))
            cropout_bor[d, 1] = min(vol.shape[d], int(
                centroid_ijk[d] + cropout_cube_size // 2))

        nodule_cropout_cube = vol[
            cropout_bor[0, 0]:cropout_bor[0, 1],
            cropout_bor[1, 0]:cropout_bor[1, 1],
            cropout_bor[2, 0]:cropout_bor[2, 1]
        ]
        nodule_cropout_cube_nii = nib.Nifti1Image(
            nodule_cropout_cube, affine=np.eye(4))
        nib.save(nodule_cropout_cube_nii, class_dir / f'nodule_{nod_anni}.nii')
        np.savetxt(sol_class_dir / f'nodule_{nod_anni}.txt', np.array(target_label).astype(np.int16), delimiter=',')


def save_nodule_data(nod_anni, nod_dir, nod_anni_idx):
    '''
    Saves the annotation data for a single nodule.

    This function creates a directory for storing annotation data of a specific nodule 
    and saves the mask, bounding box, centroid, attributes, and malignancy information.

    Args:
        nod_anni (pylidc.Annotation): The annotation object for a single nodule.
        nod_dir (Path): The directory where the nodule data will be stored.
        nod_anni_idx (int): The index of the annotation.
    '''
    anno_dir = nod_dir / f'annotation_{nod_anni_idx}'
    anno_dir.mkdir(parents=True, exist_ok=True)

    # Save annotation data
    nod_anni_mask = nod_anni.boolean_mask().astype(np.int16)
    nod_anni_bbox = np.array([
        [nod_anni.bbox()[0].start, nod_anni.bbox()[0].stop],
        [nod_anni.bbox()[1].start, nod_anni.bbox()[1].stop],
        [nod_anni.bbox()[2].start, nod_anni.bbox()[2].stop]
    ]).astype(np.int16)
    nod_anni_centroid = nod_anni.centroid.astype(int)
    nod_anni_attri = [
        nod_anni.subtlety,
        nod_anni.internalStructure,
        nod_anni.calcification,
        nod_anni.sphericity,
        nod_anni.margin,
        nod_anni.lobulation,
        nod_anni.spiculation,
        nod_anni.texture
    ]
    nod_anni_malig = [nod_anni.malignancy]

    # Save as files
    nib.save(nib.Nifti1Image(nod_anni_mask, affine=np.eye(4)),
             anno_dir / 'mask.nii')
    np.savetxt(anno_dir / 'centroid.txt', nod_anni_centroid, delimiter=',')
    np.savetxt(anno_dir / 'bbox.txt', nod_anni_bbox, delimiter=',')
    np.savetxt(anno_dir / 'attri.txt', nod_anni_attri, delimiter=',')
    np.savetxt(anno_dir / 'mal.txt', nod_anni_malig, delimiter=',')


if __name__ == '__main__':

    PYLIDC_DIR = get_pylidc_directory()

    if PYLIDC_DIR:
        print('Found directory: ', PYLIDC_DIR)
    else:
        print('No directory found.')
        sys.exit(0)

    # User-defined base directory where the data will be saved
    # The user can specify the desired path where the processed data will be stored.
    DIR = '/ForschungA/datasets/LIDC_Preprocessed2/'

    # Check if the specified directory exists
    if not os.path.exists(DIR):
        print(f'The specified directory "{DIR}" does not exist. '
              'Please check the path and try again.')
        sys.exit(0)

    create_traintest_folder(base_dir=DIR, pylidc_dir=PYLIDC_DIR)

import sys
import os
from create_traintest_data import create_traintest_folder
from create_traintest_data import get_pylidc_directory
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

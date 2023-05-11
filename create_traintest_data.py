import pylidc as pl
import numpy as np
import os
import random
import shutil
import nibabel as nib


def create_traintest_folder():
    if os.path.isdir("training_data"):
        shutil.rmtree('training_data')
    if os.path.isdir("testing_data_segmentation"):
        shutil.rmtree('testing_data_segmentation')
    if os.path.isdir("testing_data_solution_segmentation"):
        shutil.rmtree('testing_data_solution_segmentation')
    if os.path.isdir("testing_data_classification"):
        shutil.rmtree('testing_data_classification')
    if os.path.isdir("testing_data_solution_classification"):
        shutil.rmtree('testing_data_solution_classification')
    os.makedirs("training_data")
    os.makedirs("testing_data_segmentation")
    os.makedirs("testing_data_solution_segmentation")
    os.makedirs("testing_data_classification")
    os.makedirs("testing_data_solution_classification")

    scans = pl.query(pl.Scan).all()
    scan_idx = list(range(len(scans)))
    random.shuffle(scan_idx)
    testing_idx = scan_idx[:int(0.1 * len(scan_idx))]
    current_sample_idx = 0
    running_idx = 0
    while running_idx < len(scans):
        print("Creating random train/test data split, creating folders \"training_data\", \"testing_data_segmentation\", "
              "\"testing_data_solution_segmentation\", \"testing_data_classification\", \"testing_data_solution_classification\", "
              "currently at scan " + str(running_idx)+" of "+str(len(scans)))
        scan_vol = scans[scan_idx[running_idx]].to_volume()
        nodules = scans[scan_idx[running_idx]].cluster_annotations()
        scan_segm = np.zeros_like(scan_vol)
        for i, nod in enumerate(nodules):
            if len(nod) >= 3 and len(nod) <= 4:
                for nod_anni in range(len(nod)):
                    nod_anni_mask = nod[nod_anni].boolean_mask()
                    nod_anni_bbox = nod[nod_anni].bbox()
                    scan_segm[nod_anni_bbox] += nod_anni_mask
        scan_segm[np.where(scan_segm > 0)] = 1
        if np.max(scan_segm) < 1:
            print("This scan has max segm val: " + str(np.max(scan_segm)))
            running_idx += 1
            continue

        if current_sample_idx in testing_idx:
            os.makedirs("testing_data_segmentation/scan_" + str(current_sample_idx))
            os.makedirs("testing_data_solution_segmentation/scan_" + str(current_sample_idx))
            scan_vol_nii = nib.Nifti1Image(scan_vol, affine=np.eye(4))
            nib.save(scan_vol_nii, "testing_data_segmentation/scan_" + str(current_sample_idx) + "/image_total.nii")
            scan_segm_nii = nib.Nifti1Image(scan_segm, affine=np.eye(4))
            nib.save(scan_segm_nii,
                     "testing_data_solution_segmentation/scan_" + str(current_sample_idx) + "/segmentation_total.nii")
            os.makedirs("testing_data_classification/scan_" + str(current_sample_idx))
            os.makedirs("testing_data_solution_classification/scan_" + str(current_sample_idx))
            for i, nod in enumerate(nodules):
                if len(nod) >= 3 and len(nod) <= 4:
                    nod_mal = []
                    for nod_anni in range(len(nod)):
                        nod_mal.append(nod[nod_anni].malignancy)
                    for nod_anni in range(len(nod)):
                        mask = nod[nod_anni].boolean_mask()
                        bbox = nod[nod_anni].bbox()
                        vol = nod[nod_anni].scan.to_volume()
                        centroid_ijk = nod[nod_anni].centroid
                        cropout_cube_size = 48
                        cropout_cube_size_half = cropout_cube_size / 2
                        cropout_bor = np.array([[0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]]])
                        for d in range(3):
                            if int(centroid_ijk[d] - cropout_cube_size_half) < 0 or int(
                                    centroid_ijk[d] + cropout_cube_size_half) > vol.shape[d]:
                                if int(centroid_ijk[d] - cropout_cube_size_half) < 0:
                                    cropout_bor[d, 1] = cropout_cube_size
                                else:
                                    cropout_bor[d, 0] = vol.shape[d] - cropout_cube_size
                            else:
                                cropout_bor[d, 0] = int(centroid_ijk[d] - cropout_cube_size_half)
                                cropout_bor[d, 1] = int(centroid_ijk[d] + cropout_cube_size_half)
                        cube_mask = np.zeros_like(vol)
                        cube_mask[bbox][mask] = 1
                        nodule_cropout_cube = vol[cropout_bor[0, 0]:cropout_bor[0, 1],
                                              cropout_bor[1, 0]:cropout_bor[1, 1],
                                              cropout_bor[2, 0]:cropout_bor[2, 1]]
                        if np.mean(nod_mal) < 3:
                            tar_labels = [1, 0]
                        else:
                            tar_labels = [0, 1]
                        nodule_cropout_cube_nii = nib.Nifti1Image(nodule_cropout_cube, affine=np.eye(4))
                        nib.save(nodule_cropout_cube_nii,
                                 "testing_data_classification/scan_" + str(current_sample_idx) + "/nodule_"+str(nod_anni)+".nii")
                        np.savetxt("testing_data_solution_classification/scan_" + str(current_sample_idx)+ "/nodule_"+str(nod_anni)+".txt", np.array(tar_labels).astype(np.int16), delimiter=',')
        else:
            os.makedirs("training_data/scan_" + str(current_sample_idx))
            scan_vol_nii = nib.Nifti1Image(scan_vol, affine=np.eye(4))
            nib.save(scan_vol_nii, "training_data/scan_" + str(current_sample_idx) + "/image_total.nii")
            scan_segm_nii = nib.Nifti1Image(scan_segm, affine=np.eye(4))
            nib.save(scan_segm_nii, "training_data/scan_" + str(current_sample_idx) + "/segmentation_total.nii")
            for i, nod in enumerate(nodules):
                if len(nod) >= 3 and len(nod) <= 4:
                    os.makedirs("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(i))
                    for nod_anni in range(len(nod)):
                        os.makedirs("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni))
                        nod_anni_mask = nod[nod_anni].boolean_mask()
                        nod_anni_bbox = nod[nod_anni].bbox()
                        nod_anni_centroid = nod[nod_anni].centroid.astype(int)
                        nod_anni_attri = [nod[nod_anni].subtlety,
                                               nod[nod_anni].internalStructure,
                                               nod[nod_anni].calcification,
                                               nod[nod_anni].sphericity,
                                               nod[nod_anni].margin,
                                               nod[nod_anni].lobulation,
                                               nod[nod_anni].spiculation,
                                               nod[nod_anni].texture]
                        nod_anni_malig = [nod[nod_anni].malignancy]

                        nod_anni_mask_nii = nib.Nifti1Image(nod_anni_mask.astype(np.int16), affine=np.eye(4))
                        nib.save(nod_anni_mask_nii, "training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni) + "/mask.nii")
                        np.savetxt("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni) + "/centroid.txt", nod_anni_centroid, delimiter=',')
                        np.savetxt("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni) + "/bbox.txt",
                                   np.array([[nod_anni_bbox[0].start, nod_anni_bbox[0].stop],
                                          [nod_anni_bbox[1].start, nod_anni_bbox[1].stop],
                                          [nod_anni_bbox[2].start, nod_anni_bbox[2].stop]]).astype(np.int16), delimiter=',')
                        np.savetxt("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni) + "/attri.txt", nod_anni_attri, delimiter=',')
                        np.savetxt("training_data/scan_" + str(current_sample_idx) + "/nodule_" + str(
                            i) + "/annotation_" + str(nod_anni) + "/mal.txt", nod_anni_malig, delimiter=',')

        current_sample_idx += 1
        running_idx += 1
    print("\n---------------------------------------------------\nSuccessfully downloaded and split data from LIDC-IDRI!\n---------------------------------------------------")
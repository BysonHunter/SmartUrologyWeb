import os
import pydicom
# from pydicom import dcmread
# from pydicom.dataset import Dataset, FileDataset
from pydicom.filereader import dcmread
from pathlib import Path
from readDicom.getSeriesNumber import getSeriesNumber


def readDICOMDIR(dicom_dir_path):
    """Reads a DICOMDIR file and returns a list of sorted DICOM datasets.

    Args:
        dicom_dir_path (str): Path to the DICOMDIR file.

    Returns:
        list: A list of sorted DICOM datasets.
    """

    image_filenames = []
    SeriesNumbers = []
    SeriesDescription = []
    CountOfImages = []
    slices = []

    # Resolve the parent directory for ReferencedFileID paths
    root_dir = Path(dicom_dir_path).resolve().parent

    # Read the DICOMDIR file
    ds = dcmread(dicom_dir_path)

    # Iterate through the PATIENT records
    for patient in ds.patient_records:

        # Find all the STUDY records for the patient
        studies = [ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"]
        for study in studies:
            # Find all the SERIES records in the study
            all_series = [ii for ii in study.children if ii.DirectoryRecordType == "SERIES"]
            for series in all_series:
                # Find all the IMAGE records in the series
                images = [ii for ii in series.children if ii.DirectoryRecordType == "IMAGE"]

                descr = getattr(series, "SeriesDescription", None)
                if descr:  # Check if SeriesDescription is not None
                    SeriesNumbers.append(series.SeriesNumber)
                    SeriesDescription.append(descr)
                    CountOfImages.append(len(images))

    # Get the selected series number (Replace selectSeries with your actual function)
    selectedSeriesNumber = getSeriesNumber(SeriesNumbers, SeriesDescription, CountOfImages)

    # Find the selected series and get image file names
    for patient in ds.patient_records:
        studies = [ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"]
        for study in studies:
            all_series = [ii for ii in study.children if
                          ii.DirectoryRecordType == "SERIES" and ii.SeriesNumber == selectedSeriesNumber]
            for series in all_series:
                image_records = series.children
                image_filenames = [os.path.join(root_dir, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

    # Read the DICOM images from the filenames
    for file in image_filenames:
        slices.append(pydicom.dcmread(file))

    # Remove datasets without SliceLocation attribute
    new_slices = [f for f in slices if hasattr(f, 'SliceLocation')]

    # Remove datasets with different pixel array shapes (Assuming they should be the same)
    sh = slices[0].pixel_array.shape
    new_slices = [s for s in new_slices if sh == s.pixel_array.shape]

    # Sort the slices based on SliceLocation and InstanceNumber
    if (new_slices[0].InstanceNumber < new_slices[-1].InstanceNumber) and (
            new_slices[0].SliceLocation < new_slices[-1].SliceLocation):
        new_slices = sorted(new_slices, key=lambda s: s.InstanceNumber)
    else:
        new_slices = sorted(new_slices, reverse=True, key=lambda s: s.SliceLocation)

    return new_slices

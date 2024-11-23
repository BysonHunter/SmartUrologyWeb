import os
import pydicom

# from pathlib import Path
from readDicom.getSeriesNumber import getSeriesNumber


def readDicomFiles(dicom_path):
    """Reads DICOM files from a directory and returns a sorted list of datasets.
    Args:
        dicom_path (str): Path to the directory containing DICOM files.
    Returns:
        list: A list of sorted DICOM datasets.
    """
    slices = []
    series_numbers = []
    series_descriptions = []
    count_of_images = []

    # Read DICOM files from the directory
    for filename in os.listdir(dicom_path):
        filepath = os.path.join(dicom_path, filename)
        if filename.lower().endswith((".dcm", ".ima")):  # Check for common DICOM extensions
            try:
                ds = pydicom.dcmread(filepath)
                slices.append(ds)
            except pydicom.errors.InvalidDicomFile:
                print(f"Skipping invalid DICOM file: {filepath}")
                break

    # Group DICOM files by series
    if slices:
        slices = sorted(slices, key=lambda sl: sl.SeriesNumber)
        current_series = slices[0].SeriesNumber
        count = 1
        for i in range(1, len(slices)):
            if slices[i].SeriesNumber == current_series:
                count += 1
            else:
                series_numbers.append(current_series)
                series_descriptions.append(slices[i - 1].SeriesDescription)  # Use the last slic's description
                count_of_images.append(count)
                current_series = slices[i].SeriesNumber
                count = 1
        series_numbers.append(current_series)
        series_descriptions.append(slices[-1].SeriesDescription)
        count_of_images.append(count)

        # Select a series (Replace selectSeries with your function)
        selected_series = getSeriesNumber(series_numbers, series_descriptions, count_of_images)

        # Filter and sort slices based on selected series
        new_slices = [slic for slic in slices if slic.SeriesNumber == selected_series]
        if new_slices:
            # Remove datasets without SliceLocation
            new_slices = [slic for slic in new_slices if hasattr(slic, 'SliceLocation')]

            # Remove datasets with different pixel array shapes (Assuming they should be the same)
            sh = new_slices[0].pixel_array.shape
            new_slices = [slic for slic in new_slices if slic.pixel_array.shape == sh]

            # Sort slices based on SliceLocation and InstanceNumber
            if (new_slices[0].InstanceNumber < new_slices[-1].InstanceNumber) and (
                    new_slices[0].SliceLocation < new_slices[-1].SliceLocation):
                new_slices = sorted(new_slices, key=lambda s1: s1.InstanceNumber)
            else:
                new_slices = sorted(new_slices, reverse=True, key=lambda s1: s1.SliceLocation)
        else:
            print(f"No slices found for selected series: {selected_series}")

        return new_slices
    else:
        print(f"No DICOM files found in directory: {dicom_path}")
        return []

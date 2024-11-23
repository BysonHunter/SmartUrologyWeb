import json


def getSeriesNumber(series_numbers, SeriesDescription, CountOfImages):
    """Presents a listbox to the user to select a series from a DICOM dataset.

    Args:
        series_numbers (list): List of series numbers from the DICOM dataset.
        SeriesDescription (list): List of series descriptions.
        CountOfImages (list): List of the number of images in each series.
    Returns:
        int: The selected series number.
    """
    if len(series_numbers) == 1:
        selected_series_number = series_numbers[0]

    else:
        # chance = []
        # for i in range(len(series_numbers)):
        #    chance.append(f'Series: {series_numbers[i]}, Descr: {SeriesDescription[i]}, SOP: {CountOfImages[i]}')
        # for ch in chance:
        #    print(ch)
        # selected_series_number = input('please, select series....')

        for i in range(len(series_numbers)):
            if SeriesDescription[i] == "Native":
                selected_series_number = series_numbers[i]

    # jsonFileName = './series.json'
    # selected_series_number = None  # Initialize selected_series_number
    # with open(jsonFileName) as json_file:
    #    data = json.load(json_file)
    #    selected_series_number = data['selectedSeries']
    # selected_series_number = series_numbers[0]

    print(f'selected series number = {selected_series_number}')
    return selected_series_number

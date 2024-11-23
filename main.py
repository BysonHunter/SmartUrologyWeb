from readDicom.readDicomUtils import *
from readDicom.getPaths import *
from detectObj.detObjects import *
from buildObj import calcObjParam
from readDicom.response import *


def main():
    InDicomDirPath = getPaths()
    dicomDirPath = "./in"
    copyInputDirToOutputDir(InDicomDirPath, dicomDirPath)
    saveImagePath = "./workdir"
    print("Start read Dicom files and create images.")
    currentPatientSaveFolder = readDicomFolder(dicomDirPath, saveImagePath)
    print(f"Completed! Images saved to out folder {currentPatientSaveFolder} !")
    print("Start detect kidneys and stones on images.")
    detect_objects(detect_folder=currentPatientSaveFolder)
    print(f"Completed! Detected info saved to {currentPatientSaveFolder}/detect folder!")
    print("Start calculating params of stones")
    inputDirOfImages = currentPatientSaveFolder
    StonesDir = calcObjParam.main(inputDirOfImages)
    print("Completed")
    outputDir = "./out"
    responseOutPaths(outputDir)
    print("Start transfer results...")
    copyInputDirToOutputDir(StonesDir, outputDir)
    print("Oк!")


if __name__ == "__main__":
    main()

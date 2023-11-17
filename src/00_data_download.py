import synapseclient, os

"""
ASSIGN USERNAME AND PASSWORD FROM THE SYNAPSE WEBSITE
"""
# Enter your credentials here
username = ""
password = ""

## Data Download

# Importing libraries
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Downloading files
# creates an instance of the Synapse class from the synapseclient module. 
syn = synapseclient.Synapse()
# logs into the Synapse platform using the provided username and password. 
syn.login(username, password)

# Downloading all files in the cart.
# Note: Files will be cleared in synapse after downloading here. Add them again if you wish to download them again.
dl_list_file_entities = syn.get_download_list(downloadLocation="../input")
print("Downloaded files")

# Unzipping the files
import zipfile
# opens the specified zip file located at ./input/brats-2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip
file = zipfile.ZipFile('../input/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip')
# extracts the contents of the archive to the directory ./Datasets/ while preserving the directory structure
file.extractall('../input/')
# close the file to free up system resources
file.close()
print("Unzipped training data")




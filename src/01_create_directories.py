import os

# Define the directory structure
structure = {
    'data': {
        'training': {
            'images': {},
            'masks': {},
            'nnUNet_raw': {
                'Dataset001_BRATS': {
                    'dataset.json': None,
                    'imagesTr': {},
                    'labelsTr': {}
                }
            },
            'nnUNet_preprocessed': {},
            'nnUNet_results': {}
        },
        'test': {
            'images': {},
            'masks': {},
            'nnUNet_raw': {
                'imagesTs': {},
                'labelsTs': {}
            }
        }
    }
}

def create_structure(base_path, structure):
    for dir_name, sub_structure in structure.items():
        path = os.path.join(base_path, dir_name)
        os.makedirs(path, exist_ok=True)
        
        if sub_structure:  # If there are sub-directories or files
            create_structure(path, sub_structure)  # Recursive call

# Create the directory structure
create_structure('..', structure)

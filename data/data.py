"""
    File to load dataset based on user control from main file
"""


def LoadData(DATASET_NAME, preprocess=None):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    if DATASET_NAME in ['ZINC', 'ZINC-full', 'AQSOL']:
        from data.molecules import MoleculeDataset
        return MoleculeDataset(DATASET_NAME, preprocess=preprocess)
    else:
        from data.TUs import TUsDataset
        return TUsDataset(DATASET_NAME, preprocess=preprocess)

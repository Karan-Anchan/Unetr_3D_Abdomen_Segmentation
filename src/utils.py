import os

def ensure_dir(directory):
    
    """
    Ensure that the directory exists.

    Args:
        directory (str): Path to the directory.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)

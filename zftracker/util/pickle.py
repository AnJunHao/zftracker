import pickle
from .tqdm import TQDM as tqdm
import os

def pickle_with_progress(lst, file_path):
    with open(file_path, 'wb') as f:
        n = len(lst)
        for item in tqdm(lst):
            pickle.dump(item, f)
    print("\nDone pickling!")

def unpickle_with_progress(file_path):
    objects = []
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f:
        while True:
            try:
                obj = pickle.load(f)
                objects.append(obj)
                # Update progress based on the read position in the file
                read_position = f.tell()
                progress = (read_position / file_size) * 100
                print(f"\rProgress: {progress:.2f}%", end='', flush=True)
            except EOFError:
                break
    print("\nDone unpickling!")
    return objects
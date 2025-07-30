import os
import urllib.request
import tarfile

destination="data/sentiment"
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
archive_path = os.path.join(destination, "aclImdb_v1.tar.gz")
extracted_path = os.path.join(destination, "aclImdb")
os.makedirs(destination, exist_ok=True)
if not os.path.exists(archive_path):
    print("Downloading IMDb dataset...")
    urllib.request.urlretrieve(url, archive_path)
else:
    print("IMDb archive already downloaded.")
    
if not os.path.exists(extracted_path):
    print("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=destination)
else:
    print("IMDb dataset already extracted.")
print("Done. Data stored in:", extracted_path)


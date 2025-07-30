import os
import urllib.request
import zipfile

destination = "data/dailydialog"
url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
zip_filename = "ijcnlp_dailydialog.zip"
zip_path = os.path.join(destination, zip_filename)

os.makedirs(destination, exist_ok=True)

if not os.path.exists(zip_path):
    print("Downloading DailyDialog dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
else:
    print("DailyDialog zip already downloaded.")

extract_path = os.path.join(destination, "raw")
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Dataset already extracted.")

print("Data available at:", extract_path)

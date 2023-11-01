import requests
import zipfile
import os
def download_glove_embeddings():
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    output_folder = "glove.6B"
    os.makedirs(output_folder, exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(output_folder, "glove.6B.zip"), 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(os.path.join(output_folder, "glove.6B.zip"), 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        print("Download and extraction completed.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
download_glove_embeddings()

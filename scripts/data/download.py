import requests
import os

# Define the path to save downloaded files
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Define the URL of the file containing all the links
url_file = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1.txt" 

# Fetch the URLs file
response = requests.get(url_file)
response.raise_for_status()

# Split URLs by lines
urls = response.text.splitlines()

# Download each file in the list
for i, url in enumerate(urls):
    file_name = os.path.join(save_dir, os.path.basename(url))
    print(f"Downloading {i+1}/{len(urls)}: {url}")

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded: {file_name}")
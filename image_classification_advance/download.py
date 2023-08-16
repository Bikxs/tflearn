import os
import zipfile
from tqdm import tqdm
import requests


def download_data():
    # Retrieve the data
    if not os.path.exists(os.path.join('data', 'tiny-imagenet-200.zip')):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        # Get the file from web
        response = requests.get(url, stream=True)

        # Get the total file size from the Content-Length header
        total_size = int(response.headers.get('content-length', 0))


        if not os.path.exists('data'):
            os.mkdir('data')

        # Write to a file
        # Create a progress bar using tqdm
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True,desc=f"downloading tiny-imagenet-200.zip...")
        filename = os.path.join('data', 'tiny-imagenet-200.zip')
        with open(filename, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
    else:
        print("The zip file already exists.")

    if not os.path.exists(os.path.join('data', 'tiny-imagenet-200')):
        with zipfile.ZipFile(os.path.join('data', 'tiny-imagenet-200.zip'), 'r') as zip_ref:
            zip_ref.extractall('data')
    else:
        print("The extracted data already exists")
if __name__ == '__main__':
    download_data()
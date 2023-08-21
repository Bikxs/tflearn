# Downloading the data
# http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz

import os
import requests
import gzip
import shutil

def download():
    # Retrieve the data
    if not os.path.exists(os.path.join('data','Video_Games_5.json.gz')):
        url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz"
        # Get the file from web
        r = requests.get(url)

        if not os.path.exists('data'):
            os.mkdir('data')
        
        # Write to a file
        with open(os.path.join('data','Video_Games_5.json.gz'), 'wb') as f:
            f.write(r.content)
    else:
        print("The tar file already exists.")
        
    if not os.path.exists(os.path.join('data', 'Video_Games_5.json')):
        with gzip.open(os.path.join('data','Video_Games_5.json.gz'), 'rb') as f_in:
            with open(os.path.join('data','Video_Games_5.json'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("The extracted data already exists")
download()
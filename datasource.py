import os
import pandas as pd
from boto.s3.connection import S3Connection
from pathlib import Path
    
def download_data(filename_remote, filename_local):
    
    if Path(filename_local).exists():
        os.remove(filename_local)
        
    conn = S3Connection('', '', anon=True)
    bucket = conn.get_bucket('c3dis-geochem-data', validate=True)
    data = bucket.get_key(filename_remote)
    data.get_contents_to_filename(filename_local)
    
    
def load_df(filename_local):
    if Path(filename_local).exists():
        return pd.read_pickle(filename_local)
    else:
        try:
            download_data(filename_local, filename_local)
        except:
            'Error: Data needs to be downloaded first.'
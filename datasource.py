import os
from boto.s3.connection import S3Connection
from pathlib import Path
    
def download_data(filename_remote, filename_local):
    
    if Path(filename_local).exists():
        os.remove(filename_local)
        
    conn = S3Connection('', '', anon=True)
    bucket = conn.get_bucket('c3dis-geochem-data', validate=True)
    data = bucket.get_key(filename_remote)
    data.get_contents_to_filename(filename_local)
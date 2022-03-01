import sys
import json
import cv2
import firebase_admin
import numpy as np
from firebase_admin import storage
from firebase_admin import credentials
class Firebase:
    def __init__(self):
        cred = credentials.Certificate(f"{sys.path[0]}//firebase_config//certificate.json")
        with open(f"{sys.path[0]}//firebase_config//config.json") as config:
            config = json.loads(config.read())
        self.firebase_service = firebase_admin.initialize_app(cred, options = config)
        self.storage = storage.bucket(app= self.firebase_service)

    
    def upload(self, filestream : bytes, filename : str, folder : str = "", public : bool = False):
        blob = self.storage.blob(f"{folder}/{filename}")
        if public:
            blob.upload_from_string(filestream)
            blob.make_public()
            return blob.public_url
        blob.upload_from_string(filestream)
        
    
    def download(self, file):        
        blob = self.storage.blob(file)
        nparr = np.frombuffer(blob.download_as_bytes(), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR )

if __name__ == '__main__':
    from image_processor import ImageProcessor
    processor = ImageProcessor()
    firebase_service = Firebase()
    with open(f'{sys.path[0]}/kelas.jpg', 'rb') as file:
        byte = file.read()
    folder = 'test'
    filename = 'oke10.jpg'
    print(firebase_service.upload(byte, filename, folder, True))
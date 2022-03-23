from mqtt import Mqtt
from firebase_service import Firebase
from processor import ImageProcessor
import json
import sys
import cv2
from datetime import datetime as dt
import uuid
NUM_OF_THREADS = 3

with open(f'{sys.path[0]}/config.json') as config_file:
    config = json.loads(config_file.read())
    config_file.close()


drive = Firebase()
messenger = Mqtt(config['mqtt_id'], address='broker.emqx.io', port=1883)
image_processor = ImageProcessor()

def detect(frame) -> int:
    total_violations = image_processor.detect(frame)
    return total_violations
    
def process_notification(message : str):
    data = json.loads(message)
    frame = drive.download(data['link']) 
    total_violations = detect(frame)
    if total_violations:
        img = cv2.imencode('.jpg', frame)[1].tobytes()
        link = drive.upload(img, f'{uuid.uuid4()}.jpg', True)
        print(link)
        notify_server(data,total_violations, link)

def notify_server(data : dict, total_violations : int, img_link : str):
    payload = {
        "class" : data['class'],
        "number_of_violations" : total_violations,
        "photo_link" : img_link
    }
    messenger.publish(json.dumps(payload), '3deef803-2854-495d-b641-677c7bda1979')


def main():
    messenger.subscribe(config['mqtt_channel'], process_notification)
    messenger.start()
    while 1:
        continue



if __name__ == '__main__':
    main()
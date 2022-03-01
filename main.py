from mqtt import Mqtt
from firebase_service import Firebase
from image_processor import ImageProcessor
from multiprocessing import Pipe, Process
import json
NUM_OF_THREADS = 3
drive = Firebase()
messenger = Mqtt("server", address='broker.emqx.io', port=1883)
image_processor= ImageProcessor()

def detect(image_link : str) -> int:
    frame = Firebase.download(image_link) 
    ImageProcessor.detect(frame)
    
def process_notification(message : str):
    data = json.loads(message)
    total_violations = detect(data['link'])
    if total_violations:
        notify_server(data,total_violations)

def notify_server(data : dict, total_violations : int):
    payload = {
        "class" : data['class'],
        "number_of_violations" : total_violations,
        "photo_link" : data['link']
    }
    messenger.publish(json.dumps(payload), '3deef803-2854-495d-b641-677c7bda1979')


def main():
    messenger.subscribe("59288f20-4e6d-4423-938a-84b9dcfc7be4", process_notification)
    messenger.start()
    while 1:
        continue



if __name__ == '__main__':
    main()
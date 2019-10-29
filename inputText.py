import cv2
import csv

vehicle = ['articulated truck','bicycle','bus','car','motorcycle','motorized vehicle','non-motorized vehicle','pedestrian','pickup truck','single unit truck','work van']

def inputText(nameVehicle,count,frame,hitungUp,hitungDown,kepadatan1,kepadatan2,font_size,boldness):
    text_length =  cv2.getTextSize(nameVehicle+ " : " +str(hitungUp),cv2.FONT_HERSHEY_DUPLEX,font_size,1)

    cv2.putText(frame, str(hitungUp)+ " : " + nameVehicle , (frame.shape[1]-int(text_length[0][0]+15), (35*count)), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), boldness, cv2.LINE_AA)
    cv2.putText(frame, nameVehicle + " : " + str(hitungDown), (20, (35*count)), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), boldness, cv2.LINE_AA)

def printHasil(hitungUp,hitungDown,frame,kepadatan1,kepadatan2):
    font_size = 0.65
    boldness = 2
    text_length =  cv2.getTextSize("kepadatan : "+kepadatan2,cv2.FONT_HERSHEY_DUPLEX,font_size,1)
    cv2.putText(frame, "Kepadatan : "+kepadatan1, (20, (35)), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), boldness, cv2.LINE_AA)
    cv2.putText(frame, "kepadatan : "+kepadatan2 , (frame.shape[1]-int(text_length[0][0]+15), (35)), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), boldness, cv2.LINE_AA)

    # cv2.putText(frame, "Durasi "+waktuu, (20, (frame.shape[0]-100)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    for i in range(0,10):
        inputText(vehicle[i],i+2, frame, hitungUp[i],hitungDown[i],kepadatan1,kepadatan2,font_size,boldness)

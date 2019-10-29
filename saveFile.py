import json
from collections import OrderedDict
from pymongo import MongoClient

# Inisiasi Server MongoDB
client = MongoClient('mongodb://192.168.20.189:27017')
db = client["object_detection"]
collection = db["traffic_analisis"]


def saveFile(namafile,tanggal,waktu,vehicleUp,vehicleDown,capacity,capacity2,waktu11):
    vehicle = ['articulated truck','bicycle','bus','car','motorcycle','motorized vehicle','non-motorized vehicle','pedestrian','pickup truck','single unit truck','work van']

    right_ =  {vehicle[i]:str( vehicleUp[i]) for i in range(0,10) if vehicleUp[i] >0}
    right = {'kepadatan':capacity}

    left_ = {vehicle[i]:str( vehicleDown[i]) for i in range(0,10) if vehicleDown[i] >0 }
    left = {'kepadatan':capacity2}

    left.update(left_)
    right.update(right_)
    
    json_dict = {
                'detection_time': str(tanggal),
                'durasi': waktu,   
                'waktu proses':waktu11/60,           
                'data':{
                        'kanan':right,
                        'kiri': left
                }
                }

    with open(namafile,"a") as write :
        json.dump(json_dict,write,indent=4)

    collection.insert(json_dict)  # Save Log to MongoDB

def exportJson(namafile,tanggal,waktu,vehicleUp,vehicleDown,capacity,capacity2,waktu11):
    vehicle = ['articulated truck','bicycle','bus','car','motorcycle','motorized vehicle','non-motorized vehicle','pedestrian','pickup truck','single unit truck','work van']

    right_ =  {vehicle[i]:str( vehicleUp[i]) for i in range(0,10)}
    right = {'kepadatan':capacity}

    left_ = {vehicle[i]:str( vehicleDown[i]) for i in range(0,10)}
    left = {'kepadatan':capacity2}

    left.update(left_)
    right.update(right_)
    
    json_dict = {
                'detection_time': str(tanggal),
                'durasi': waktu,   
                'waktu proses':waktu11/60,           
                'data':{
                        'kanan':right,
                        'kiri': left
                }
                }


    return (json_dict)
import json
import pprint

className = {
    1:'car',
    16:'bird',
    17:'cat',
    21:'cow',
    18:'dog',
    19:'horse',
    20:'sheep',
    5:'aeroplane',
    2:'bicycle',
    9:'boat',
    6:'bus',
    3:'person',
    4:'motorbike',
    7:'train',
    44:'bottle',
    62:'chair',
    67:'dining table',
    64:'potted plant',
    63:'sofa',
    72:'tvmonitor'
}

classNum = [1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]

def writeNum(Num):
    with open("pascol_voc.json","w+") as f:
        f.write(str(Num))

# with open("instances_val2014.json","r+") as f:
#     data = json.load(f)
    # annData = data["annotations"]
    # print(annData[0])
    # for x in annData[0]:
    #     if(x == "image_id"):
    #         print(type(x))
    #         print(x+ ":" + str(annData[0][x]))
    #     if (x == "image_id" or x == "bbox" or x == "category_id"):
    #         print(x + ":" + annData[0][x])
    #     if (x == "image_id" or x == "bbox" or x == "category_id"):
    #         print(x+ ":" + annData[0][x])

# with open("test.json","w") as f:
#     json.dump(annData, f, ensure_ascii=False)

inputfile = []
inner = {}
with open("MCDC_train_1000.coco.json","r+") as f:
    allData = json.load(f)
    data = allData["annotations"]
    image = allData["images"]

    print(data[0])
    print("read ready")
    print(image[0])
print("------begin-------")
print(image[0])
for i in range(7):
    print(data[i]["bbox"])
    print(data[i]["image_id"])
print("------end-------")
t=0;
for i in data:
    t=t+1
    if(i['category_id'] in classNum):
        filename = image[i["image_id"]]["file_name"].split('/')[-1]
        inner = {
            "pos":i['pos'],
            "filename": filename,
            "name": className[i["category_id"]],
            "bndbox":i["bbox"],
            "imageid": i["image_id"],
            "height": image[i["image_id"]]["height"],
            "width": image[i["image_id"]]["width"],
            "channel":3,
            "type":i['category_id'],
            "total_num":image[i["image_id"]]["anns_num"]
        }
        # if t>10:
        #     break;
        # if(i["car_rear"].has_key("rear_box")):
        #     print(i["car_rear"]["rear_box"])
        inputfile.append(inner)
    print(i["type"])
    print(i["category_id"])
# print(inputfile)
print(inputfile[0])
inputfile = json.dumps(inputfile)
writeNum(inputfile)
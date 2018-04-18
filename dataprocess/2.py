#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/19 10:05
# @Author  : He Hangjiang
# @Site    : 
# @File    : creatXML.py
# @Software: PyCharm

import xml.dom
import xml.dom.minidom
import os
# from PIL import Image
import cv2
import json

# xml文件规范定义

# _TXT_PATH= '../../SynthText-master/myresults/groundtruth'

_INDENT= ''*4
_NEW_LINE= '\n'
_FOLDER_NODE= 'MAIN'
_ROOT_NODE= 'annotation'
_DATABASE_NAME= 'MCDC'
_ANNOTATION= 'COCO2014'
_AUTHOR= 'YH'
_SEGMENTED= '0'
_DIFFICULT= '0'
_TRUNCATED= '0'
_POSE= 'Unspecified'


# _IMAGE_COPY_PATH= 'JPEGImages'
_ANNOTATION_SAVE_PATH= 'Annotations'
# _IMAGE_CHANNEL= 3

# 封装创建节点的过�?
def createElementNode(doc,tag, attr):  # 创建一个元素节�?
    element_node = doc.createElement(tag)

    # 创建一个文本节�?
    text_node = doc.createTextNode(attr)

    # 将文本节点作为元素节点的子节�?
    element_node.appendChild(text_node)

    return element_node

# 封装添加一个子节点的过�?
def createChildNode(doc,tag, attr,parent_node):

    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)


# object节点比较特殊
def createObjectNode(doc,attrs):

    object_node = doc.createElement('object')

    createChildNode(doc, 'name', attrs['name'],
                    object_node)

    createChildNode(doc, 'pose',
                    _POSE, object_node)

    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)

    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)

    bndbox_node = doc.createElement('bndbox')

    createChildNode(doc, 'xmin', str(int(attrs['bndbox'][0])),
                    bndbox_node)

    createChildNode(doc, 'ymin', str(int(attrs['bndbox'][1])),
                    bndbox_node)

    createChildNode(doc, 'xmax', str(int(attrs['bndbox'][0]+attrs['bndbox'][2])),
                    bndbox_node)

    createChildNode(doc, 'ymax', str(int(attrs['bndbox'][1]+attrs['bndbox'][3])),
                    bndbox_node)

    object_node.appendChild(bndbox_node)

    return object_node

# 将documentElement写入XML文件�?
def writeXMLFile(doc,filename):

    tmpfile =open('tmp.xml','w')

    doc.writexml(tmpfile, addindent=''*4,newl = '\n',encoding = 'utf-8')

    tmpfile.close()

    # 删除第一行默认添加的标记

    fin =open('tmp.xml')
    # print(filename)
    fout =open(filename, 'w')
    # print(os.path.dirname(fout))

    lines = fin.readlines()

    for line in lines[1:]:

        if line.split():
            fout.writelines(line)

        # new_lines = ''.join(lines[1:])

        # fout.write(new_lines)

    fin.close()

    fout.close()


if __name__ == "__main__":
    ##读取图片列表
    fileList=[];
    with open("pascol_voc.json", "r") as f:
        ann_data = json.load(f)
    for i in ann_data:
        fileList.append(i["filename"])
    current_dirpath = os.path.dirname(os.path.abspath('__file__'))

    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)

    # if not os.path.exists(_IMAGE_COPY_PATH):
    #     os.mkdir(_IMAGE_COPY_PATH)
    pos_img=0
    file_flag=True
    image_id=0
    step=0
    for i in range(len(fileList)):
        if step>1:
            step=step-1
            continue

        imageName=fileList[i]
        saveName= imageName.strip(".jpg")
        print(saveName)
        # pos = fileList[xText].rfind(".")
        # textName = fileList[xText][:pos]

        # ouput_file = open(_TXT_PATH + '/' + fileList[xText])
        # ouput_file =open(_TXT_PATH)

        # lines = ouput_file.readlines()

        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))
        # with open(xml_file_name,"w") as f:
        #     pass



        height,width,channel=ann_data[pos_img]["height"],ann_data[pos_img]["width"],ann_data[pos_img]["channel"]
        # print(height,width,channel)



        my_dom = xml.dom.getDOMImplementation()

        doc = my_dom.createDocument(None,_ROOT_NODE,None)

        # 获得根节�?
        root_node = doc.documentElement

        # folder节点

        createChildNode(doc, 'folder',_FOLDER_NODE, root_node)

        # filename节点

        createChildNode(doc, 'filename', saveName+'.jpg',root_node)

        # source节点

        source_node = doc.createElement('source')

        # source的子节点

        createChildNode(doc, 'database',_DATABASE_NAME, source_node)

        createChildNode(doc, 'annotation',_ANNOTATION, source_node)

        createChildNode(doc, 'image','flickr', source_node)

        createChildNode(doc, 'flickrid','NULL', source_node)

        root_node.appendChild(source_node)

        # owner节点

        owner_node = doc.createElement('owner')

        # owner的子节点

        createChildNode(doc, 'flickrid','NULL', owner_node)

        createChildNode(doc, 'name',_AUTHOR, owner_node)

        root_node.appendChild(owner_node)

        # size节点

        size_node = doc.createElement('size')

        createChildNode(doc, 'width',str(width), size_node)

        createChildNode(doc, 'height',str(height), size_node)

        createChildNode(doc, 'depth',str(channel), size_node)

        root_node.appendChild(size_node)

        # segmented节点

        createChildNode(doc, 'segmented',_SEGMENTED, root_node)
        step=ann_data[i]["total_num"]
        print('----step-------')
        print(step)
        for j in range(i,i+step):
            print(j)
            bndbox_node=createObjectNode(doc,ann_data[j])
            root_node.appendChild(bndbox_node)
        pos_img=pos_img+1

        # obj_node = doc.createElement('object')
        # createChildNode(doc, 'name',str(width), obj_node)
        # createChildNode(doc, 'pose',str(height), obj_node)
        # createChildNode(doc, 'truncated',str(channel), obj_node)
        # createChildNode(doc, 'difficult',str(channel), obj_node)
        # bnd_node = obj_node.createElement('bndbox')
        # createChildNode(doc, 'truncated',str(channel), obj_node)
        # createChildNode(doc, 'truncated',str(channel), obj_node)
        # createChildNode(doc, 'truncated',str(channel), obj_node)
        # createChildNode(doc, 'truncated',str(channel), obj_node)



        for ann in ann_data:
            if(saveName==ann["filename"]):
                # object节点
                object_node = createObjectNode(doc, ann)
                root_node.appendChild(object_node)

            else:
                continue

        # 构建XML文件名称

        print(xml_file_name)

        # 创建XML文件
        print('------num-----------')
        print(pos_img)
        print('-----end------------')

        # createXMLFile(attrs, width, height, xml_file_name)


        # # 写入文件
        #
        writeXMLFile(doc, xml_file_name)
import os
import random
path_img = 'train_images' #文件夹目录  
filelist=[]
file_img=[]
for root, dirs, files in os.walk(path_img):  
    # print(root) #当前目录路径  
    # print(files) #当前路径下所有非目录子文件
    for i in range(len(files)):
    	filelist.append(files[i])
for i in range(len(filelist)):
	if len(filelist[i])>24:
		print(filelist[i])
		continue
	file_img.append(filelist[i])
	
# print(len(filelist))

# random.shuffle(filelist)
path_xml='Annotations'
files_xml=os.listdir(path_xml)
file_temp=[]
# for i in files_xml:
# 	file_temp=i.spilt('.')[0]+".jpg"
# print(len(files_xml))
# ret_list = [item for item in filelist if item not in files_xml]
# print(ret_list)
# print(len(ret_list))
# print(len("2017-07-01-20-27-57.jpg"))
print(file_img)
print(len(file_img))
random.shuffle(file_img)
print(file_img[1:3])


N=800
ft = open("trainlist.txt",'w+')
lists=[line+"\n" for line in file_img[0:N]]
ft.writelines(lists)
ft.close()

fp = open("testlist.txt",'w+')
lists=[line+"\n" for line in file_img[N:1000]]
fp.writelines(lists)
fp.close()


# -*- coding: utf-8 -*-
import os
import random

def get_one_attribute(file_list,file_out, save_path, file_path, idx = 1):
    fid = open(file_list)
    fid_out = open(file_out, "w")
    total_num = int(fid.readline())
    type_line = fid.readline()
    types = type_line.split(' ')
    for t in range(len(types)):
        print "item %d is : %s"%(t,types[t])
    #print types
    for line in fid.readlines():
        #print line
        items = line.split(' ')
        new_items = []
        for item in items:
            if item != '':
                new_items.append(item)
        
        filename = new_items[0]
        Attractive = new_items[3]
        Eyeglasses = new_items[16]
        Male = new_items[21]
        Smiling = new_items[32]
        Young = new_items[40]
        attributes = (Attractive, Eyeglasses, Male, Smiling, Young)
        if os.path.isfile(file_path+filename):
            if attributes[idx] =="1":
                tag = "1"
            else:
                tag = "0"
            fid_out.write(file_path+filename+"\t"+tag+"\n")
    
    fid_out.close()

def get_multi_attribute(file_list,file_out, save_path, file_path, idxs = []):
    fid = open(file_list)
    fid_out = open(file_out, "w")
    total_num = int(fid.readline())
    type_line = fid.readline()
    types = type_line.split(' ')
    for t in range(len(types)):
        print "item %d is : %s"%(t,types[t])
    #print types
    for line in fid.readlines():
        #print line
        items = line.split(' ')
        new_items = []
        for item in items:
            if item != '':
                new_items.append(item)
        
        filename = new_items[0]
        Attractive = new_items[3]
        Eyeglasses = new_items[16]
        Male = new_items[21]
        Smiling = new_items[32]
        Young = new_items[40]
        attributes = (Attractive, Eyeglasses, Male, Smiling, Young)
        if os.path.isfile(file_path+filename):
            fid_out.write(file_path+filename)
            for idx in idxs:
                if attributes[idx] =="1":
                    tag = "1"
                else:
                    tag = "0"
                fid_out.write("\t"+tag)
            fid_out.write("\n")
    fid_out.close()
    
    
def div_train_val(file_list, file_list_train, file_list_val):
    val_rate = 10
    i  = 0
    fid_train = open(file_list_train, 'w')
    fid_val = open(file_list_val,'w')
    for line in open(file_list).readlines():
        if i%val_rate == 0:
            fid_val.write(line)
        else:
            fid_train.write(line)
        i+=1
    fid_train.close()
    fid_val.close()

def get_data_static(file_list):
	'''
	@brief: 统计每个类别的分布数目
	
	file_list
		filename label
	''' 
	label_static = {}
	with open(file_list) as fid:
		for line in fid.readlines():
			#print line
			items = line.split('\t')
			label = int(items[1])
			if label in label_static:
				label_static[label]+=1
			else:
				label_static[label]=1

	print max(label_static.values())
	for key,value in label_static.items():
		print key,value
		
	return label_static
        
if __name__ == "__main__":
    file_list = "list_attr_celeba.txt"
    file_path = '/media/crw/MyBook/MyDataset/dlib_detect/CelebrayA/'
    save_path = "./"
    #attributes = (Attractive, Eyeglasses, Male, Smiling, Young)
    #file_out = 'smile.list'
    #file_out = 'gender.list'
    #file_out = 'attractive.list'
    #get_one_attribute(file_list,file_out, save_path, file_path, idx = 3)
    #get_data_static(file_out)
    #file_list_train = 'smile_train.list'
    #file_list_val = 'smile_val.list'
    #div_train_val(file_out, file_list_train, file_list_val)
    
#    file_out = 'attractive_smile.list'
#    get_multi_attribute(file_list,file_out, save_path, file_path, idxs = [0,3])
#    file_list_train = 'attractive_smile_train.list'
#    file_list_val = 'attractive_smile_val.list'
#    div_train_val(file_out, file_list_train, file_list_val)
    
    file_out = 'attractive_gender_smile.list'
    get_multi_attribute(file_list,file_out, save_path, file_path, idxs = [0,2,3])
    file_list_train = 'attractive_gender_smile_train.list'
    file_list_val = 'attractive_gender_smile_val.list'
    div_train_val(file_out, file_list_train, file_list_val)
    

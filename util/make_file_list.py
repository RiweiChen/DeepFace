# -*- coding: utf-8 -*-
"""
@brief 生成数据的部分（从IMDB中适配）
@update 2016.04.21: 原始的label数据有出生年龄标注为0的情况，更新去除这一部分的数据。 
        2016.04.26: 原先的训练数据和验证数据划分存在着人物之前的重叠，更正为根据
                    不同的人进行划分训练和测试数据。
        2016.05.02: 适配OnlineData。
        2016.05.04: 去除掉57和77标注的人脸数据。
"""
import os

def make_dict_from_file(file_list):
    """
    @brief 从文件中获取字典，
    @return 字典：key：id+file_name, value: [child,gender]
    """
    ret_dict = dict()
    with open(file_list) as fid:
        fid.readline()
        for line in fid:
            items = line.split("\t")
            id_ = items[0]
            name_ = items[1]
            gender_ = int(items[4])-1
            child_ = int(items[5])-1
            if gender_ == 1:
                gender_ = 0
            elif gender_ == 0:
                gender_ = 1
            ret_dict[id_+"/"+name_+".jpg"] = [child_, gender_]
    return ret_dict


def make_train_file_list_gender_no_child(root_path, input_file_list):
    """
    @brief 生成IMDB的数据标签列表文件
    """
    fid_train = open("gender_nochild_train_lmdb.list","w")
    fid_val = open("gender_nochild_val_lmdb.list","w")
    train_rate = 10
    dict_label = make_dict_from_file(input_file_list)
    id_list = os.listdir(root_path)
    count_id = 0
    for id_ in id_list:
        sub_path = os.path.join(root_path,id_)
        count_id +=1
        file_list = os.listdir(sub_path)
        for file_ in file_list:
            if (id_+"/"+file_) in dict_label:
                value = dict_label[id_ + "/"+file_]
                child, gender = str(value[0]),str(value[1])
                if gender == "98":# mean unknown
                    break
                if child == "1":
                    break
                if count_id %train_rate == 0:
                    fid_val.write(id_+"/"+file_+"\t"+gender+"\n")
                else:
                    fid_train.write(id_+"/"+file_+"\t"+gender+"\n")
    fid_val.close()
    fid_train.close()

def make_train_file_list_gender(root_path, input_file_list):
    """
    @brief 生成IMDB的数据标签列表文件
    """
    fid_train = open("gender_train_lmdb.list","w")
    fid_val = open("gender_val_lmdb.list","w")
    train_rate = 10
    dict_label = make_dict_from_file(input_file_list)
    id_list = os.listdir(root_path)
    count_id = 0
    for id_ in id_list:
        sub_path = os.path.join(root_path,id_)
        count_id +=1
        file_list = os.listdir(sub_path)
        for file_ in file_list:
            if (id_+"/"+file_) in dict_label:
                value = dict_label[id_ + "/"+file_]
                child, gender = str(value[0]),str(value[1])
                if gender == "98":# mean unknown
                    break
                #if child == "98":
                #    break
                if count_id %train_rate == 0:
                    fid_val.write(id_+"/"+file_+"\t"+gender+"\n")
                else:
                    fid_train.write(id_+"/"+file_+"\t"+gender+"\n")
    fid_val.close()
    fid_train.close()

def make_train_file_list_child(root_path, input_file_list):
    """
    @brief 生成IMDB的数据标签列表文件
    """
    fid_train = open("child_train_lmdb.list","w")
    fid_val = open("child_val_lmdb.list","w")
    train_rate = 10 # 表示95%的人为训练数据 
    dict_label = make_dict_from_file(input_file_list)
    id_list = os.listdir(root_path)
    count_id = 0
    for id_ in id_list:
        sub_path = os.path.join(root_path,id_)
        count_id+=1
        file_list = os.listdir(sub_path)
        for file_ in file_list:
            if (id_+"/"+file_) in dict_label:
                value = dict_label[id_ + "/"+file_]
                child, gender = str(value[0]),str(value[1])
                if child == "98":
                    break
                #if gender =="98":
                #    break
                if count_id % train_rate == 0:
                    fid_val.write(id_+"/"+file_+"\t"+child+"\n")
                else:
                    fid_train.write(id_+"/"+file_+"\t"+child+"\n")

    fid_train.close()
    fid_val.close()

def make_file_list_from_top(root_path, input_file_list, cluster_file):

    fid_cluster = open(cluster_file)
    clusters = set()
    for line in fid_cluster:
        clusters.add(line.strip())

    fid_train = open("gender_train_top3.list","w")
    fid_val = open("gender_val_top3.list","w")

    fid_cluster_label_val = open("cluster_val_label.list","w")
    fid_cluster_label_train = open("cluster_train_label.list","w")
    train_rate = 10
    dict_label = make_dict_from_file(input_file_list)
    id_list = os.listdir(root_path)
    count_id = 0
    for id_ in id_list:
        if id_  not in clusters:
            continue
        sub_path = os.path.join(root_path,id_)
        count_id +=1
        file_list = os.listdir(sub_path)
        tag_add_one = False
        for file_ in file_list:
            if (id_+"/"+file_) in dict_label:
                value = dict_label[id_ + "/"+file_]
                child, gender = str(value[0]),str(value[1])
                if gender == "98":# mean unknown
                    break
                #if child == "98":
                #    break
                if count_id %train_rate == 0:
                    if tag_add_one == False:
                        fid_cluster_label_val.write(id_+"\t"+str(gender)+"\t"+str(child)+"\n")
                        tag_add_one = True
                    fid_val.write(id_+"/"+file_+"\t"+gender+"\n")
                else:
                    if tag_add_one == False:
                        fid_cluster_label_train.write(id_+"\t"+str(gender)+"\t"+str(child)+"\n")
                        tag_add_one = True
                    fid_train.write(id_+"/"+file_+"\t"+gender+"\n")
    fid_val.close()
    fid_train.close()
if __name__ == "__main__":
    input_file_list = "gender_child.list"
    root_path = "margin_0.5_256/"
    cluster_file = "cluster_top3_add.list"
    make_file_list_from_top(root_path, input_file_list, cluster_file)
    #make_train_file_list_gender_no_child(root_path, input_file_list)


import json

datasets={
    'UAV123':[ 'bike1','group1_1', 'group1_2', 'group1_3', 'group1_4', \
               'group2_1', 'group2_2', 'group2_3', 'group3_1', 'group3_2', \
               'group3_3', 'group3_4','person4_1', 'person4_2','person9',\
               'person19_1', 'person19_2','person11','person18','person20'],
    'UAV20L':[],
    'OTB100':['Girl2','Human3','Human4-2','Jogging-1','Walking','Walking2','Woman']
}

def extract_json():
    setpath = '/home/rislab/Workspace/pysot/testing_dataset/Following/'
    uav123_setpath = '/home/rislab/Workspace/pysot/testing_dataset/UAV123/UAV123.json'
    uav20l_setpath = '/home/rislab/Workspace/pysot/testing_dataset/UAV123/UAV20L.json'
    otb_setpath = '/home/rislab/Workspace/pysot/testing_dataset/OTB100/OTB100.json'

    with open(uav123_setpath) as f:
        uav123_json = json.load(f)

    with open(uav20l_setpath) as f:
        uav20l_json = json.load(f)

    with open(otb_setpath) as f:
        otb_json = json.load(f)

    json_info = {}

    for set in datasets['UAV123']:
        json_info[set] = uav123_json[set]

    for set in datasets['UAV20L']:
        json_info[set] = uav20l_json[set]

    for set in datasets['OTB100']:
        json_info[set] = otb_json[set]

    with open(setpath+'Following.json','w') as f:
        json.dump(json_info, f)

if __name__ == '__main__':
    extract_json()
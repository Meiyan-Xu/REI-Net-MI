import sys
import yaml
import os
import pickle
sys.path.append('../..')
from MI.Transfer.Tool.load_data_2 import get_trans_data_OpenBMI,get_trans_data_bci_1,get_trans_data_Kaya_3_ratio,get_trans_data_bci_2,get_trans_data_bci_3_ratio,get_trans_data_bci


if __name__ == '__main__':

    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None

    # import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    channDic = getChannels()

    if yml["Meta"]['Datasets'] == 'BCIC2a':
        sub_all = list(range(1, 10))
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
    elif yml['Meta']['Datasets'] == 'OpenBMI':
        sub_all = list(range(1, 55))
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
    elif yml['Meta']['Datasets'] == 'Kaya2018':
        sub_all = list(range(1, 12))
        eventTypeDic = {
            0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }

    for sub in sub_all:
        model_id = []
        ratio = yml['Trans']['ratio']
        # load data 数据处理切记先交叉再分割
        if yml["Meta"]['Datasets'] == 'BCIC2a':
            savemat_file_name = yml['Trans']['matPath']
            if not os.path.exists(savemat_file_name):
                os.makedirs(savemat_file_name)
            if yml['Trans']['DataMethod'] ==1:
                print('-------Start method 1 generate-------')
                all_data = get_trans_data_bci_1(yml, sub,eventTypeDic)
            elif yml['Trans']['DataMethod'] ==2:
                print('-------Start method 2 generate-------')
                all_data = get_trans_data_bci_2(yml, sub, eventTypeDic)
            elif yml['Trans']['DataMethod'] ==3:
                print('-------Start method 3 generate-------')
                all_data = get_trans_data_bci_3_ratio(yml, sub, eventTypeDic,ratio)
            elif yml['Trans']['DataMethod'] ==4:
                print('-------Start method 4 generate-------')
                all_data = get_trans_data_bci(yml, sub, eventTypeDic)
            with open(savemat_file_name +'/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100,sub), 'wb') as f:
                pickle.dump(all_data, f)
        elif yml["Meta"]['Datasets'] == 'OpenBMI':
            savemat_file_name = yml['Trans']['matPath']
            if not os.path.exists(savemat_file_name):
                os.makedirs(savemat_file_name)
            all_data = get_trans_data_OpenBMI(yml, channDic, sub,eventTypeDic,ratio)
            with open(savemat_file_name +'/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100,sub), 'wb') as f:
                pickle.dump(all_data, f)
        elif yml['Meta']['Datasets'] == 'Kaya2018':
            savemat_file_name = yml['Trans']['matPath']
            if not os.path.exists(savemat_file_name):
                os.makedirs(savemat_file_name)
            all_data = get_trans_data_Kaya_3_ratio(yml,channDic,sub,eventTypeDic,ratio)
            with open(savemat_file_name +'/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100,sub), 'wb') as f:
                pickle.dump(all_data, f)

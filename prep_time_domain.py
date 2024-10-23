import MI
from MNUMI.preprocess.BCI2008 import time_domain
import sys
import yaml
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from MI.loadData import getEvents
from meya.loadData_YML import getData_chanSeq as getData
from meya.fileAction import saveFile

#


if __name__ =='__main__':
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1],encoding='UTF-8'), Loader=yaml.FullLoader)
    for pkg,function in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(function)
        exec (stri)

    channDic = getChannels()
    pick_smp_freq = yml['Meta']['downSampling']
    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    num_class = yml['Meta']['ClassNum']
    Channel_format = yml['Meta']['Channel']
    n_subjs= yml['Meta']['subsample']
    k_folds = yml['ML']['folderNum']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    Datasets = yml['Meta']['Datasets']
    step= yml['Meta']['step']
    interp = yml['Meta']['interpType']
    eventTypeDic = getEvents(Datasets,num_class)

    #如果使用matlab进行了resit插值，则需要填写该路径
    setPath=yml['Meta']['setPath']
    RawDataPath = BasePath +'/raw'
    ProcessDataSavePath = BasePath + "/ProcessData/{}Hz_{}chan_{}_{}Hz_notch_{}_filter".format(pick_smp_freq,Channel_format,minFreq,maxFreq,interp)
    if not os.path.exists(ProcessDataSavePath):
        os.makedirs(ProcessDataSavePath)
    save_path = "/data/Running/BCI_IV_2a"+ '/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_{}_padding_test'.format(TrainType, num_class, pick_smp_freq,Channel_format,minFreq,maxFreq,step,interp)
    # save_path = BasePath + '/Traindata/FBCNet/Strokesh_re_multi_test_fix_w2.5_step0.05_ali_intra_test_split_EA'
    # save_path = BasePath + '/Traindata/EEGconformer/OpenBMI_pretrain_fix_w2.5_filter_right'
    # save_path = '/home/xumeiyan/Public/Code/MI/ComperCode/FBCNet_valloss/data/Strokesh_inter_re_test_fix_w2.5_step0.05_for_leftsub/originalData'
    # save_path = '/data0/meya/code/JJ/ComperCode/FBCNet_valloss/data/Strokesh_inter_re_test_fix_w2.5_step0.05_swap/originalData'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_count = os.listdir(ProcessDataSavePath)

    saveFile(save_path, sys.argv[1], cover=True)

    if Datasets =='OpenBMI':
        if len(file_count) != 434:
            '生成ProcessData'
            MI.preprocess.OpenBMI.time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, n_subjs,yml)
            if os.path.exists(setPath) and not os.path.exists(ProcessDataSavePath):
                MI.preprocess.OpenBMI.time_domain.Subject_session_DataGenerate_set(setPath, ProcessDataSavePath, n_subjs,yml)
            saveFile(ProcessDataSavePath, sys.argv[1], cover=True)

        if TrainType == 'dependent':
            '对象内数据生成'
            MI.preprocess.OpenBMI.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                                          k_folds=k_folds,
                                                                          chanDic=channDic,
                                                                          ProDataPath=ProcessDataSavePath,
                                                                          save_path=save_path,
                                                                          num_class=num_class,
                                                                          n_subjs=n_subjs)
        elif TrainType == 'independent':
            '跨对象的数据生成'
            MI.preprocess.OpenBMI.time_domain.subject_independent_setting_spilt(yml=yml,
                                                                 chanDic = channDic,
                                                                 ProDataPath=ProcessDataSavePath,
                                                                 save_path=save_path,
                                                                 num_class=num_class,
                                                                 n_subjs=n_subjs)
        elif TrainType == 'Transfer':
            MI.preprocess.OpenBMI.time_domain.subject_transfer_setting_spilt(yml=yml,
                                                                                chanDic=channDic,
                                                                                ProDataPath=ProcessDataSavePath,
                                                                                save_path=save_path,
                                                                                num_class=num_class,
                                                                                n_subjs=n_subjs)
    if Datasets =='BCIC2a':
        if len(file_count) != 36:
            'BCI2008数据生成方法'
            time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, num_class, n_subjs,yml)
            saveFile(ProcessDataSavePath, sys.argv[1], cover=True)
            if os.path.exists(setPath) and not os.path.exists(ProcessDataSavePath):
                #读取set数据
                time_domain.Subject_session_DataGenerate_Set(setPath,ProcessDataSavePath,num_class,n_subjs,yml)

        if TrainType == 'dependent':
            '对象内数据生成'

            time_domain.subject_dependent_setting_spilt(yml=yml,
                                                    k_folds=k_folds,
                                                    ProDataPath=ProcessDataSavePath,
                                                    save_path=save_path,
                                                    num_class=num_class,
                                                    n_subjs=n_subjs)

        elif TrainType =='independent':
            time_domain.subject_independent_setting_spilt(yml=yml,
                                                        ProDataPath=ProcessDataSavePath,
                                                        save_path=save_path,
                                                        num_class=num_class,
                                                        n_subjs=n_subjs)
    if Datasets == "StrokeData":
        if len(file_count) !=100:
            '生成ProcessData'
            MI.preprocess.StrokeData.time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, n_subjs,
                                                                              yml)

        if TrainType == 'dependent':
            MI.preprocess.StrokeData.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                                                 k_folds=k_folds,
                                                                                   chanDic=channDic,
                                                                                   ProDataPath=ProcessDataSavePath,
                                                                                   save_path=save_path,
                                                                                   num_class=num_class,
                                                                                   n_subjs=n_subjs)


        if TrainType == 'independent':
            MI.preprocess.StrokeData.time_domain.subject_independent_setting_spilt(yml=yml,
                                                                                chanDic=channDic,
                                                                                ProDataPath=ProcessDataSavePath,
                                                                                save_path=save_path,
                                                                                num_class=num_class,
                                                                                n_subjs=n_subjs)
        if TrainType == 'Transfer':
            MI.preprocess.StrokeData.time_domain.subject_transfer_setting_spilt(yml=yml,
                                                                                   chanDic=channDic,
                                                                                   ProDataPath=ProcessDataSavePath,
                                                                                   save_path=save_path,
                                                                                   n_subjs=n_subjs)

    if Datasets == "StrokeSh":
        if len(file_count) !=26:
            '生成ProcessData'
            MI.preprocess.Stroke_SH.time_domain.Subject_session_DataGenerate(RawDataPath, ProcessDataSavePath, n_subjs,
                                                                              yml)

        if TrainType == 'dependent':
            MI.preprocess.Stroke_SH.time_domain.subject_dependent_setting_spilt(yml=yml,
                                                                                 k_folds=k_folds,
                                                                                   chanDic=channDic,
                                                                                   ProDataPath=ProcessDataSavePath,
                                                                                   save_path=save_path,
                                                                                   num_class=num_class,
                                                                                   n_subjs=n_subjs)

        if TrainType == 'independent':
            MI.preprocess.Stroke_SH.time_domain.subject_independent_setting_spilt(yml=yml,
                                                                                chanDic=channDic,
                                                                                ProDataPath=ProcessDataSavePath,
                                                                                save_path=save_path,
                                                                                num_class=num_class,
                                                                                n_subjs=n_subjs)
        if TrainType == 'Transfer':
            MI.preprocess.StrokeData.time_domain.subject_transfer_setting_spilt(yml=yml,
                                                                                   chanDic=channDic,
                                                                                   ProDataPath=ProcessDataSavePath,
                                                                                   save_path=save_path,
                                                                                   n_subjs=n_subjs)




    if Datasets == "Kaya2018":
        if len(file_count) !=44:
            '生成ProcessData'
            # dataEGenerate:保存set文件和event文件 方便后续上采样或者下采样
            MI.preprocess.Kaya_2018.time_domain.Subject_session_DataGenerate(RawDataPath,ProcessDataSavePath,n_subjs,yml)
        #     saveFile(ProcessDataSavePath, sys.argv[1], cover=True)
            # 生成经过滤波，插值，重采样之后的数据
            if os.path.exists(setPath) and not os.path.exists(ProcessDataSavePath):
                MI.preprocess.Kaya_2018.time_domain.Subject_session_DataGenerate_set(setPath, ProcessDataSavePath, n_subjs, yml)
        if TrainType == 'dependent':
            MI.preprocess.Kaya_2018.time_domain.subject_dependent_setting_spilt(yml,k_folds=k_folds,
                                                                                ProDataPath=ProcessDataSavePath,
                                                                                save_path=save_path,
                                                                                chanDic=channDic,
                                                                                num_class=num_class,
                                                                                n_subjs=n_subjs)
        elif TrainType =='independent':
                MI.preprocess.Kaya_2018.time_domain.subject_independent_setting_spilt(yml,chanDic=channDic,
                                                                                ProDataPath=ProcessDataSavePath,
                                                                                save_path=save_path,
                                                                                num_class=num_class,
                                                                                n_subjs=n_subjs)

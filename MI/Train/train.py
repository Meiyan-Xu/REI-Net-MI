import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import pickle
import sys
sys.path.append('../..')
import yaml
import logging
import numpy as np
import multiprocessing
from meya.fileAction import joinPath, saveFile, saveRawDataToFile,getModelDic
from meya.appLog import createLog
from meya.MLTrain import trainCount
from MI.EEGML import ML_Model
from meya.loadModel import GetModelPath
from shutil import copy2
from os import makedirs

def Train(yml, train_subs, loadData, channDic,loadDataPath, eventTypeDic, getData, folderIndex,modelPath):
    print('————————————————————————sub————————————————————————————————:', os.getpid())
    # 获取每折的数据
    print("Strat Subject{}_fold{} data load".format(train_subs, folderIndex))
    print('Loda Data from:{}'.format(loadDataPath))
    num_class = 2
    TrainData, Tralabel = load_train_Data(train_subs, loadDataPath, folderIndex)
    ValidData, Vallabel = load_val_data(train_subs, loadDataPath, folderIndex)
    tramax,tramin = np.max(TrainData),np.min(TrainData)
    valmax, valmin = np.max(ValidData), np.min(ValidData)
    # Tralabel = Tralabel+1
    # Vallabel = Vallabel+1
    print(
        "Check dimension of training data {},training label {},val data {},val label{} ".format(
            TrainData.shape,Tralabel.shape, ValidData.shape,Vallabel.shape))
    #learnData learnLabel中存放的是所有训练对象的数据
    # learnData, learnLabel = getTrainData_MulSub(yml, train_subs, loadData, channDic, eventTypeDic, getData)
    if (TrainData is not None and len(Tralabel) > 0 and ValidData is not None and len(Vallabel)>0):
        subFolderPath = joinPath(yml["Meta"]['basicFolder'], yml["Meta"]['folderName'])
        validData = (ValidData,Vallabel)
        trainCount(subFolderPath, folderIndex, yml, TrainData, Tralabel,train_subs,validData,ML_Model=ML_Model,
                   eventTypeDic=eventTypeDic, modelPath=modelPath)
    print("Compelete folder %d Train!" % folderIndex)

def load_train_Data(sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_train_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x,allow_pickle=True)
        X_mean = np.mean(np.absolute(X))
        # X = np.swapaxes(X,1,2)
        print('Raw meam:{}'.format(X_mean))
        y = np.load(file_y, allow_pickle=True)
        # y=y+1
        # if yml['ML']['loss']=='binary_crossentropy':
        #     print(y[0])
        #     y = K.argmax(y,axis=-1)
        #     y = y.numpy().tolist()
        #     y = np.array(y, dtype='int32')
        print('Train_X shape{},Train_y shape{}'.format(X.shape,y.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X,y
def load_val_data(sub,load_path,folderIndex):
    try:
        file_x = load_path + '/X_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        file_y = load_path + '/y_val_S{:03d}_fold{:03d}.npy'.format(sub, folderIndex)
        X = np.load(file_x, allow_pickle=True)
        X_mean = np.mean(np.absolute(X))
        # X = np.swapaxes(X,1,2)
        print('Raw meam:{}'.format(X_mean))
        y = np.load(file_y, allow_pickle=True)
        # y=y+1
        # if yml['ML']['loss']=='binary_crossentropy':
        #     y = K.argmax(y,axis=-1)
        #     y = y.numpy().tolist()
        #     y = np.array(y, dtype='int32')
        print('Val_X shape{},Val_y shape{}'.format(X.shape, y.shape))
    except:
        raise Exception(
            'Path Error: file does not exist, please check this path {}, and {}'.format(file_x, file_y))
    return X, y



def read_pickle(work_path):
    with open(work_path,'rb') as f:
        try:
            history_data = pickle.load(f)
        except Exception as err:
            print(err)
            log.error(err)
            raise err
    folde_val_loss = min(history_data['val_loss'])
    return folde_val_loss


def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    saveRawDataToFile(save_path + '/X_train_' + NAME + '.npy', X_train)
    saveRawDataToFile(save_path + '/X_val_' + NAME + '.npy', X_val)
    saveRawDataToFile(save_path + '/X_test_' + NAME + '.npy', X_test)
    saveRawDataToFile(save_path + '/y_train_' + NAME + '.npy', y_train)
    saveRawDataToFile(save_path + '/y_val_' + NAME + '.npy', y_val)
    saveRawDataToFile(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')


if __name__ == '__main__':
    #多进程启动方式
    multiprocessing.set_start_method('forkserver', force=True)
    print('————————————————————————Main1————————————————————————————————:', os.getpid())

    # 步长为100
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1],encoding='UTF-8'), Loader=yaml.FullLoader)
    # yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None
    # imports

    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    BasePath = yml['Meta']['initDataFolder']
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']

    # saveFolder=/data0/meya/MI/Intersub_eprime/1S_FB1-100_S100_chanSeq_20220405_FNoE
    saveFolder = joinPath(basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
        if 'curFolderIndex' in yml['ML']:
            yml['ML']['curFolderIndex'] = 0
        else:
            yml['ML'].setdefault('curFolderIndex', 0)
        with open(sys.argv[1], "w") as f:
            yaml.dump(yml, f, sort_keys=False)

    mainLog = createLog(basicFolder, folderName)
    channDic = getChannels()
    # Step-2: training para

    doshuffle = False
    if "doShuffle" in yml['Meta']:
        # true
        doshuffle = yml['Meta']['doShuffle']
    filterBank = None
    if 'FilterBank' in yml['Meta']:
        filterBank = yml['Meta']['FilterBank']
    #使用MIN2Net生成数据
    Dataset = yml['Meta']['Datasets']  #Dataset name:ex[BCIC2a/OpenBMI]
    Datatype = 'time_domain'
    num_class = yml['Meta']['ClassNum']    # number of classes:ex[2,3,4]

    if Dataset =='OpenBMI':
        num_subject = 54
    #这部分只是为了计算权重
        if num_class ==2:
            eventTypeDic = {
                0:{'Name':'right','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                }
        if num_class == 3:
            eventTypeDic = {
                0:{'Name':'rest','StaTime':0,'TimeSpan':4000,'IsExtend':False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2:{'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
                }
    elif Dataset =="BCIC2a":
        num_subject = 9
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
        if num_class == 4:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'foot', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                3: {'Name': 'tongue', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                }
    elif Dataset =="StrokeData":
        num_subject = 50
        if num_class ==2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset =='StrokeSh':
        num_subject = 15
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'fix', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset =="Kaya2018":
        num_subject = 11
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset == "Physionet":
        num_subject = 109
        eventTypeDic = {
            0: {'Name': 'base', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            3: {'Name': 'LR', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            4: {'Name': 'F', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
        }
    #min2net数据路径

    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    pick_smp_freq = yml['Meta']['downSampling']
    Channel_format = yml['Meta']['Channel']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    step = yml['Meta']['step']
    interp = yml['Meta']['interpType']
    Min2DataPath = BasePath + '/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_{}_padding'.format(TrainType, num_class,
                                                                                  pick_smp_freq, Channel_format,
                                                                                  minFreq, maxFreq, step,interp)


    # Min2DataPath = "/data0/meya/Data/MI/datasets/Physionet/EEGML"
    #                "/Traindata/dependent/2_class/400Hz_49chan_0.5_100Hz_0.06"
    Subnum = []
    step = yml['Meta']['step']
    segmentName = yml['Meta']['segmentName']
    # Should it run in process?
    runInProcess = True
    if 'runInProcess' in yml['ML']:
        runInProcess = yml['ML']['runInProcess']
    log = logging.getLogger()
    global folderIndex
    folderNum = 3
    if 'folderNum' in yml['ML']:
        folderNum = yml['ML']['folderNum']
    folderIndex = 0

    saveFile(saveFolder, sys.argv[1], cover=True)
    # for person in range(1,num_subject+1):
    #选择被试 模型优化只选择10个被试进行训练 跨对象训练和BCI2008训练使用54个被试
    if yml['Meta']['SelectSub']:
        # select_sub = [45]
        # select_sub = list(range(1,21))
        # select_sub = list(range(1, 55))
        select_sub = list(range(3, 55))
    else:
        # select_sub = [ 4, 5, 6,  8, 9,  12, 13, 14, 15, 16, 17, 18, 19,
        #               20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        #               44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        # select_sub = [4, 5, 6, 8, 9, 12, 13, 14, 15, 16]
        select_sub = list(range(1, num_subject+1))
    for person in select_sub:
    # for person in range(1,num_subject+1):
        foldval_loss = []
        for i in range(0,5):
            try:
                folderIndex = i + 1
                #创建folder文件夹
                curModelFolder = '%s/%s/folder_%d' % (saveFolder, 'sub{0:02d}'.format(person), folderIndex)
                if not os.path.exists(curModelFolder):
                    os.makedirs(curModelFolder)
                doTrain = True
                modelPath = None
                h5TimeList, modelPath = getModelDic(curModelFolder)
                if len(h5TimeList) > 0:
                    if 'loadModel' not in yml['ML'] or not yml['ML']['loadModel']:
                        doTrain = False
                print(doTrain)
                if doTrain:
                    print('folder Index: %d' % folderIndex)
                    if runInProcess:
                        p = multiprocessing.Process(target=Train, args=(
                        yml, person, loadData, channDic,Min2DataPath, eventTypeDic, getData, folderIndex, modelPath))
                        p.start()
                        p.join()
                    else:
                        #getData_chanSeq as getData
                         Train(yml, person, loadData, channDic, Min2DataPath,eventTypeDic, getData, folderIndex, modelPath)
            except Exception as err:
                print(err)
                log.error(err)
                raise err
            # 读取PKL文件
            subFolderPath = joinPath(yml["Meta"]['basicFolder'], yml["Meta"]['folderName'])
            save_dir = "%s/%s/folder_%d" % (subFolderPath, 'sub{0:02d}'.format(person),
                                            folderIndex)
            Pklpath, H5path,_ = GetModelPath(save_dir)
            print(save_dir, Pklpath)
            history_file = joinPath(save_dir, Pklpath)
            foldval_loss.append(read_pickle(history_file))
        # 找出五折中最小的valloss的折数
        fold_index = foldval_loss.index(min(foldval_loss))
        minlossfold = fold_index + 1
        best_dir = joinPath(subFolderPath, 'sub{0:02d}/best'.format(person))
        # if not os.path.exists(best_dir):
        #     os.makedirs(best_dir)
        makedirs(best_dir, exist_ok=True)
        with open(joinPath(best_dir, "fold_bestcv.txt"), 'a') as f:
            f.write("sub{}, fold{}\n".format(person, minlossfold))
        model_dir = "%s/%s/folder_%d" % (subFolderPath, 'sub{0:02d}'.format(person),
                                         minlossfold)
        BestPkl, BestH5,_ = GetModelPath(model_dir)
        # copy2(joinPath(model_dir, 'modle-e{}-f{}.h5'.format(epochs, minlossfold)),
        #         joinPath(best_dir, 'model-sub{0:02}.h5'.format(sub)))
        copy2(joinPath(model_dir, BestH5),
              joinPath(best_dir, BestH5))
        copy2(joinPath(model_dir, BestPkl),
              joinPath(best_dir, BestPkl))


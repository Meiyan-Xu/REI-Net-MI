import os
import sys
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
sys.path.append('../..')
import yaml
import logging
from MI.EEGML import ML_Model
from meya.appLog import createLog
# from meya.MLTestIntraSub import test_Inter
import datetime as dt
import glob
from meya.fileAction import joinPath
from meya.MLTest import test2
import tensorflow as tf
import multiprocessing

def getTestData(yml,test_sub,index):
    edf_list = glob.glob(joinPath(yml['Meta']['initDataFolder'], 'chb{0:02d}'.format(sub), '*.edf'))
    edf_list.sort()
    edfFile=os.path.basename(edf_list[index])
    edfData, edfLabel = loadData(yml, test_sub, channDic, eventTypeDic, getData, isTrain=False,CurEdf=edfFile)
    if 'addFakeClass' in yml['ML'] and yml['ML']['addFakeClass']:
        edfLabel=[l+1 for l in edfLabel]
    if edfData is not None and len(edfLabel) > 0:
        # seizureIndex = np.where(np.array(edfLabel, dtype=np.int) == [0, 1])[0]
        # testData = [edfData[dIndex] for dIndex in seizureIndex]
        # testLabel = [edfLabel[dIndex] for dIndex in seizureIndex]
        if 'testTitle' in yml['ML']:
            testTitle=yml['ML']['testTitle']
            accuIndex=[i for i in range(len(testTitle)) if testTitle[i]=='acc'][0]
            lossIndex=[i for i in range(len(testTitle)) if testTitle[i]=='loss'][0]
            # return {"Data":testData,"Label":testLabel,'AccuIndex':accuIndex,'LossIndex':lossIndex}
            return {"Data": edfData, "Label": edfLabel, 'AccuIndex': accuIndex, 'LossIndex': lossIndex}
        else:
            return {"Data": edfData, "Label": edfLabel}
    return None




if __name__ == '__main__':

    #设置多进程启动方式
    multiprocessing.set_start_method('forkserver',force=True)
    print('————————————————————————Main1————————————————————————————————:', os.getpid())
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]),Loader=yaml.FullLoader)
    getLayer = None
    getData =None

    # Import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)
    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']

    saveFolder = "%s/%s" % (basicFolder, folderName)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
        if 'curFolderIndex' in yml['ML']:
            yml['ML']['curFolderIndex']=0
        else:
            yml['ML'].setdefault('curFolderIndex', 0)
        with open(sys.argv[1], "w") as f:
            yaml.dump(yml, f,sort_keys=False)

    mainLog=createLog(basicFolder,folderName)
    # channDic = getChannels()
    # eventTypeDic = getEvents()
    channDic = getChannels()
    #设置测试数据参数
    Dataset = yml['Meta']['Datasets']  # Dataset name:ex[BCIC2a/OpenBMI]
    Datatype = 'time_domain'
    num_class = yml['Meta']['ClassNum']  # number of classes:ex[2,3,4]

    if Dataset == 'OpenBMI':
        num_subject = 54
        # 这部分只是为了计算权重
        if num_class == 2:
            eventTypeDic = {
                0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
        if num_class == 3:
            eventTypeDic = {
                0: {'Name': 'rest', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset == "BCIC2a":
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
        if num_class == 2:
            eventTypeDic = {
                1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
                2: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
            }
    elif Dataset =='StrokeSh':
        num_subject = 14
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
    # min2net数据路径

    BasePath = yml['Meta']['initDataFolder']
    TrainType = yml['Meta']['TrainType']
    pick_smp_freq = yml['Meta']['downSampling']
    Channel_format = yml['Meta']['Channel']
    minFreq = yml['Meta']['minFreq']
    maxFreq = yml['Meta']['maxFreq']
    step = yml['Meta']['step']
    interp = yml['Meta']['interpType']
    Min2DataPath = BasePath + '/Traindata/{}/{}_class/{}Hz_{}chan_{}_{}Hz_{}_{}_padding'.format(TrainType, num_class,pick_smp_freq,
                                                                                  Channel_format,minFreq,maxFreq,step,interp)

    # Min2DataPath = '/data/Running/BCI_IV_2a/Traindata/independent/2_class/400Hz_49chan_0.5_100Hz_0.06_zero_padding'
    # Min2DataPath = "/data0/meya/Data/MI/datasets/Physionet/EEGML"
    log = logging.getLogger()
    subsample = int(yml['Meta']['subsample']) + 1

    runInProcess = True
    if 'runInProcess' in yml['ML']:
        runInProcess = yml['ML']['runInProcess']
    timeStr=dt.datetime.now().strftime('%Y%m%d%H%M')
    persub = 0
    # for sub in range(1,num_subject+1):
    if yml['Meta']['SelectSub']:
        # select_sub = [1,2,3,7,10,11,24,25,26,27]
        # select_sub = [4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        # select_sub = list(range(1,21))
        select_sub = list(range(1, 55))
    else:
        # select_sub = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        #               20, 21, 22, 23, 24, 25, 26, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        #               41,42,43,44,45,46,47,48,49,50,51,52,53,54]
        select_sub = list(range(1,num_subject+1))
    for sub in select_sub:
    # for sub in range(1,num_subject+1):
        # sub = crossValue['Train'][i]
        excel_sub=sub
        try:
            #5
            folderNum = yml['ML']['folderNum']
            for folder in range(1,6):
                test_sub=[]
                #模型路径
                curModelFolder = '%s/%s/folder_%d' % (saveFolder,'sub{0:02d}'.format(sub), folder)
                # curModelFolder = '%s/sub%d/folder_%d' % (saveFolder, sub, folder)
                if not os.path.exists(curModelFolder):
                    continue
                print('Testing subject:%d in folder Index: %d' %(sub,folder))
                # edf_list = glob.glob(joinPath(yml['Meta']['initDataFolder'], 'chb{0:02d}'.format(sub), '*.edf'))
                # folderNum=len(edf_list)
                test_sub.append(sub)
                # test_Inter(ML_Model, yml, sub, eventTypeDic,folderName,folderNum=folderNum,dateTimeStr=timeStr,getTestData=getTestData,checkCrossValue=False,testFunc=test_Model)
                if runInProcess:
                    p = multiprocessing.Process(target=test2,args=(ML_Model,yml,excel_sub,test_sub,persub,loadData,loadData_test,getData,
                                                                   Min2DataPath,channDic,eventTypeDic,curModelFolder,folderName,timeStr,folder))
                    p.start()
                    p.join()
                else:
                    test2(ML_Model, yml,excel_sub, test_sub,persub, loadData, loadData_test,getData, Min2DataPath,channDic, eventTypeDic,
                      curModelFolder, folderName,dateTimeStr=timeStr, folderIndex=folder)
                tf.keras.backend.clear_session()
        except Exception as err:
            raise err
    print("Done.")
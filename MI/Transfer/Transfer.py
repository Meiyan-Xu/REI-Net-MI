import os, tensorflow as tf
import xlwt
import xlrd
from xlutils.copy import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import sys

sys.path.append('../..')
import yaml
import logging
import numpy as np
from meya.fileAction import joinPath,saveYML
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from MI.EEGML import ML_Model
import os
import tensorflow.keras as keras
import multiprocessing
import pickle
import time
import datetime
import random

from sklearn.utils.class_weight import compute_class_weight
from meya.loadModel import GetModelPath
from MI.Transfer.Tool.load_data_2 import get_trans_data_OpenBMI
from tensorflow.keras.optimizers import Nadam
import tensorflow.keras.backend as K
from sklearn.metrics import recall_score, confusion_matrix, f1_score, precision_score
from tensorflow.keras.models import Model as M
from meya.effect import calculate_performance_dual

#设置随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def setDict(data_path,sub,fold):
    all_data = {}
    for i in range(1,fold+1):
        TraDataPath = data_path+'/X_train_S{0:03d}_fold{1:03d}.npy'.format(sub,i)
        TralabelPath = data_path + '/y_train_S{0:03d}_fold{1:03d}.npy'.format(sub, i)
        TraData = np.load(TraDataPath,allow_pickle=True)
        Tralabel = np.load(TralabelPath,allow_pickle=True)

        ValDataPath = data_path+'/X_val_S{0:03d}_fold{1:03d}.npy'.format(sub,i)
        VallabelPath = data_path + '/y_val_S{0:03d}_fold{1:03d}.npy'.format(sub, i)
        valData = np.load(ValDataPath, allow_pickle=True)
        vallabel = np.load(VallabelPath, allow_pickle=True)

        TeDataPath = data_path + '/X_test_S{0:03d}_fold{1:03d}.npy'.format(sub, i)
        TelabelPath = data_path + '/y_test_S{0:03d}_fold{1:03d}.npy'.format(sub, i)
        teData = np.load(TeDataPath, allow_pickle=True)
        telabel = np.load(TelabelPath, allow_pickle=True)

        all_data['fune_tra_fold{}_data'.format(i)] = TraData
        all_data['fune_tra_fold{}_label'.format(i)] = Tralabel
        all_data['fune_val_fold{}_data'.format(i)] = valData
        all_data['fune_val_fold{}_label'.format(i)] = vallabel
        all_data['test_fold{}_data'.format(i)] = teData
        all_data['test_fold{}_label'.format(i)] = telabel

    return all_data

def save_predict(save_path,y_pre):
    np.save(save_path+'/y_pre.npy',y_pre)

def Transfer(yml, sub,fold, all_data, model_path, eventTypeDic,model_id):
    print('————————————————————————sub process————————————————————————:', os.getpid())
    print("Start Sub{} fold{} trainning".format(sub,fold+1))
    for i in range(5):
        print("TraData:{} TraLabel:{}".format(all_data['fune_tra_fold{}_data'.format(i+1)].shape,
                                              all_data['fune_tra_fold{}_label'.format(i+1)].shape))
        print("ValidData:{} ValidLabel:{}".format(all_data['fune_val_fold{}_data'.format(i + 1)].shape,
                                              all_data['fune_val_fold{}_label'.format(i + 1)].shape))
        print("testData:{} testLabel:{}".format(all_data['test_fold{}_data'.format(i + 1)].shape,
                                              all_data['test_fold{}_label'.format(i + 1)].shape))
        class_weight_tra = class_weight_f(eventTypeDic, all_data['fune_tra_fold{}_label'.format(i + 1)])
        print('tra calss-weight {}'.format(class_weight_tra))
        class_weight_val = class_weight_f(eventTypeDic, all_data['fune_val_fold{}_label'.format(i + 1)])
        print('val calss-weight {}'.format(class_weight_val))
        class_weight_te = class_weight_f(eventTypeDic, all_data['test_fold{}_label'.format(i+ 1)])
        print('test calss-weight {}'.format(class_weight_te))
    #load model
    if yml['Meta']['Datasets'] =='StrokeData':
        best_dir = joinPath(model_path, 'sub{0:02}'.format(1), 'best')
    else:
        best_dir = joinPath(model_path, 'sub{0:02}'.format(sub), 'best')
    pkl_file, h5_file,_ = GetModelPath(best_dir)
    from_weight = True
    best_model_path = joinPath(best_dir, h5_file)
    model = keras.models.load_model(best_model_path)
    modelNet = ML_Model(yml, eventTypeDic,dataShape=yml['ML']['dataShape'])
    # if not isinstance(ML_Model, modelComFun):
    #     modelNet = ML_Model(yml, eventTypeDic)
    if from_weight:
        modelNet.build_model(yml)
    modelNet.get_model(best_model_path, from_weight)
    print("Base Model")
    # modelNet.model.summary()
    #set model
    transfer_model = set_model(yml,modelNet,model_id)
    #
    # feature_extractor = M(inputs=transfer_model.inputs, outputs=transfer_model.get_layer(index=-2).output)

    #创建实验结果保存文件夹
    ratio = yml['Trans']['ratio']
    excelFileName = yml['Meta']['excelFileName'] + '_{}%'.format(ratio * 100)
    excelfile = joinPath(yml['Meta']['basicFolder'], yml['Meta']['folderName'], excelFileName)
    if not os.path.exists(excelfile):
        os.makedirs(excelfile)
    #compile
    #tensorboard 路径
    log_dir = excelfile+"/tensorboard_log/sub_{}/fold_{}/".format(sub,fold)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    callback = [
        EarlyStopping(monitor=yml['ML']['callback'], min_delta=0, patience=30, verbose=0, mode=yml['ML']['backMode'],
                      restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=yml['ML']['batch_size'],
                    write_graph=True, write_images=True, update_freq='epoch')
    ]
    class_weight = class_weight_f(eventTypeDic, all_data['fune_tra_fold{}_label'.format(fold + 1)])

    loss = yml['ML']['loss']
    if loss == 'categorical_crossentropy':
        Acc = keras.metrics.CategoricalAccuracy(name='acc')
    else:
        Acc = keras.metrics.BinaryAccuracy(name='acc')

    transfer_model.compile(loss=loss,
                     optimizer=Nadam(learning_rate=yml['ML']['learningRate'], beta_1=0.9, beta_2=0.999),
                     metrics=[Acc,
                              keras.metrics.TruePositives(name='tp'),
                              keras.metrics.FalsePositives(name='fp'),
                              keras.metrics.TrueNegatives(name='tn'),
                              keras.metrics.FalseNegatives(name='fn'),
                              keras.metrics.Precision(name='precision'),
                              keras.metrics.Recall(name='recall'),
                              keras.metrics.AUC(name='auc')])

    start_time = time.time()

    transfer_model.fit(
        x=all_data['fune_tra_fold{}_data'.format(fold + 1)], y=all_data['fune_tra_fold{}_label'.format(fold + 1)],
        shuffle=yml['ML']['shuffle'],
        batch_size=yml['ML']['batch_size'],
        validation_data=(all_data['fune_val_fold{}_data'.format(fold+1)],all_data['fune_val_fold{}_label'.format(fold+1)]),
        epochs=yml['ML']['trainEpoch'],
        callbacks=callback,
        class_weight=class_weight,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 获取最后一层的特征值
    # feature_extractor = M(inputs=transfer_model.inputs, outputs=transfer_model.get_layer(index=-2).output)
    # feature = feature_extractor.predict(all_data['test_fold{}_data'.format(fold + 1)])
    # 保存预测结果 以便后续分析

    print('Save Done')
    labels_test = all_data['test_fold{}_label'.format(fold + 1)]
    y_pre = transfer_model.predict(all_data['test_fold{}_data'.format(fold + 1)])
    save_path = excelfile + '/predict/sub{}/fold{}'.format(sub, fold + 1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_predict(save_path, y_pre)

    result = transfer_model.evaluate(all_data['test_fold{}_data'.format(fold + 1)], all_data['test_fold{}_label'.format(fold + 1)])
    # 获取model.evalute中的评价指标
    if loss == "categorical_crossentropy":
        class_num = yml['Meta']['ClassNum']
        if class_num == 2:
            label = [0, 1]
        elif class_num == 3:
            label = [0, 1, 2]
        elif class_num == 4:
            label = [0, 1, 2, 3]
        y_t = K.argmax(labels_test, axis=-1)
        y_p = K.argmax(y_pre, axis=-1)
        f1 = f1_score(y_t, y_p, labels=label, average='macro')
        TP, FP, FN, TN, sens, speci, preci = calculate_performance_dual(labels_test, y_pre, hotCode=True)
    elif loss == "binary_crossentropy":
        labels_test = labels_test.reshape(len(labels_test), 1)
        y_pre = np.around(y_pre, 0).astype(int)
        TN, FP, FN, TP = confusion_matrix(labels_test, y_pre).ravel()
        sens = recall_score(labels_test, y_pre, average='binary')
        f1 = f1_score(labels_test, y_pre, average='binary')
        preci = precision_score(labels_test, y_pre, average='binary')
        speci = TN / (TN + FP)

    #save result
    make_excel(yml,sub,fold,result,TP,FP,TN,FN,sens,speci,preci,f1,elapsed_time,excelfile,excelFileName)

def make_excel(yml,sub,fold,result,TP,FP,TN,FN,sens,speci,preci,f1,time,excelfile,excelFileName):
    colIndex = 2
    excelPath = joinPath(excelfile, '%s_test.xls' % excelFileName)
    if os.path.exists(excelPath):
        try:
            r_wb = xlrd.open_workbook(excelPath)
            w_wb = copy(r_wb)
        except Exception as err:
            print(err)
            os.remove(excelPath)
            w_wb = xlwt.Workbook()
    else:
        w_wb = xlwt.Workbook()  # 创建excel文件
    sheetName = 'all_result'
    try:
        test_sheet = w_wb.get_sheet(sheetName)  # 如果已经存在sheet则会选择相应的sheet
    except:
        test_sheet = w_wb.add_sheet(sheetName, cell_overwrite_ok=True)  # 如果不存在则新建sheet

    titleStyle = xlwt.easyxf('font: bold 1')
    test_sheet.write(0, 0, 'Subject', titleStyle)
    test_sheet.write(0, 1, 'Folder', titleStyle)
    test_sheet.write(0, 2, 'Accuracy', titleStyle)
    test_sheet.write(0, 3, 'Loss', titleStyle)
    test_sheet.write(0, 4, 'Precision', titleStyle)
    test_sheet.write(0, 5, 'TP', titleStyle)
    test_sheet.write(0, 6, 'FP', titleStyle)
    test_sheet.write(0, 7, 'TN', titleStyle)
    test_sheet.write(0, 8, 'FN', titleStyle)
    test_sheet.write(0, 9, 'F1-score', titleStyle)
    test_sheet.write(0, 10,'Sensitivity', titleStyle)
    test_sheet.write(0, 11,'Specificity', titleStyle)
    test_sheet.write(0, 12, 'trainTime', titleStyle)

    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), 0, str(sub))  # 添加被试id
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), 1, str(fold+1))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex, float(result[1]))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 1, float(result[0]))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 2, float(preci))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 3, int(TP))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 4, int(FP))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 5, int(TN))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 6, int(FN))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 7, float(f1))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 8, float(sens))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold+1), colIndex + 9, float(speci))
    test_sheet.write((sub - 1) * yml['ML']['folderNum'] + (fold + 1), colIndex + 10, float(time))
    saveYML(test_sheet,yml,1,colIndex+14)
    print(
        "Subject %d : accuracy- %f , loss- %f, presion- %f, recall- %f,specificity- %f,f1-score- %f, tp- %d, tn- %d, fp- %d, fn- %d" % (
        sub, result[1], result[0], preci, sens, speci, f1, TP, TN, FP, FN))
    w_wb.save(excelPath)

def add_dense(yml,model):
    filter_num = yml['Trans']['Dense_filter_num']
    print('------ Add Dense number of {} ------'.format(yml['Trans']['Dense']))
    print('------ Dense {} filter set {} ------'.format(yml['Trans']['Dense'],filter_num[:yml['Trans']['Dense']]))
    for i in range(yml['Trans']['Dense']):
        model = keras.layers.Dense(filter_num[i], name='Common_dense_{}'.format(i+1))(model)
        model = keras.layers.Dropout(0.25,name='Dr_{}'.format(i+1))(model)
    if yml['Trans']['Dense'] !=0:
        if yml["ML"]['loss'] == 'binary_crossentropy':
            model = keras.layers.Dense(1, name='final_dense')(model)
            model = keras.layers.Activation('sigmoid', name='sigmoid')(model)
        elif yml['ML']['loss'] == 'final_dense':
            model = keras.layers.Dense(2, name='MLP')(model)
            model = keras.layers.Activation('softmax', name='softmax')(model)
    return model


def set_model(yml,model,model_id):
    # 如果要新加层 设置为23 如果不加
    if yml['Trans']['Dense'] ==0:
        print('--- not add layers use origin classifier ---')
        base_model = M(inputs=model.model.input, outputs=model.model.output)
    else:
        print('--- add layers use new classifier ---')
        base_model = M(inputs=model.model.input, outputs=model.model.get_layer(index=23).output)

    print(id(base_model))
    if id(base_model) not in model_id:
        model_id.append(id(base_model))
    else:
        print("model path Repeat")
        raise err
    #添加dense层
    if yml['Meta']['Datasets'] !='StrokeData':
        x = base_model.output
        x = add_dense(yml,x)
        transfer_model = M(inputs=base_model.input, outputs=x)
        trans_model = reset_model(yml, transfer_model)  # index = new 解冻最后一层卷积层 以及新添加的dense层
        trans_model.summary()
    else:
        trans_model = reset_model(yml,base_model)  # index = new 解冻最后一层卷积层 以及新添加的dense层
        trans_model.summary()
    return trans_model

def reset_model(yml,Model):
    #获取指定层的索引
    for index,layer in enumerate(Model.layers):
        select_layer = yml['Trans']['FuneLayer']
        if layer.name == select_layer:
            select_index = index
    print('------you select layer {}`s index is {}------\n'.format(select_layer,select_index))

    for layer in Model.layers:
        # 屏蔽预训练模型的权重
        layer.trainable = False

    #解冻指定层之后的所有层
    for layer in Model.layers[select_index:]:
        #解冻模型
        print("----- 解冻{}层 -----".format(layer.name))
        layer.trainable = True


    new_model = Model
    return new_model

def class_weight_f(eventTypeDic,label):
    classes = np.array(list(eventTypeDic.keys()))
    sklearnWeight = compute_class_weight(class_weight='balanced',classes=classes,y=label )
    classes = range(len(classes))
    trainWeight = dict(zip(classes, sklearnWeight))
    return trainWeight




if __name__ == '__main__':
    #multiprocess
    multiprocessing.set_start_method('forkserver', force=True)
    print('————————————————————————Main1——————————————————————————:', os.getpid())

    # import yml
    assert len(sys.argv) >= 2, 'Please enter model file of yaml.'
    print(sys.argv[1])
    yml = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    getLayer = None

    #import package
    for pkg, functions in yml['imports'].items():
        stri = 'from ' + pkg + ' import ' + ','.join(functions)
        exec(stri)

    # meta settings
    subsample = yml['Meta']['subsample']
    basicFolder = yml["Meta"]['basicFolder']
    folderName = yml["Meta"]['folderName']
    epochs = yml["ML"]["trainEpoch"]


    channDic = getChannels()

    doshuffle = False
    if "doShuffle" in yml['Meta']:
        # true
        doshuffle = yml['Meta']['doShuffle']

    Subnum = []
    step = yml['Meta']['step']
    segmentName = yml['Meta']['segmentName']
    log = logging.getLogger()
    global folderIndex
    folderIndex = 0

    crossIndexs = [0, 1, 2, 3, 4]

    # 模型加载路径
    model_path = joinPath(basicFolder,folderName)
    if yml["Meta"]['Datasets'] == 'BCIC2a':
        sub_all = list(range(1, 10))
        loss = yml['ML']['loss']
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
    elif yml['Meta']['Datasets'] == 'OpenBMI':
        sub_all = list(range(1, 55))
        # sub_all = [45,49,52]
        loss = yml['ML']['loss']
        eventTypeDic = {
            0: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
    elif yml['Meta']['Datasets'] == 'Kaya2018':
        sub_all = list(range(1, 12))
        loss = yml['ML']['loss']
        eventTypeDic = {
            0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }
    elif yml['Meta']['Datasets'] == 'StrokeData':
        sub_all = list(range(1, 51))
        loss = yml['ML']['loss']
        eventTypeDic = {
            0: {'Name': 'left', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False},
            1: {'Name': 'right', 'StaTime': 0, 'TimeSpan': 4000, 'IsExtend': False}
        }

    print('Dataset:', yml["Meta"]['Datasets'])
    print('loss:', loss)

    for sub in sub_all:
        model_id = []
        ratio = yml['Trans']['ratio']
        # load data 数据处理切记先交叉再分割
        if yml["Meta"]['Datasets'] == 'BCIC2a':
            matPath = yml['Trans']['matPath']
            print("load data from {}".format(matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100, sub)))
            with open(matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100, sub), 'rb') as f:
                all_data = pickle.load(f)
        elif yml["Meta"]['Datasets'] == 'OpenBMI':
            matPath = yml['Trans']['matPath']
            print("load data from {}".format(
                matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'], ratio * 100, sub)))
            with open(matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'], ratio * 100, sub), 'rb') as f:
                all_data = pickle.load(f)
        elif yml['Meta']['Datasets'] == 'Kaya2018':
            matPath = yml['Trans']['matPath']
            print("load data from {}".format(matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100, sub)))
            with open(matPath + '/{}_{}%_sub{}.pkl'.format(yml['Trans']['matName'],ratio*100, sub), 'rb') as f:
                all_data = pickle.load(f)
        elif yml['Meta']['Datasets'] == 'StrokeData':
            matPath = yml['Trans']['matPath']
            print("load data from {}".format(matPath))
            all_data = setDict(matPath,sub,5)


        for fold in range(0,5):
            try:
                if yml['ML']['runInProcess']:
                    p = multiprocessing.Process(target=Transfer, args=(
                        yml, sub, fold,all_data, model_path, eventTypeDic,model_id))
                    p.start()
                    p.join()
                else:
                    # getData_chanSeq as getData
                    Transfer(yml, sub, fold,all_data, model_path, eventTypeDic,model_id)
            except Exception as err:
                print(err)
                log.error(err)
                raise err













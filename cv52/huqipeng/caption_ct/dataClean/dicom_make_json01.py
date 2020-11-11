
import pandas as pd
import os
import json
import pydicom


def pre_pro():
    patients = {}
    root_path = r'E:\data\NewCT\出血'
    annotation_xlsx = r'E:\data\NewCT\脑出血.xlsx'
    json_patients = r'E:\data\data1023\patients01.json'

    # 遍历读取Excel信息录入patients
    patient_num = 0
    excel_data = pd.read_excel(annotation_xlsx, sheet_name='sheet1')
    # data = np.array(excel_data.loc[2,:])
    # print( data)
    for i in range(len(excel_data['acc'])):
        acc_num = str(excel_data['acc'][i])
        if patients.get(acc_num) is None:
            patients[acc_num] = {'id': '', 'numkey': i+2, 'img_path': root_path+'\\'+acc_num, 'SeriesInstanceUID':{},'StudyInstanceID':{}}
            patient_num += 1
    print('patient_num:%d' % patient_num)

    # 遍历读取dicom信息录入patients
    for acc,patient in patients.items():
        patient_dir = patient['img_path']
        for root, dirs, files in os.walk(patient_dir):
            if (len(files) > 0) & (len(dirs) == 0):
                dcm = pydicom.read_file(root + '\\' + files[0])
                id = dcm.PatientID
                patients[acc]['id'] = id
                for i, file in enumerate(files):
                    dcm1 = pydicom.read_file(root + '\\' + file)
                    study_date = dcm1.StudyDate
                    series_date = dcm1.SeriesDate
                    SeriesID = dcm1.SeriesInstanceUID
                    StudyID = dcm1.StudyID
                    if patients[acc]['SeriesInstanceUID'].get(SeriesID) is None:
                        patients[acc]['SeriesInstanceUID'][SeriesID] = {"num": 1, 'started_num': i, 'series_date': series_date}
                    else:
                        patients[acc]['SeriesInstanceUID'][SeriesID]['num'] += 1

                    if patients[acc]['StudyInstanceID'].get(StudyID) is None:
                        patients[acc]['StudyInstanceID'][StudyID] = {"series_id": [SeriesID], 'study_date': study_date}
                    else:
                        if SeriesID not in patients[acc]['StudyInstanceID'][StudyID]["series_id"]:
                            patients[acc]['StudyInstanceID'][StudyID]["series_id"].append(SeriesID)

    # patients = json.load(open(json_patients, 'r'))
    # 信息扫描完成,去除冗余信息，生成对应patients文件
    for acc in list(patients.keys()):
        # 去除series长度小于等于2的检查序列，需要删除对应的series和study内容
        StudyInstance = patients[acc]['StudyInstanceID']
        for series_id in list(patients[acc]['SeriesInstanceUID'].keys()):
            series_num = patients[acc]['SeriesInstanceUID'][series_id]['num']
            if series_num <= 2:
                del patients[acc]['SeriesInstanceUID'][series_id]
                for study,s_list in StudyInstance.items():
                    if series_id in StudyInstance[study]["series_id"]:
                        StudyInstance[study]["series_id"].remove(series_id)
                patients[acc]['StudyInstanceID'] = StudyInstance

    with open(json_patients, 'w') as f:
        json.dump(patients, f)


pre_pro()

import pandas as pd
import numpy as np

tr_file = '/home/byeolyi/activelearning/Spreadsheets/prime_trex_compressed.csv'
te_file = '/home/byeolyi/activelearning/Spreadsheets/prime_trex_test_new.csv'

def get_stats(tr_file, te_file):
    sheet1 = pd.read_csv(tr_file)
    sheet2 = pd.read_csv(te_file)
    sheet = pd.concat([sheet1, sheet2])
    add_sheet = '/home/byeolyi/Dropbox (GhassanGT)/Zoe/InSync/BIGandDATA/stats_intermed.xlsx'
    output_file = '/home/byeolyi/Dropbox (GhassanGT)/Zoe/InSync/BIGandDATA/Patient_Stats.xlsx'
    intermed_data = pd.read_excel(add_sheet)
    unique_ids = []
    visit_dic = {}
    for i in range(len(sheet)):
        path = sheet['File_Path'].iloc[i]
        type = path.split('/')[1]
        if type == 'Prime_FULL':
            id = path.split('/')[2]
            if id not in unique_ids:
                unique_ids.append(id)
                visit_dic[id] = sheet['Visit'].iloc[i]
            else:
                current_visit_num = visit_dic[id]
                if int(sheet['Visit'].iloc[i]) > int(current_visit_num):
                    visit_dic[id] = sheet['Visit'].iloc[i]
                # id already accounted for; increment visit
        else:
            id = path.split('/')[3]
            if id not in unique_ids:
                unique_ids.append(id)
                visit_dic[id] = sheet['Visit'].iloc[i]
            else:
                current_visit_num = visit_dic[id]
                if int(sheet['Visit'].iloc[i]) > int(current_visit_num):
                    visit_dic[id] = sheet['Visit'].iloc[i]
                # id already accounted for; increment visit
    # create dataframe from patient IDs and mapped visit vals
    new_df = pd.DataFrame(visit_dic.items())
    for j in range(len(new_df)):
        cur_id = new_df.iloc[j, 0]
        for vals in range(len(intermed_data)):
            search_id = intermed_data['Patients'].iloc[vals]
            if search_id == cur_id:
                weeks = intermed_data['# weeks enrolled'].iloc[vals]
                inj = intermed_data['# injections'].iloc[vals]
                new_df.at[j, 'Enrolled Weeks'] = weeks
                new_df.at[j, '# Injections'] = inj
    new_df.to_csv(output_file)

    return

## for oasis, cxr, etc
def patient_visits(file='/media/zoe/HD/Datasets/OASIS-2/Spreadsheets/new_train_file.csv', id_name='Subject ID'):
    spreadsheet = pd.read_csv(file)
    num_visits = []
    print('Length of spreadsheet: ', len(spreadsheet))
    for i in range(len(spreadsheet)):
        current_id = str(spreadsheet[id_name].iloc[i])
        #print(current_id)
        spreadsheet[id_name] = spreadsheet[id_name].astype(str)
        same_patient = spreadsheet[spreadsheet[id_name].str.findall(current_id).apply(len) > 0]
        patient_visits = len(same_patient['Visit'].value_counts())
        num_visits.append(patient_visits)
    print('Max visit observed by a patient: ', max(num_visits))
    print('Average # of patient visits: ', np.mean(np.array(num_visits)))
    return

#patient_visits(file='/media/zoe/HD/Datasets/CXR8/Spreadsheets/train_file.csv', id_name='Patient ID')
# max # of visits in OASIS is 5; average of 2.69 visits
# in CXR: max visits = 108; average = 16

def shorten_train(file='/media/zoe/HD/Datasets/CXR8/Spreadsheets/train_file.csv'):
    spreadsheet = pd.read_csv(file)
    multiple = '|'
    no = 'No Finding'
    label5 = 'Pneumothorax'
    label2 = 'Effusion'
    label3 = 'Cardiomegaly'
    label4 = 'Infiltration'
    idxs = []
    idxs1 = []
    unique_patients_a = []
    unique_patients_b = []
    unique_patients_c = []
    unique_patients_d = []
    unique_patients_e = []
    # imbalance in infil number
    infil_limit = 20000
    count1 = 0
    count2 = 0
    count3 = 0
    count5 = 0
    count6 = 0
    for row in range(len(spreadsheet)):
        current_finding = spreadsheet['Finding Labels'].iloc[row]
        if multiple not in current_finding:
            if label2 in current_finding:
                current_id = str(spreadsheet['Patient ID'].iloc[row])
                if current_id not in unique_patients_a:
                    unique_patients_a.append(current_id)
                count1 += 1
                idxs1.append(row)
            elif label5 in current_finding:
                current_id = str(spreadsheet['Patient ID'].iloc[row])
                if current_id not in unique_patients_a:
                    unique_patients_d.append(current_id)
                count5 += 1
                idxs1.append(row)
            elif label3 in current_finding:
                current_id = str(spreadsheet['Patient ID'].iloc[row])
                if current_id not in unique_patients_b:
                    unique_patients_b.append(current_id)
                count2 += 1
                idxs.append(row)
            elif label4 in current_finding:
                current_id = str(spreadsheet['Patient ID'].iloc[row])
                if current_id not in unique_patients_c:
                    if len(unique_patients_c) != infil_limit:
                        unique_patients_c.append(current_id)
                        count3 += 1
                        idxs.append(row)
                else:
                    count3 += 1
                    idxs.append(row)
    #         elif no in current_finding:
    #             current_id = str(spreadsheet['Patient ID'].iloc[row])
    #             if current_id not in unique_patients_e:
    #                 if len(unique_patients_e) != 5466:
    #                     unique_patients_e.append(current_id)
    #                     count6 += 1
    #                     idxs1.append(row)
    #             else:
    #                 count6 += 1
    #                 idxs1.append(row)
    # print('Eff unique patients: ', len(unique_patients_a))
    print(count1)
    print('Infil unique patients: ', len(unique_patients_c))
    print(count3)
    print('Cardio unique patients: ', len(unique_patients_b))
    print(count2)
    print('Pneu unique patients: ', len(unique_patients_d))
    print(count5)
    # print('healthy unique patients: ', len(unique_patients_e))
    # print(count6)
    idxs_arr = np.array(idxs1)
    new_df = spreadsheet.iloc[idxs_arr].reset_index().iloc[:,1:]
    ##
    new_df = new_df.drop(['Label'], axis=1)

    for row in range(len(new_df)):
        current_finding = new_df['Finding Labels'].iloc[row]
        if label2 in current_finding:
            label = 1
        elif label5 in current_finding:
            label = 0
        else:
            print('ERROR!!!!')
            print(row)
            print(current_finding)
        new_df.loc[row, 'Label'] = label
    new_df.to_csv('/media/zoe/HD/Datasets/CXR8/Spreadsheets/total_file-2.csv', index=False)
    print(new_df['Label'].value_counts())

    return

def one_vs_all(file, chosen):
    labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    spreadsheet = pd.read_csv(file)

    spreadsheet = spreadsheet.drop(['Label'], axis=1)
    idxs = []
    for i in range(len(spreadsheet)):
        current_finding = spreadsheet['Finding Labels'].iloc[i]
        if '|' not in current_finding:
            if current_finding in labels:
                idxs.append(i)
                if current_finding == chosen:
                    label = 1
                else:
                    label = 0
                spreadsheet.loc[i, 'Label'] = label
    new_df = spreadsheet.iloc[np.array(idxs)].reset_index().iloc[:,1:]
    new_df.to_csv('/media/zoe/HD/Datasets/CXR8/Spreadsheets/total_file-3.csv', index=False)
    print(new_df['Label'].value_counts())
    return

def patient_overlap(file1, file2):
    s1 = pd.read_csv(file1)
    s2 = pd.read_csv(file2)
    unique_patients_1 = []
    unique_patients_2 = []
    count = 0
    for row in range(len(s1)):
        current_id = s1['Subject'].iloc[row]
        if current_id not in unique_patients_1:
            unique_patients_1.append(current_id)
            same_patient = s2[s2['Subject'].str.findall(current_id).apply(len) > 0]
            if len(same_patient) > 0:
                count += 1
    for j in range(len(s2)):
        current_id = s1['Subject'].iloc[j]
        if current_id not in unique_patients_2:
            unique_patients_2.append(current_id)
    print('Unique patients in 1: ', len(unique_patients_1))
    print('Unique patients in 2: ', len(unique_patients_2))
    print('How many overlap between the two: ', count)

    return
file = '/media/zoe/HD/Datasets/CXR8/Spreadsheets/total_files.csv'
fi = pd.read_csv(file)
#shorten_train(file)
#print(fi['Label'].value_counts())
# patient_overlap(file1='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/BIGandDATA/Spreadsheets/ADNI/ADNI1_Complete_2Yr_1.5T_6_30_2023.csv',
#                 file2='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/BIGandDATA/Spreadsheets/ADNI/ADNI1_Complete_3Yr_1.5T_6_30_2023.csv')
one_vs_all(file=file, chosen='Cardiomegaly')

# Between ADNI-1 year1 and 2:
# Unique patients in 1:  636
# Unique patients in 2:  506
# How many overlap between the two:  455
# Between ADNI-1 year 2 and year 3:
# Unique patients in 1:  458
# Unique patients in 2:  212
# How many overlap between the two:  364

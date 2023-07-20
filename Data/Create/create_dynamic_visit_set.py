import pandas as pd
file = '/home/byeolyi/activelearning/Spreadsheets/prime_trex_compressed_test.csv'

test_file = pd.read_csv(file)
new_path = '/home/byeolyi/activelearning/Spreadsheets/prime_trex_test_dynamic.csv'
max_visits = 20
num_sample = int(80/2)
new = pd.DataFrame([])

for visit in range(1, max_visits):
    selection = test_file[test_file['Visit'] == visit].groupby('Label').sample(n=num_sample, replace=False)
    new = pd.concat([new, selection])

for vis in range(max_visits, 22):
    selection = test_file[test_file['Visit'] == vis].groupby('Label').sample(n=(40), replace=False)
    new = pd.concat([new, selection])

new = new.reset_index().iloc[:, 1:]
new.to_csv(new_path)
import pandas as pd
import numpy as np

main_csv = pd.read_csv('/home/gujingxiao/projects/HumanProteinData/duplicates/HPAv18RGB.csv')
dupli_csv = pd.read_csv('/home/gujingxiao/projects/HumanProteinData/duplicates/TestEtraMatchingUnder_259_R14_G12_B10.csv')

main_id = main_csv['Id']
main_label = main_csv['Target']

dupli_extra = dupli_csv['Extra']
dupli_test = dupli_csv['Test']

new_id = []
new_label = []
count = 0
for index, item in enumerate(dupli_extra):
    for idx in range(len(main_id)):
        if item == main_id[idx]:
            print(count, idx, dupli_test[index], main_label[idx])
            count += 1
            new_id.append(dupli_test[index])
            new_label.append(main_label[idx])

save = pd.DataFrame({'Id':new_id, 'Target':new_label})
save.to_csv('duplicate_labels.csv', index=False)
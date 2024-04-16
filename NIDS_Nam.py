import pandas as pd
import numpy as np
from scipy.stats import entropy
import struct, socket
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score



def ip_to_numeric(ip):
    return struct.unpack("!L", socket.inet_aton(ip))[0]


train_data = pd.read_csv('./clean_train.csv',
                         dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"}, low_memory=False)

val_data = pd.read_csv('./clean_val.csv', dtype={"srcip": "string", "sport": "string", "dstip": "string", "dsport": "string"}, low_memory=False)


train_data['srcip'] = train_data['srcip'].apply(ip_to_numeric)
val_data['srcip'] = val_data['srcip'].apply(ip_to_numeric)
train_data['dstip'] = train_data['dstip'].apply(ip_to_numeric)
val_data['dstip'] = val_data['dstip'].apply(ip_to_numeric)

train_data['sport'] = pd.to_numeric(train_data['sport'], errors='coerce')
val_data['sport'] = pd.to_numeric(val_data['sport'], errors='coerce')
train_data['dsport'] = pd.to_numeric(train_data['dsport'], errors='coerce')
val_data['dsport'] = pd.to_numeric(val_data['dsport'], errors='coerce')

train_data['ct_ftp_cmd'] = train_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)
val_data['ct_ftp_cmd'] = val_data['ct_ftp_cmd'].replace(' ', '-1').fillna('-1').astype(str)

label_column = 'Label'
attack_cat_column = 'attack_cat'

selected_features = []
selected_features2 = []



print('------------------------- PART 1: Feature Analysis and Selection --------------------------------')
print('\n')
print('Entropy feature ranks for Label')
print('\n')

information_gain = {}

# Entropy for each column using entropy method
target_counts = train_data[label_column].value_counts(normalize=True)
target_entropy = entropy(target_counts, base=2)

# Calculate information gain for each column and rank them
for column in train_data.columns:
    if column != label_column and column != attack_cat_column:
        feature_counts = train_data[column].value_counts(normalize=True)
        feature_entropy = entropy(feature_counts, base=2)
        
        information_gain[column] = target_entropy - feature_entropy

ranked_features = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)

for i, (feature, gain) in enumerate(ranked_features, start=1):
    print(f"{i}. Feature: {feature}, Information Gain: {gain}")
    if (gain > 0):
        selected_features.append(feature)

# This code is for outputting a csv file
# ranked = pd.DataFrame(ranked_features, columns=['Feature', 'Information Gain'])
# ranked.to_csv("ranked_features.csv", index=False)

print('\n')
print('Entropy feature ranks for attack_cat')
print('\n')

target_counts = train_data[attack_cat_column].value_counts(normalize=True)
target_entropy = entropy(target_counts, base=2)

# Calculate information gain for each column and rank them
for column in train_data.columns:
    if column != label_column and column != attack_cat_column:
        feature_counts = train_data[column].value_counts(normalize=True)
        feature_entropy = entropy(feature_counts, base=2)
        
        information_gain[column] = target_entropy - feature_entropy

ranked_features = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)

for i, (feature, gain) in enumerate(ranked_features, start=1):
    print(f"{i}. Feature: {feature}, Information Gain: {gain}")
    if (gain > 0):
        selected_features2.append(feature)


# This code is for outputting a csv file
# ranked = pd.DataFrame(ranked_features, columns=['Feature', 'Information Gain'])
# ranked.to_csv("ranked_features.csv", index=False)

print('\n')
print('----------------------------- PART 2: Label Classification --------------------------------------')
print('\n')

#selected_features = ["is_sm_ips_ports", "trans_depth", "res_bdy_len", "ct_ftp_cmd", "is_ftp_login","dwin","swin"]


X_train = train_data[selected_features]
y_train = train_data[label_column]

X_test = val_data[selected_features]
y_test = val_data[label_column]


classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

report = classification_report(y_test, pred, output_dict=True)

print("Using classifications WITH selected features")
print("Classifier: Gradient Boosting")
print(f"Validation Accuracy: {report['accuracy']}")
print("Validation Classification Report:")
print("                 precision   recall      f1-score   support")
print(f"           0     {round(report['0']['precision'], 2)}        {round(report['0']['recall'], 2)}        {round(report['0']['f1-score'], 2)}        {round(report['0']['support'])}")
print(f"           1     {round(report['1']['precision'], 2)}        {round(report['1']['recall'], 2)}        {round(report['1']['f1-score'], 2)}       {round(report['1']['support'])}")
print(f"    accuracy                             {round(report['accuracy'], 2)}")
print(f"   macro avg     {round(report['macro avg']['precision'], 2)}        {round(report['macro avg']['recall'], 2)}        {round(report['macro avg']['f1-score'], 2)}       {round(report['macro avg']['support'])}")
print(f"weighted avg     {round(report['weighted avg']['precision'], 2)}        {round(report['weighted avg']['recall'], 2)}        {round(report['weighted avg']['f1-score'], 2)}       {round(report['weighted avg']['support'])}")

print('\n')
print('\n')
imputer = SimpleImputer(strategy='mean')

train_data_imputed = imputer.fit_transform(train_data.drop(columns=[label_column, attack_cat_column]))
val_data_imputed = imputer.transform(val_data.drop(columns=[label_column, attack_cat_column]))

X_train = train_data_imputed
y_train = train_data[label_column]

X_test = val_data_imputed
y_test = val_data[label_column]

classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

report = classification_report(y_test, pred, digits = 2, output_dict=True)

print("Using classifications WITHOUT selected features")
print("Classifier: Gradient Boosting")
print(f"Validation Accuracy: {report['accuracy']}")
print("Validation Classification Report:")
print("                 precision   recall      f1-score   support")
print(f"           0     {round(report['0']['precision'], 2)}        {round(report['0']['recall'], 2)}        {round(report['0']['f1-score'], 2)}        {round(report['0']['support'])}")
print(f"           1     {round(report['1']['precision'], 2)}        {round(report['1']['recall'], 2)}        {round(report['1']['f1-score'], 2)}       {round(report['1']['support'])}")
print(f"    accuracy                             {round(report['accuracy'], 2)}")
print(f"   macro avg     {round(report['macro avg']['precision'], 2)}        {round(report['macro avg']['recall'], 2)}        {round(report['macro avg']['f1-score'], 2)}       {round(report['macro avg']['support'])}")
print(f"weighted avg     {round(report['weighted avg']['precision'], 2)}        {round(report['weighted avg']['recall'], 2)}        {round(report['weighted avg']['f1-score'], 2)}       {round(report['weighted avg']['support'])}")


print('\n')
print('----------------------------- PART 3: attack_cat Classification ---------------------------------')
print('\n')

print('Classifying with selected features')
print('\n')

# selected_features2 = ["is_sm_ips_ports", "trans_depth", "res_bdy_len", "is_ftp_login", "ct_ftp_cmd", "dwin", "swin", "ct_flw_http_mthd", "ct_state_ttl", "dttl", "state", "sttl", "proto", "service"]
data_to_train = train_data[selected_features2]
label_to_train = train_data[attack_cat_column]

data_to_test = val_data[selected_features2]
label_to_test = val_data[attack_cat_column]

classifier = GradientBoostingClassifier()
classifier.fit(data_to_train, label_to_train)

pred = classifier.predict(data_to_test)

report = classification_report(label_to_test, pred, output_dict=True)

precision_micro = precision_score(label_to_test, pred, average='micro')
precision_macro = precision_score(label_to_test, pred, average='macro')
precision_weighted = precision_score(label_to_test, pred, average='weighted')

recall_micro = recall_score(label_to_test, pred, average='micro')
recall_macro = recall_score(label_to_test, pred, average='macro')
recall_weighted = recall_score(label_to_test, pred, average='weighted')

f1_micro = f1_score(label_to_test, pred, average='micro')
f1_macro = f1_score(label_to_test, pred, average='macro')
f1_weighted = f1_score(label_to_test, pred, average='weighted')


print("Using classifications WITH selected features")
print("Classifier: Gradient Boosting")
print(f"Validation Accuracy: {report['accuracy']}")
print("Validation Classification Report:")
print("                   precision   recall      f1-score   support")
print(f"          None     {round(report['-1']['precision'], 2)}        {round(report['-1']['recall'], 2)}        {round(report['-1']['f1-score'], 2)}        {round(report['-1']['support'])}")
print(f"       Generic     {round(report['0']['precision'], 2)}        {round(report['0']['recall'], 2)}        {round(report['0']['f1-score'], 2)}       {round(report['0']['support'])}")
print(f"       Fuzzers     {round(report['1']['precision'], 2)}        {round(report['1']['recall'], 2)}        {round(report['1']['f1-score'], 2)}        {round(report['1']['support'])}")
print(f"      Exploits     {round(report['2']['precision'], 2)}        {round(report['2']['recall'], 2)}        {round(report['2']['f1-score'], 2)}       {round(report['2']['support'])}")
print(f"           DoS     {round(report['3']['precision'], 2)}        {round(report['3']['recall'], 2)}        {round(report['3']['f1-score'], 2)}        {round(report['3']['support'])}")
print(f"Reconnaissance     {round(report['4']['precision'], 2)}        {round(report['4']['recall'], 2)}        {round(report['4']['f1-score'], 2)}       {round(report['4']['support'])}")
print(f"      Analysis     {round(report['5']['precision'], 2)}        {round(report['5']['recall'], 2)}        {round(report['5']['f1-score'], 2)}        {round(report['5']['support'])}")
print(f"     Shellcode     {round(report['6']['precision'], 2)}        {round(report['6']['recall'], 2)}        {round(report['6']['f1-score'], 2)}       {round(report['6']['support'])}")
print(f"      Backdoor     {round(report['7']['precision'], 2)}        {round(report['7']['recall'], 2)}        {round(report['7']['f1-score'], 2)}        {round(report['7']['support'])}")
print(f"         Worms     {round(report['8']['precision'], 2)}        {round(report['8']['recall'], 2)}        {round(report['8']['f1-score'], 2)}       {round(report['8']['support'])}")
print(f"     micro avg     {round(precision_micro, 2)}        {round(recall_micro, 2)}        {round(f1_micro, 2)}    {round(report['10']['support'])}")   
print(f"     macro avg     {round(report['macro avg']['precision'], 2)}        {round(report['macro avg']['recall'], 2)}        {round(report['macro avg']['f1-score'], 2)}      {round(report['macro avg']['support'])}") 
print(f"  weighted avg     {round(report['weighted avg']['precision'], 2)}        {round(report['weighted avg']['recall'], 2)}        {round(report['weighted avg']['f1-score'], 2)}     {round(report['weighted avg']['support'])}")

imputer = SimpleImputer(strategy='mean')

train_data_imputed = imputer.fit_transform(train_data.drop(columns=[label_column, attack_cat_column]))
val_data_imputed = imputer.transform(val_data.drop(columns=[label_column, attack_cat_column]))

label_to_train = train_data[attack_cat_column]
label_to_test = val_data[attack_cat_column]

classifier = GradientBoostingClassifier()
classifier.fit(train_data_imputed, label_to_train)

pred = classifier.predict(val_data_imputed)

report = classification_report(label_to_test, pred, output_dict=True)

precision_micro = precision_score(label_to_test, pred, average='micro')
precision_macro = precision_score(label_to_test, pred, average='macro')
precision_weighted = precision_score(label_to_test, pred, average='weighted')

recall_micro = recall_score(label_to_test, pred, average='micro')
recall_macro = recall_score(label_to_test, pred, average='macro')
recall_weighted = recall_score(label_to_test, pred, average='weighted')

f1_micro = f1_score(label_to_test, pred, average='micro')
f1_macro = f1_score(label_to_test, pred, average='macro')
f1_weighted = f1_score(label_to_test, pred, average='weighted')


print('\n')
print('Using classifications WITHOUT selected features')
print("Classifier: Gradient Boosting")
print(f"Validation Accuracy: {report['accuracy']}")
print("Validation Classification Report:")
print("                   precision   recall      f1-score   support")
print(f"          None     {round(report['-1']['precision'], 2)}        {round(report['-1']['recall'], 2)}        {round(report['-1']['f1-score'], 2)}        {round(report['-1']['support'])}")
print(f"       Generic     {round(report['0']['precision'], 2)}        {round(report['0']['recall'], 2)}        {round(report['0']['f1-score'], 2)}       {round(report['0']['support'])}")
print(f"       Fuzzers     {round(report['1']['precision'], 2)}        {round(report['1']['recall'], 2)}        {round(report['1']['f1-score'], 2)}        {round(report['1']['support'])}")
print(f"      Exploits     {round(report['2']['precision'], 2)}        {round(report['2']['recall'], 2)}        {round(report['2']['f1-score'], 2)}       {round(report['2']['support'])}")
print(f"           DoS     {round(report['3']['precision'], 2)}        {round(report['3']['recall'], 2)}        {round(report['3']['f1-score'], 2)}        {round(report['3']['support'])}")
print(f"Reconnaissance     {round(report['4']['precision'], 2)}        {round(report['4']['recall'], 2)}        {round(report['4']['f1-score'], 2)}       {round(report['4']['support'])}")
print(f"      Analysis     {round(report['5']['precision'], 2)}        {round(report['5']['recall'], 2)}        {round(report['5']['f1-score'], 2)}        {round(report['5']['support'])}")
print(f"     Shellcode     {round(report['6']['precision'], 2)}        {round(report['6']['recall'], 2)}        {round(report['6']['f1-score'], 2)}       {round(report['6']['support'])}")
print(f"      Backdoor     {round(report['7']['precision'], 2)}        {round(report['7']['recall'], 2)}        {round(report['7']['f1-score'], 2)}        {round(report['7']['support'])}")
print(f"         Worms     {round(report['8']['precision'], 2)}        {round(report['8']['recall'], 2)}        {round(report['8']['f1-score'], 2)}       {round(report['8']['support'])}")
print(f"     micro avg     {round(precision_micro, 2)}        {round(recall_micro, 2)}        {round(f1_micro, 2)}    {round(report['10']['support'])}")   
print(f"     macro avg     {round(report['macro avg']['precision'], 2)}        {round(report['macro avg']['recall'], 2)}        {round(report['macro avg']['f1-score'], 2)}      {round(report['macro avg']['support'])}") 
print(f"  weighted avg     {round(report['weighted avg']['precision'], 2)}        {round(report['weighted avg']['recall'], 2)}        {round(report['weighted avg']['f1-score'], 2)}     {round(report['weighted avg']['support'])}")
print('\n')


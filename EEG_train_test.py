import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("Processed_Data.csv")
features = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'Attention', 'Mediation']
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['user-definedlabel'])
print(f"Data Split: {len(train_df)} training samples, {len(test_df)} testing samples.")

def preprocessing(): #Processes original dataset and saves the processed data in new file (already made- Processed_Data.csv)
   data=pd.read_csv("data_EEG.csv")
   data=data.drop(['predefinedlabel'],axis=1)
   data[['user-definedlabel','SubjectID','VideoID']]=data[['user-definedlabel','SubjectID','VideoID']].astype(int)
   data.to_csv("Processed_Data.csv")

def train_random_forest(train_df):
    participant_models = {}
    participant_thresholds = {}

    for subject in train_df['SubjectID'].unique():
        sub_df = train_df[train_df['SubjectID'] == subject]
        X = sub_df[features]
        y = sub_df['user-definedlabel']
        
        #Random Forest with 100 trees- better generalization
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X, y)
        participant_models[subject] = rf_model

        #Extracting feature importance-based thresholds (median feature values per class)
        threshold_dict = {}
        for feature in features:
            threshold_dict[feature] = sub_df.groupby('user-definedlabel')[feature].median().to_dict()
        participant_thresholds[subject] = threshold_dict

    return participant_models, participant_thresholds

def predict1(eeg_values, subject_id, participant_models, participant_thresholds):
    # Check if the participant has a trained model
    if subject_id not in participant_models:
        print(f"No trained model available for Subject {subject_id}")
        return None

    model = participant_models[subject_id]

    input_data = pd.DataFrame([eeg_values], columns=features)
    
    # Display thresholds
    print(f"\nCalculated Feature Thresholds for Participant {subject_id}:")

    if subject_id in participant_thresholds:
        for feature, threshold_dict in participant_thresholds[subject_id].items():
            confusion_threshold = threshold_dict.get(1, "N/A")  # Median feature value when confused
            no_confusion_threshold = threshold_dict.get(0, "N/A")  # Median feature value when not confused
            print(f"{feature}: Confusion ≤ {confusion_threshold}, No Confusion ≤ {no_confusion_threshold}")
    else:
        print("Thresholds not available for this participant.")

    try:
        prediction = model.predict(input_data)[0] # Returns 0(Not Confused) or 1(Confused)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
    #Display Final Prediction
    print(f"\nPredicted Confusion State for Participant {subject_id}: {'Confused' if prediction == 1 else 'Not Confused'}")
    return prediction

def predict2(eeg_values, subject_id, participant_models):
    if subject_id not in participant_models:
        return None
    model = participant_models[subject_id]
    input_data = pd.DataFrame([eeg_values], columns=features)
    return model.predict(input_data)[0]

def evaluate_model(test_df, participant_models):
    y_true, y_pred = [], []

    for i in range(len(test_df)):
        eeg_values = test_df.iloc[i][features].to_dict()  
        actual_label = test_df.iloc[i]['user-definedlabel']  
        subject_id = test_df.iloc[i]['SubjectID']  
        predicted_label = predict2(eeg_values, subject_id, participant_models)
        y_true.append(actual_label)
        y_pred.append(predicted_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nModel Evaluation Results (Random Forest):")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    return accuracy, precision, recall, f1
#Driver Code
participant_models, participant_thresholds = train_random_forest(train_df)
#evaluate_model(test_df, participant_models)

#EEG Input examples
#eeg_values={'Delta': 1817666, 'Theta':259465, 'Alpha1':22213,'Alpha2': 71445,'Beta1': 69577,'Beta2': 105431,'Gamma1': 141042, 'Gamma2': 29641,'Attention': 44,'Mediation':56}
#eeg_values={'Delta': 983561, 'Theta':408268, 'Alpha1':73047,'Alpha2': 49538,'Beta1': 14162,'Beta2': 47428,'Gamma1': 60168, 'Gamma2': 43369,'Attention': 43,'Mediation':57}
eeg_values={'Delta': 161098, 'Theta':12119, 'Alpha1':1963,'Alpha2': 809,'Beta1': 1271,'Beta2': 3186,'Gamma1': 3266, 'Gamma2': 2518,'Attention': 40,'Mediation':61}
#eeg_values = {'Delta': 4516, 'Theta': 13275, 'Alpha1': 6048, 'Alpha2': 30937,'Beta1': 12945, 'Beta2': 4998, 'Gamma1': 2338, 'Gamma2': 3378,'Attention': 51, 'Mediation': 57}

# Predict using user input
p=predict1(eeg_values, 0, participant_models, participant_thresholds)
print(p)

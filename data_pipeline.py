import os
import pandas as pd
import numpy as np
import warnings
import re

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("  HepSense V2 - Data Preprocessing Pipeline")
    print("  Processing Clinical Tabular Trajectory Dataset")
    print("=" * 60)

    # 1. Load Datasets
    print("\n[1/5] Loading datasets from NEW DATASET/...")
    cohort = pd.read_csv('NEW DATASET/cirrhosis_cohort.csv')
    labels = pd.read_csv('NEW DATASET/decompensation_labels.csv')
    labs = pd.read_csv('NEW DATASET/labs_cirrhosis.csv')
    notes = pd.read_csv('NEW DATASET/discharge_notes.csv')

    print(f"  Cohort shape:        {cohort.shape}")
    print(f"  Labels shape:        {labels.shape}")
    print(f"  Labs shape:          {labs.shape}")
    print(f"  Discharge notes:     {notes.shape}")

    # 2. Merge Cohort and Labels
    print("\n[2/5] Merging Cohort and Labels...")
    cohort_labels = pd.merge(cohort, labels, on='subject_id', how='inner')
    print(f"  Merged Cohort + Labels: {cohort_labels.shape}")

    # 3. Process Lab Trajectories
    print("\n[3/5] Extracting lab features (max, min, latest, velocity)...")
    # Convert timestamps
    cohort_labels['index_admittime'] = pd.to_datetime(cohort_labels['index_admittime'])
    cohort_labels['index_dischtime'] = pd.to_datetime(cohort_labels['index_dischtime'])
    labs['charttime'] = pd.to_datetime(labs['charttime'])

    # Merge labs with admission windows for time-filtering
    labs_merged = pd.merge(
        labs,
        cohort_labels[['subject_id', 'index_admittime', 'index_dischtime']],
        on='subject_id',
        how='inner'
    )

    # Filter labs that are strictly within the index admission window
    labs_in_window = labs_merged[
        (labs_merged['charttime'] >= labs_merged['index_admittime']) &
        (labs_merged['charttime'] <= labs_merged['index_dischtime'])
    ].copy()

    print(f"  Filtered lab measurements within admission windows: {len(labs_in_window)}")

    # Sort labs by charttime for sequential calculations
    labs_in_window = labs_in_window.sort_values(by=['subject_id', 'lab_test_name', 'charttime'])

    # Pivot / Group to calculate max, min, latest, velocity
    # We will compute these metrics for each subject and lab test name
    lab_features = {}
    
    grouped = labs_in_window.groupby(['subject_id', 'lab_test_name'])
    
    # Pre-allocate dictionary for faster feature building
    for (subject_id, test_name), group in grouped:
        if subject_id not in lab_features:
            lab_features[subject_id] = {}
        
        values = group['valuenum'].values
        times = group['charttime']
        
        v_min = np.min(values)
        v_max = np.max(values)
        v_latest = values[-1]
        
        # Calculate velocity: rate of change per day
        if len(values) > 1:
            time_diff_days = (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400.0
            if time_diff_days > 0:
                v_velocity = (values[-1] - values[0]) / time_diff_days
            else:
                v_velocity = 0.0
        else:
            v_velocity = 0.0
            
        test_key = test_name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        lab_features[subject_id][f"{test_key}_min"] = v_min
        lab_features[subject_id][f"{test_key}_max"] = v_max
        lab_features[subject_id][f"{test_key}_latest"] = v_latest
        lab_features[subject_id][f"{test_key}_velocity"] = v_velocity

    # Convert to DataFrame
    df_lab_features = pd.DataFrame.from_dict(lab_features, orient='index')
    df_lab_features.index.name = 'subject_id'
    df_lab_features = df_lab_features.reset_index()
    print(f"  Extracted lab features for {len(df_lab_features)} patients.")

    # 4. Discharge Notes NLP
    print("\n[4/5] Extracting clinical mentions from discharge notes...")
    notes['text_lower'] = notes['text'].str.lower().fillna('')
    
    # Regex compilation for faster matching
    re_enceph = re.compile(r'encephalopathy|hepatic coma|asterixis|confusion', re.IGNORECASE)
    re_ascites = re.compile(r'ascites|ascitic|paracentesis', re.IGNORECASE)
    re_varices = re.compile(r'variceal bleed|varices bleed|esophageal varices bleeding|gastric varices bleeding|hematemesis|melena|varices|variceal', re.IGNORECASE)

    notes['has_encephalopathy_mention'] = notes['text_lower'].apply(lambda x: 1 if re_enceph.search(x) else 0)
    notes['has_ascites_mention'] = notes['text_lower'].apply(lambda x: 1 if re_ascites.search(x) else 0)
    notes['has_variceal_bleeding_mention'] = notes['text_lower'].apply(lambda x: 1 if re_varices.search(x) else 0)

    # Aggregate notes to subject_id + hadm_id level
    notes_agg = notes.groupby(['subject_id', 'hadm_id'])[[
        'has_encephalopathy_mention',
        'has_ascites_mention',
        'has_variceal_bleeding_mention'
    ]].max().reset_index()

    # 5. Merge Everything to Create Training Matrix
    print("\n[5/5] Merging all features...")
    # Start with cohort + labels
    final_matrix = cohort_labels.copy()
    
    # Merge lab features (inner join to drop patients with no lab measurements in their window)
    final_matrix = pd.merge(final_matrix, df_lab_features, on='subject_id', how='inner')
    
    # Merge NLP notes flags on subject_id and index_hadm_id
    final_matrix = pd.merge(
        final_matrix,
        notes_agg.rename(columns={'hadm_id': 'index_hadm_id'}),
        on=['subject_id', 'index_hadm_id'],
        how='left'
    )

    # Fill NaN for notes flags (meaning no discharge notes matched this admission)
    nlp_cols = ['has_encephalopathy_mention', 'has_ascites_mention', 'has_variceal_bleeding_mention']
    final_matrix[nlp_cols] = final_matrix[nlp_cols].fillna(0).astype(int)

    # Encode Gender (M = 1, F = 0)
    final_matrix['gender'] = final_matrix['gender'].map({'M': 1, 'F': 0}).fillna(0).astype(int)

    # Save training matrix
    output_path = 'hepsense_training_matrix.csv'
    final_matrix.to_csv(output_path, index=False)
    print(f"\n[OK] Final HepSense training matrix saved to: {output_path}")
    print(f"     Matrix dimensions: {final_matrix.shape}")
    print(f"     Decompensation rate: {final_matrix['decompensation_90day'].mean() * 100:.2f}%")

if __name__ == "__main__":
    main()

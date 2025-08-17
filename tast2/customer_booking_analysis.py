# Customer booking prediction: EDA, feature engineering, RandomForest training and evaluation
# Usage: run this script from repository root. It will search for a CSV dataset and run the whole pipeline.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer

sns.set(style='whitegrid')

# 1. Locate dataset (CSV) in repository

def find_candidate_csvs(root='.'):
    csvs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(dirpath, f))
    return csvs


def load_best_csv():
    csvs = find_candidate_csvs('.')
    if not csvs:
        raise FileNotFoundError('No CSV files found in repository. Please place your dataset (CSV) in the repo and re-run.')

    # Prefer files referenced in the Getting Started notebook path (tast2) or with 'booking' in name
    preferred = [p for p in csvs if 'tast2' in p or 'booking' in os.path.basename(p).lower()]
    candidates = preferred + [p for p in csvs if p not in preferred]

    for path in candidates:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        # check for booking target columns
        cols = set([c.lower() for c in df.columns])
        if 'booking_complete' in cols or 'booking' in cols or 'booking_complete' in cols:
            print(f"Selected dataset: {path}")
            return df, path

    # fallback: return first CSV
    print(f"No CSV with expected target name found. Using first CSV: {candidates[0]}")
    return pd.read_csv(candidates[0], low_memory=False), candidates[0]


if __name__ == '__main__':
    df, data_path = load_best_csv()

    out_dir = 'tast2/model_outputs'
    os.makedirs(out_dir, exist_ok=True)

    # 2. Basic EDA
    print('\n=== Basic info ===')
    print('path:', data_path)
    print('shape:', df.shape)
    print('\ncolumns:', df.columns.tolist())
    print('\nmissing values:')
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    # display simple statistics to a csv
    desc = df.describe(include='all')
    desc.to_csv(os.path.join(out_dir, 'dataset_description.csv'))

    # 3. Clean column names
    df.columns = [c.strip() for c in df.columns]

    # 4. Identify target
    target_candidates = [c for c in df.columns if c.lower() in ('booking_complete', 'booking')]
    if not target_candidates:
        # try partial match
        for c in df.columns:
            if 'book' in c.lower():
                target_candidates.append(c)
    if not target_candidates:
        raise ValueError('Could not find a target column (booking or booking_complete). Please rename the target column or update the script.')

    target = target_candidates[0]
    print('Using target column:', target)

    # 5. Basic cleaning and feature engineering
    data = df.copy()

    # Convert boolean-like columns
    for col in data.columns:
        if data[col].dropna().isin([0,1]).all():
            data[col] = data[col].astype('Int64')

    # Create route origin/dest if route like 'AKLDEL'
    if 'route' in data.columns:
        try:
            data['route'] = data['route'].astype(str)
            data['route_origin'] = data['route'].str[:3]
            data['route_dest'] = data['route'].str[-3:]
        except Exception:
            pass

    # flight_day -> is_weekend
    if 'flight_day' in data.columns:
        data['is_weekend'] = data['flight_day'].astype(str).str.lower().isin(['sat','sun','saturday','sunday']).astype(int)

    # flight_hour -> hour_bin
    if 'flight_hour' in data.columns:
        def hour_bin(x):
            try:
                x = int(x)
            except Exception:
                return 'unknown'
            if x >= 5 and x < 12:
                return 'morning'
            if x >= 12 and x < 17:
                return 'afternoon'
            if x >= 17 and x < 21:
                return 'evening'
            return 'night'
        data['flight_hour_bin'] = data['flight_hour'].apply(hour_bin)

    # Purchase lead bins
    if 'purchase_lead' in data.columns:
        data['lead_bin'] = pd.cut(data['purchase_lead'].fillna(-1), bins=[-1,7,30,90,365,99999], labels=['missing_or_<7','7-30','30-90','90-365','>365'])

    # passengers bins
    if 'num_passengers' in data.columns:
        data['group_size'] = pd.cut(data['num_passengers'].fillna(1), bins=[0,1,2,4,10], labels=['solo','couple','small_group','large_group'])

    # length_of_stay bin
    if 'length_of_stay' in data.columns:
        data['stay_bin'] = pd.cut(data['length_of_stay'].fillna(0), bins=[-1,1,3,7,30,9999], labels=['same_day','short','week','month','long'])

    # Target cleaning
    y = data[target].copy()
    # convert strings like 'Yes'/'No'
    if y.dtype == 'object' or y.dtype.name.startswith('str'):
        y = y.str.lower().map({'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0})
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    data = data.drop(columns=[target])

    # Select features: drop high-cardinality columns that are identifiers
    drop_cols = []
    for c in data.columns:
        if data[c].nunique() == data.shape[0] and data[c].dtype == object:
            drop_cols.append(c)
    # drop obvious identifiers
    for c in ['booking_id','id','session_id']:
        if c in data.columns:
            drop_cols.append(c)

    data.drop(columns=list(set(drop_cols)), inplace=True, errors='ignore')

    # Keep only reasonable column types
    # Fill missing numeric with median, categorical with 'missing'
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object','category']).columns.tolist()

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    if len(num_cols) > 0:
        data[num_cols] = num_imputer.fit_transform(data[num_cols])
    for c in cat_cols:
        data[c] = data[c].fillna('missing').astype(str)

    # One-hot encode categorical features using pandas.get_dummies
    data_enc = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Align X and y
    X = data_enc.copy()
    print('Final feature matrix shape:', X.shape)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 7. Model training
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    # 8. Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print('\nPerforming cross-validation (5-fold stratified) ...')
    acc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    try:
        auc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    except Exception:
        auc_scores = np.array([np.nan]*len(acc_scores))

    print('CV accuracy: mean=%.4f std=%.4f' % (acc_scores.mean(), acc_scores.std()))
    print('CV ROC AUC: mean=%.4f std=%.4f' % (np.nanmean(auc_scores), np.nanstd(auc_scores)))

    # 9. Test set evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, 'predict_proba') else None

    print('\n=== Test set evaluation ===')
    print(classification_report(y_test, y_pred))
    if y_proba is not None and len(np.unique(y_test))==2:
        print('Test ROC AUC:', roc_auc_score(y_test, y_proba))

    # save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC curve
    if y_proba is not None and len(np.unique(y_test))==2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_proba):.3f}')
        plt.plot([0,1],[0,1],'--',color='grey')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
        plt.close()

    # 10. Feature importances
    fi = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_})
    fi.sort_values('importance', ascending=False, inplace=True)
    fi.to_csv(os.path.join(out_dir, 'feature_importances.csv'), index=False)

    # Plot top 20
    plt.figure(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=fi.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importances_top20.png'))
    plt.close()

    # 11. Persist model (optional) - use joblib
    try:
        import joblib
        joblib.dump(clf, os.path.join(out_dir, 'random_forest_model.joblib'))
    except Exception:
        pass

    # 12. Save summary metrics to a markdown file
    summary = []
    summary.append('# Model run summary\n')
    summary.append('Data path: %s\n' % data_path)
    summary.append('Number of rows: %d\n' % df.shape[0])
    summary.append('Number of features after encoding: %d\n' % X.shape[1])
    summary.append('\n## Cross-validation results (train set)\n')
    summary.append('- CV accuracy: mean=%.4f std=%.4f\n' % (acc_scores.mean(), acc_scores.std()))
    if not np.isnan(auc_scores).all():
        summary.append('- CV ROC AUC: mean=%.4f std=%.4f\n' % (np.nanmean(auc_scores), np.nanstd(auc_scores)))

    summary.append('\n## Test set results\n')
    summary.append('```
%s
```
' % (classification_report(y_test, y_pred)))
    if y_proba is not None and len(np.unique(y_test))==2:
        summary.append('- Test ROC AUC: %.4f\n' % roc_auc_score(y_test, y_proba))

    # top features
    summary.append('\n## Top features\n')
    for i, row in fi.head(20).iterrows():
        summary.append('- %s: %.6f\n' % (row['feature'], row['importance']))

    summary_path = os.path.join(out_dir, 'MODEL_RESULTS.md')
    with open(summary_path, 'w') as f:
        f.writelines(summary)

    print('\nAll outputs written to', out_dir)
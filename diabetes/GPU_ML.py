import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 設定使用 Agg 非互動式後端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, RFECV
import os
import time

# 建立輸出目錄
output_dir = 'diabetes/gpu_enhanced_models'
os.makedirs(output_dir, exist_ok=True)

# 設定使用微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 如果還是有問題，可以嘗試這樣設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def plot_confusion_matrix(cm, title='Confusion Matrix', save_path=None):
    """
    繪製混淆矩陣
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩陣已保存至: {save_path}")
    plt.close()

def load_and_preprocess_data(file_path):
    """
    載入糖尿病資料集並進行增強預處理
    """
    print(f"載入資料集: {file_path}")
    start_time = time.time()
    df = pd.read_csv(file_path)
    print(f"資料集形狀: {df.shape}")
    print(f"載入時間: {time.time() - start_time:.2f} 秒")
    
    # 檢查資料缺失值
    missing_values = df.isnull().sum()
    print(f"缺失值統計:")
    print(missing_values)
    
    # 增強特徵工程
    start_time = time.time()
    print("開始特徵工程...")
    
    # 1. BMI類別 (更細緻的分類)
    df['BMI_Category'] = pd.cut(
        df['BMI'], 
        bins=[0, 18.5, 23, 25, 27.5, 30, 35, 40, 100], 
        labels=[0, 1, 2, 3, 4, 5, 6, 7]  # 更細緻的BMI分類
    ).astype(int)
    
    # 2. 年齡分組
    df['Age_Group'] = pd.cut(
        df['Age'], 
        bins=[0, 3, 6, 9, 12, 13], 
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # 3. 身體健康綜合指標 (改進計算方式)
    df['Health_Score'] = df['GenHlth'] + df['PhysHlth']/30 + df['MentHlth']/30
    
    # 4. 風險因子綜合指標 (加入BMI權重)
    df['Risk_Score'] = df['HighBP']*1.5 + df['HighChol']*1.3 + df['Stroke']*1.7 + df['HeartDiseaseorAttack']*1.8
    
    # 5. 生活方式綜合指標 (調整權重)
    df['Lifestyle_Score'] = df['PhysActivity']*1.5 + df['Fruits'] + df['Veggies'] - df['Smoker']*1.8 - df['HvyAlcoholConsump']*1.2
    
    # 6. BMI與高血壓交互作用
    df['BMI_BP_Interaction'] = df['BMI'] * df['HighBP']
    
    # 7. 年齡與健康狀況交互作用
    df['Age_Health_Interaction'] = df['Age'] * df['GenHlth']
    
    # 8. 風險因子總數
    df['Total_Risk_Factors'] = df['HighBP'] + df['HighChol'] + df['Stroke'] + df['HeartDiseaseorAttack'] + df['Smoker']
    
    # 9. 健康行為總數
    df['Total_Healthy_Behaviors'] = df['PhysActivity'] + df['Fruits'] + df['Veggies'] + (1 - df['Smoker']) + (1 - df['HvyAlcoholConsump'])
    
    # 10. BMI平方項 (捕捉非線性關係)
    df['BMI_Squared'] = df['BMI'] ** 2
    
    # 11. 年齡與BMI交互作用
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    
    # 確保所有特徵都是數值型別
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(int)
            print(f"將 {col} 從類別型別轉換為整數型別")
    
    print(f"特徵工程時間: {time.time() - start_time:.2f} 秒")
    
    # 特徵相關性
    correlation = df.corr()['Diabetes_binary'].sort_values(ascending=False)
    print("\n與目標變數的相關性:")
    print(correlation.head(10))  # 只顯示前10個相關性最高的
    
    return df

def enhanced_feature_selection(X, y):
    """
    使用進階特徵選擇方法 - 高效率版本
    """
    print("\n執行進階特徵選擇...")
    start_time = time.time()
    
    # 使用輕量級隨機森林進行初步特徵重要性評估
    base_selector = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    base_selector.fit(X, y)
    
    # 獲取特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': base_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性排名:")
    print(feature_importance.head(15))  # 只顯示前15個
    
    # 使用特徵重要性先篩選前50%的特徵，提高RFECV效率
    # 但至少保留15個特徵，最多保留30個特徵
    preselect_count = min(max(15, X.shape[1] // 2), 30)
    preselected_features = feature_importance.head(preselect_count)['feature'].tolist()
    X_preselected = X[preselected_features]
    
    print(f"\n預先篩選出的特徵數量: {len(preselected_features)}")
    
    # 在預先篩選的特徵上使用RFECV，提高步長以加速收斂
    print("\n使用RFECV進行特徵選擇...")
    # 使用更輕量的隨機森林分類器
    lighter_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    
    rfecv = RFECV(
        estimator=lighter_rf,
        step=2,  # 增加步長以加速
        cv=StratifiedKFold(3, shuffle=True, random_state=42),  # 減少交叉驗證折數
        scoring='accuracy',
        min_features_to_select=5,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=0
    )
    
    # 在預選特徵上執行RFECV
    rfecv.fit(X_preselected, y)
    
    # 獲取RFECV選定的特徵索引
    final_indices = rfecv.support_
    final_feature_names = np.array(preselected_features)[final_indices].tolist()
    
    print(f"最佳特徵數量: {len(final_feature_names)}")
    print(f"RFECV選定的特徵: {final_feature_names}")
    
    # 繪製RFECV選擇結果
    if hasattr(rfecv, 'cv_results_'):
        cv_scores = rfecv.cv_results_['mean_test_score']
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cv_scores) + 1), cv_scores)
        plt.xlabel('特徵數量')
        plt.ylabel('交叉驗證分數')
        plt.title('RFECV特徵選擇結果')
        plt.savefig(f"{output_dir}/rfecv_feature_selection.png")
        plt.close()
    elif hasattr(rfecv, 'grid_scores_'):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.xlabel('特徵數量')
        plt.ylabel('交叉驗證分數')
        plt.title('RFECV特徵選擇結果')
        plt.savefig(f"{output_dir}/rfecv_feature_selection.png")
        plt.close()
    
    # 繪製選定特徵的重要性
    selected_importance = feature_importance[feature_importance['feature'].isin(final_feature_names)]
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=selected_importance)
    plt.title('選定特徵的重要性')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/selected_feature_importance.png")
    plt.close()
    
    print(f"特徵選擇時間: {time.time() - start_time:.2f} 秒")
    
    # 如果RFECV選擇的特徵太少，則從特徵重要性中選擇前N個特徵
    if len(final_feature_names) < 8:
        print("RFECV選擇的特徵太少，改為從特徵重要性中選擇...")
        top_features = feature_importance.head(12)['feature'].tolist()
        print(f"從特徵重要性中選擇的前12個特徵: {top_features}")
        return top_features, feature_importance
    
    return final_feature_names, feature_importance

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """
    訓練XGBoost模型
    """
    print("\n訓練XGBoost模型...")
    
    # XGBoost參數
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  # 使用hist算法
        'random_state': 42
    }
    
    # 創建模型
    xgb_model = xgb.XGBClassifier(**params)
    
    start_time = time.time()
    
    # 將訓練數據和測試數據轉換為DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 準備watchlist用於評估
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # 轉換參數至xgb.train的格式
    train_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.05,  # 等同於learning_rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'seed': 42
    }
    
    # 使用原生API訓練，支持早停
    bst = xgb.train(
        train_params,
        dtrain,
        num_boost_round=300,
        evals=watchlist,
        early_stopping_rounds=20,
        verbose_eval=True
    )
    
    # 將訓練後的模型保存到XGBClassifier
    xgb_model._Booster = bst
    
    train_time = time.time() - start_time
    print(f"XGBoost訓練時間: {train_time:.2f} 秒")
    
    # 預測
    y_pred = (xgb_model.predict(X_test) > 0.5).astype(int)
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nXGBoost模型效能:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    
    return xgb_model, accuracy, train_time

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    訓練多個模型並評估效能
    """
    print("\n訓練和評估多個模型...")
    
    # 準備縮放器
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 轉換回DataFrame以保留欄位名稱
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 定義模型 (移除SVM)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8,
            objective='binary:logistic', 
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
    }
    
    # 用於儲存結果
    results = {}
    
    # 訓練和評估每個模型
    for name, model in models.items():
        print(f"\n訓練 {name} 模型...")
        start_time = time.time()
        
        # 選擇適合的資料預處理方式
        if name == 'Logistic Regression':
            X_train_model = X_train_scaled_df  # 對邏輯迴歸使用標準化後的資料
            X_test_model = X_test_scaled_df
        else:
            X_train_model = X_train  # 其他模型使用原始特徵
            X_test_model = X_test
        
        # 使用交叉驗證評估模型
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"交叉驗證準確率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # 在整個訓練集上訓練
        model.fit(X_train_model, y_train)
        
        # 預測
        y_pred = model.predict(X_test_model)
        
        # 計算各種指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        train_time = time.time() - start_time
        
        print(f"{name} 效能:")
        print(f"準確率: {accuracy:.4f}")
        print(f"精確率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分數: {f1:.4f}")
        print(f"訓練時間: {train_time:.2f} 秒")
        
        # 繪製混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} 混淆矩陣')
        plt.xlabel('預測標籤')
        plt.ylabel('真實標籤')
        plt.savefig(f"{output_dir}/{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()
        
        # 如果模型支援probability，則繪製ROC曲線
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_model)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, lw=2, label=f'ROC曲線 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假陽性率')
            plt.ylabel('真陽性率')
            plt.title(f'{name} ROC曲線')
            plt.legend(loc="lower right")
            plt.savefig(f"{output_dir}/{name.replace(' ', '_')}_roc_curve.png")
            plt.close()
        
        # 儲存結果
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model,
            'confusion_matrix': cm,
            'train_time': train_time,
            'cv_score': np.mean(cv_scores)
        }
        
        if 'roc_auc' in locals():
            results[name]['roc_auc'] = roc_auc
    
    # 比較模型效能
    model_comparison = pd.DataFrame({
        '模型': list(results.keys()),
        '準確率': [results[model]['accuracy'] for model in results],
        '精確率': [results[model]['precision'] for model in results],
        '召回率': [results[model]['recall'] for model in results],
        'F1分數': [results[model]['f1'] for model in results],
        '訓練時間(秒)': [results[model]['train_time'] for model in results],
        '交叉驗證準確率': [results[model]['cv_score'] for model in results]
    })
    
    model_comparison = model_comparison.sort_values('準確率', ascending=False)
    print("\n模型效能比較:")
    print(model_comparison)
    
    # 儲存比較結果
    model_comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # 繪製模型比較圖
    plt.figure(figsize=(14, 10))
    
    # 創建子圖: 評估指標比較
    plt.subplot(2, 1, 1)
    model_comparison_melt = pd.melt(model_comparison, id_vars=['模型'], 
                                   value_vars=['準確率', '精確率', '召回率', 'F1分數'],
                                   var_name='指標', value_name='分數')
    
    sns.barplot(x='模型', y='分數', hue='指標', data=model_comparison_melt)
    plt.title('模型評估指標比較')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 創建子圖: 訓練時間比較
    plt.subplot(2, 1, 2)
    sns.barplot(x='模型', y='訓練時間(秒)', data=model_comparison)
    plt.title('模型訓練時間比較')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()
    
    # 找出最佳模型
    best_model_name = model_comparison.iloc[0]['模型']
    print(f"\n最佳模型: {best_model_name}")
    
    return results, best_model_name

def create_voting_ensemble(X_train, X_test, y_train, y_test, models_results):
    """
    創建加權投票集成模型
    """
    print("\n創建加權Voting集成模型...")
    
    # 選擇表現最好的模型
    model_names = list(models_results.keys())
    model_accuracies = [models_results[name]['accuracy'] for name in model_names]
    
    # 按準確率從高到低排序模型
    sorted_indices = np.argsort(model_accuracies)[::-1]
    sorted_models = [model_names[i] for i in sorted_indices]
    
    top_models = sorted_models[:3]  # 使用前3個模型
    print(f"使用以下模型創建加權投票集成: {top_models}")
    
    # 創建基礎分類器列表
    estimators = []
    for name in top_models:
        estimators.append((name.replace(' ', '_'), models_results[name]['model']))
    
    # 根據準確率分配權重
    weights = []
    for name in top_models:
        weights.append(models_results[name]['accuracy'])
    
    # 標準化權重
    weights = np.array(weights) / sum(weights)
    print(f"模型權重: {dict(zip(top_models, weights))}")
    
    # 創建加權投票分類器
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1  # 使用所有CPU核心
    )
    
    start_time = time.time()
    
    # 訓練加權投票模型
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    train_time = time.time() - start_time
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n加權Voting集成模型效能:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    print(f"訓練時間: {train_time:.2f} 秒")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('加權投票集成模型混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.savefig(f"{output_dir}/weighted_voting_confusion_matrix.png")
    plt.close()
    
    # 繪製ROC曲線
    if hasattr(voting_clf, 'predict_proba'):
        y_prob = voting_clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC曲線 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假陽性率')
        plt.ylabel('真陽性率')
        plt.title('加權投票集成模型ROC曲線')
        plt.legend(loc="lower right")
        plt.savefig(f"{output_dir}/weighted_voting_roc_curve.png")
        plt.close()
    
    # 儲存加權Voting模型
    import pickle
    with open(f"{output_dir}/weighted_voting_model.pkl", 'wb') as f:
        pickle.dump(voting_clf, f)
    
    print(f"\n加權Voting集成模型已儲存: {output_dir}/weighted_voting_model.pkl")
    
    return voting_clf, accuracy

def main():
    """
    主執行函數
    """
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入和預處理資料
    file_path = 'diabetes\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    df = load_and_preprocess_data(file_path)
    
    # 分離特徵和目標變數
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # 進行特徵選擇
    selected_features, feature_importance = enhanced_feature_selection(X, y)
    X_selected = X[selected_features]
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
    
    # 訓練XGBoost模型
    xgb_model, xgb_acc, xgb_train_time = train_xgboost_model(X_train, X_test, y_train, y_test)
    
    # 訓練和評估其他模型
    models_results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 創建加權投票集成模型
    voting_clf, voting_acc = create_voting_ensemble(X_train, X_test, y_train, y_test, models_results)
    
    # 保存最佳模型
    best_model = models_results[best_model_name]['model']
    import pickle
    with open(f"{output_dir}/best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    # 保存XGBoost模型
    xgb_model.save_model(f"{output_dir}/xgboost_model.json")
    
    # 輸出結果比較
    print("\n===== 結果比較 =====")
    print(f"最佳單一模型 ({best_model_name}) 準確率: {models_results[best_model_name]['accuracy']:.4f}")
    print(f"XGBoost模型準確率: {xgb_acc:.4f}")
    print(f"集成模型準確率: {voting_acc:.4f}")
    
    # 繪製最終結果比較圖
    models_final = ['XGBoost', f'最佳: {best_model_name}', '集成模型']
    accs_final = [xgb_acc, models_results[best_model_name]['accuracy'], voting_acc]
    
    # 在main函數中，修改最終比較圖的標題
    plt.figure(figsize=(10, 6))
    plt.bar(models_final, accs_final, color=['blue', 'green', 'red'])
    plt.title('最終模型準確率比較')
    plt.ylim(min(accs_final)-0.01, max(accs_final)+0.01)
    
    # 添加數據標籤
    for i, v in enumerate(accs_final):
        plt.text(i, v+0.001, f"{v:.4f}", ha='center')
    
    plt.savefig(f"{output_dir}/final_model_comparison.png")
    plt.close('all')
    
    print("\n模型訓練與評估完成!")
    print(f"所有模型和圖表已保存到 {output_dir}")

if __name__ == "__main__":
    main()
    plt.close('all')
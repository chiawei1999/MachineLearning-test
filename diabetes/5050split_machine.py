# 匯入所需的套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import os

# 設定中文字型以正確顯示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 建立輸出目錄
output_dir = 'diabetes/optimized_models'
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    載入糖尿病資料集並進行預處理
    """
    print(f"載入資料集: {file_path}")
    df = pd.read_csv(file_path)
    print(f"資料集形狀: {df.shape}")
    print(f"資料集欄位名稱: {df.columns.tolist()}")
    print(f"目標變數分佈:\n{df['Diabetes_binary'].value_counts()}")
    
    # 檢查資料缺失值
    missing_values = df.isnull().sum()
    print(f"缺失值統計:")
    print(missing_values)
    
    # 檢查基本統計
    print("\n基本統計資訊:")
    print(df.describe())
    
    # 特徵工程
    # 1. 新增BMI類別 (使用整數型別而非類別型別)
    df['BMI_Category'] = pd.cut(
        df['BMI'], 
        bins=[0, 18.5, 25, 30, 100], 
        labels=[0, 1, 2, 3]  # 0: 過輕, 1: 正常, 2: 過重, 3: 肥胖
    ).astype(int)  # 轉換為整數，避免類別型態
    
    # 2. 新增年齡分組 (使用整數型別而非類別型別)
    df['Age_Group'] = pd.cut(
        df['Age'], 
        bins=[0, 3, 6, 9, 12, 13], 
        labels=[0, 1, 2, 3, 4]  # 0: 18-29, 1: 30-49, 2: 50-69, 3: 70+, 4: 80+
    ).astype(int)  # 轉換為整數，避免類別型態
    
    # 3. 身體狀況綜合指標
    df['Health_Score'] = df['GenHlth'] + df['PhysHlth']/30 + df['MentHlth']/30
    
    # 4. 風險因子綜合指標
    df['Risk_Score'] = df['HighBP'] + df['HighChol'] + df['Stroke'] + df['HeartDiseaseorAttack']
    
    # 5. 生活方式綜合指標
    df['Lifestyle_Score'] = df['PhysActivity'] + df['Fruits'] + df['Veggies'] - df['Smoker'] - df['HvyAlcoholConsump']
    
    # 檢查資料型態
    print("\n資料型態檢查:")
    print(df.dtypes)
    
    # 確保所有特徵都是數值型別
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(int)
            print(f"將 {col} 從類別型別轉換為整數型別")
    
    # 特徵相關性
    correlation = df.corr()['Diabetes_binary'].sort_values(ascending=False)
    print("\n與目標變數的相關性:")
    print(correlation)
    
    return df

def feature_selection(X, y):
    """
    使用模型基礎的特徵選擇
    """
    # 使用隨機森林進行特徵重要性評估
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X, y)
    
    # 獲取特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性排名:")
    print(feature_importance)
    
    # 繪製特徵重要性圖
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('前15個重要特徵')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    
    # 選擇重要特徵 (重要性前80%)
    cumulative_importance = feature_importance['importance'].cumsum()
    importance_threshold = 0.80  # 累積重要性閾值
    threshold_idx = (cumulative_importance >= importance_threshold).idxmax()
    
    selected_features = feature_importance.iloc[:threshold_idx+1]['feature'].tolist()
    print(f"\n選擇了 {len(selected_features)} 個特徵，累積重要性: {cumulative_importance.iloc[threshold_idx]:.2%}")
    print(f"選定的特徵: {selected_features}")
    
    return selected_features, feature_importance

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    訓練多個模型並評估效能
    """
    # 準備縮放器
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 轉換回DataFrame以保留欄位名稱
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 定義模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', random_state=42, enable_categorical=True),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # 用於儲存結果
    results = {}
    
    # 訓練和評估每個模型
    for name, model in models.items():
        print(f"\n訓練 {name} 模型...")
        
        # SVM需要縮放，其他模型可以使用原始特徵
        if name == 'SVM':
            X_train_model = X_train_scaled_df
            X_test_model = X_test_scaled_df
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # 在整個訓練集上訓練
        model.fit(X_train_model, y_train)
        
        # 預測
        y_pred = model.predict(X_test_model)
        
        # 計算各種指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name} 效能:")
        print(f"準確率: {accuracy:.4f}")
        print(f"精確率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分數: {f1:.4f}")
        
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
            plt.plot(fpr, tpr, lw=2, label=f'ROC 曲線 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('偽陽性率')
            plt.ylabel('真陽性率')
            plt.title(f'{name} ROC 曲線')
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
            'confusion_matrix': cm
        }
        
        if 'roc_auc' in locals():
            results[name]['roc_auc'] = roc_auc
    
    # 比較模型效能
    model_comparison = pd.DataFrame({
        '模型': list(results.keys()),
        '準確率': [results[model]['accuracy'] for model in results],
        '精確率': [results[model]['precision'] for model in results],
        '召回率': [results[model]['recall'] for model in results],
        'F1分數': [results[model]['f1'] for model in results]
    })
    
    model_comparison = model_comparison.sort_values('準確率', ascending=False)
    print("\n模型效能比較:")
    print(model_comparison)
    
    # 儲存比較結果
    model_comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # 繪製模型比較圖
    plt.figure(figsize=(12, 8))
    model_comparison_melt = pd.melt(model_comparison, id_vars=['模型'], 
                                    value_vars=['準確率', '精確率', '召回率', 'F1分數'],
                                    var_name='指標', value_name='分數')
    
    sns.barplot(x='模型', y='分數', hue='指標', data=model_comparison_melt)
    plt.title('模型效能比較')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.close()
    
    # 找出最佳模型
    best_model_name = model_comparison.iloc[0]['模型']
    print(f"\n最佳模型: {best_model_name}")
    
    return results, best_model_name

def optimize_best_model(X_train, X_test, y_train, y_test, best_model_name, models_results):
    """
    針對最佳模型進行超參數調整
    """
    print(f"\n為 {best_model_name} 調整超參數...")
    
    # 根據最佳模型定義參數網格
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', None],  # 簡化網格以加速訓練
            'solver': ['lbfgs', 'newton-cg'],
            'class_weight': [None, 'balanced']
        }
        model = LogisticRegression(max_iter=2000, random_state=42)
        
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
        model = RandomForestClassifier(random_state=42)
        
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
        
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, enable_categorical=True)
        
    elif best_model_name == 'SVM':
        # SVM需要縮放資料
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 'scale'],
            'kernel': ['rbf', 'linear'],
            'class_weight': [None, 'balanced']
        }
        model = SVC(probability=True, random_state=42)
    
    # 使用網格搜索進行超參數調整
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n最佳超參數:")
    print(grid_search.best_params_)
    
    # 使用最佳參數進行評估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n最佳 {best_model_name} 模型效能:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    
    # 比較調整前後
    before_accuracy = models_results[best_model_name]['accuracy']
    improvement = (accuracy - before_accuracy) / before_accuracy * 100
    
    print(f"\n調整前後比較:")
    print(f"調整前準確率: {before_accuracy:.4f}")
    print(f"調整後準確率: {accuracy:.4f}")
    print(f"改善幅度: {improvement:.2f}%")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'最佳 {best_model_name} 混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.savefig(f"{output_dir}/best_{best_model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    # 儲存最佳模型
    import pickle
    with open(f"{output_dir}/best_{best_model_name.replace(' ', '_')}_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\n最佳模型已儲存: {output_dir}/best_{best_model_name.replace(' ', '_')}_model.pkl")
    
    return best_model

def create_ensemble_model(X_train, X_test, y_train, y_test, models_results):
    """
    創建集成模型
    """
    print("\n創建集成模型...")
    
    # 選擇表現最好的三個模型
    model_names = list(models_results.keys())
    model_accuracies = [models_results[name]['accuracy'] for name in model_names]
    top_indices = np.argsort(model_accuracies)[-3:]  # 取最高的三個
    
    top_models = [model_names[i] for i in top_indices]
    print(f"使用以下模型創建集成: {top_models}")
    
    # 創建投票分類器
    estimators = []
    for name in top_models:
        estimators.append((name.replace(' ', '_'), models_results[name]['model']))
    
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    
    # 訓練集成模型
    voting_clf.fit(X_train, y_train)
    
    # 評估集成模型
    y_pred = voting_clf.predict(X_test)
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n集成模型效能:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('集成模型混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.savefig(f"{output_dir}/ensemble_confusion_matrix.png")
    plt.close()
    
    # 儲存集成模型
    import pickle
    with open(f"{output_dir}/ensemble_model.pkl", 'wb') as f:
        pickle.dump(voting_clf, f)
    
    print(f"\n集成模型已儲存: {output_dir}/ensemble_model.pkl")
    
    return voting_clf

def main():
    # 載入和預處理資料
    file_path = 'diabetes\diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    df = load_and_preprocess_data(file_path)
    
    # 分離特徵和目標變數
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # 特徵選擇
    selected_features, feature_importance = feature_selection(X, y)
    X_selected = X[selected_features]
    
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
    
    # 訓練和評估多個模型
    models_results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 對最佳模型進行超參數調整
    best_model = optimize_best_model(X_train, X_test, y_train, y_test, best_model_name, models_results)
    
    # 創建集成模型
    ensemble_model = create_ensemble_model(X_train, X_test, y_train, y_test, models_results)
    
    print("\n模型訓練和評估完成！")
    print(f"所有模型和圖表已儲存在 {output_dir} 目錄中")

if __name__ == "__main__":
    main()
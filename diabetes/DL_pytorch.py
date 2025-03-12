import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # 設定使用 Agg 非互動式後端
import matplotlib.pyplot as plt
import seaborn as sns

# 設定使用微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from imblearn.over_sampling import SMOTE

# 建立輸出目錄
output_dir = 'diabetes/improved_pytorch_models'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子以確保可重複性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 檢查是否可用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 自訂Dataset類別
class DiabetesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 定義改進版神經網絡模型 - 添加殘差連接和進階激活函數
class ImprovedDiabetesNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.25):
        super(ImprovedDiabetesNet, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 創建隱藏層
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # 為每對相鄰的隱藏層添加一個層和殘差連接
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            
            # 殘差連接需要調整輸入維度到輸出維度
            if hidden_sizes[i] != hidden_sizes[i+1]:
                self.skip_connections.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False))
            else:
                self.skip_connections.append(nn.Identity())
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 輸入層
        x = self.input_layer(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # 隱藏層和殘差連接
        for i in range(len(self.hidden_layers)):
            identity = x
            
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            x = self.leaky_relu(x)
            
            # 添加殘差連接
            x = x + self.skip_connections[i](identity)
            x = self.dropout(x)
        
        # 輸出層
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

# 加權BCE損失函數
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight  # 正類別權重
        
    def forward(self, outputs, targets):
        # 計算BCE損失
        loss = - (self.pos_weight * targets * torch.log(outputs + 1e-7) + 
                  (1 - targets) * torch.log(1 - outputs + 1e-7))
        
        return torch.mean(loss)

# 早停機制類
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model is not None:
                    model.load_state_dict(self.best_model)

def advanced_feature_engineering(df):
    """
    進階特徵工程處理
    """
    print("開始進階特徵工程...")
    start_time = time.time()
    
    # 保留原有特徵工程
    # 1. BMI類別 (更細緻的分類)
    df['BMI_Category'] = pd.cut(
        df['BMI'], 
        bins=[0, 18.5, 23, 25, 27.5, 30, 35, 40, 100], 
        labels=[0, 1, 2, 3, 4, 5, 6, 7]
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
    df['Risk_Factor_Count'] = df['HighBP'] + df['HighChol'] + df['Stroke'] + df['HeartDiseaseorAttack'] + df['Smoker']
    
    # 9. 健康行為總數
    df['Healthy_Behaviors'] = df['PhysActivity'] + df['Fruits'] + df['Veggies'] + (1 - df['Smoker']) + (1 - df['HvyAlcoholConsump'])
    
    # 10. BMI平方項 (捕捉非線性關係)
    df['BMI_Squared'] = df['BMI'] ** 2
    
    # 11. 年齡與BMI交互作用
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    
    # --- 新增特徵 ---
    
    # 12. 風險因子聚合得分 - 使用對數轉換增強低值的辨別度
    df['Risk_Score_Log'] = np.log1p(df['Risk_Score'])
    
    # 13. BMI與年齡標準化交互項 (針對不同年齡段BMI的影響不同)
    df['BMI_Age_Norm'] = df['BMI'] / (df['Age'] + 1)
    
    # 14. 健康與生活方式組合指標
    df['Health_Lifestyle_Combined'] = df['Health_Score'] * df['Lifestyle_Score']
    
    # 15. 健康風險比例 - 風險因子與健康行為的比例
    df['Health_Risk_Ratio'] = (df['Risk_Score'] + 1) / (df['Healthy_Behaviors'] + 1)
    
    # 16. 醫療獲取難度指標 - 結合經濟狀況和醫療服務
    df['Healthcare_Access'] = df['AnyHealthcare'] * (7 - df['NoDocbcCost']) * (df['Income'] + 1)
    
    # 17. 健康集中度 - 反映各方面健康指標的分佈程度
    health_features = ['GenHlth', 'PhysHlth', 'MentHlth', 'DiffWalk']
    df['Health_Concentration'] = df[health_features].std(axis=1) / (df[health_features].mean(axis=1) + 1)
    
    # 18. 生活質量指數 - 綜合收入、教育和健康因素
    df['Life_Quality'] = (df['Income'] + 1) * (df['Education'] + 1) / (df['GenHlth'] + 1)
    
    # 19. 健康行為的一致性 - 衡量健康行為是否一致
    health_behavior_vars = ['PhysActivity', 'Fruits', 'Veggies', 'Smoker', 'HvyAlcoholConsump']
    df['Behavior_Consistency'] = df[health_behavior_vars].var(axis=1)
    
    # 20. 非線性轉換特徵 - 使用指數和對數轉換捕捉非線性關係
    df['BMI_Exp'] = np.exp(df['BMI'] / 10) - 1  # 指數轉換，避免數值過大
    df['Age_Log'] = np.log1p(df['Age'])
    
    # 21. 複合風險交互項
    df['BP_Chol_Interaction'] = df['HighBP'] * df['HighChol']
    df['Stroke_Heart_Interaction'] = df['Stroke'] * df['HeartDiseaseorAttack']
    
    # 22. 多項式特徵 - BMI與年齡的二次項
    df['BMI_Age_Poly'] = df['BMI'] * df['Age'] * df['Age'] / 100  # 除以100避免數值過大
    
    # 23. 基於臨床知識的特徵
    # 心血管風險綜合指標
    df['Cardio_Risk'] = df['HighBP'] + df['HighChol'] + df['HeartDiseaseorAttack'] + df['Stroke'] + df['BMI_Category'] / 2
    
    # 24. 心理健康對糖尿病的額外影響 (與整體健康交互)
    df['Mental_GenHealth_Impact'] = df['MentHlth'] * df['GenHlth'] / 5
    
    # 25. 社會經濟地位影響健康的綜合指標
    df['SES_Health_Impact'] = (df['Income'] + df['Education']) * (6 - df['GenHlth']) / 10
    
    # 確保所有特徵都是數值型別
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(int)
            
    print(f"特徵工程時間: {time.time() - start_time:.2f} 秒")
    print(f"總特徵數: {df.shape[1]}")
    
    return df

def select_optimal_features(df, target_col='Diabetes_binary', method='importance', top_n=25):
    """
    選擇最佳特徵子集
    
    參數:
    - df: 資料框
    - target_col: 目標變數列名
    - method: 'correlation' 或 'importance'
    - top_n: 選擇的特徵數量
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if method == 'correlation':
        # 使用相關係數選擇特徵
        correlation = df.corr()[target_col].abs().sort_values(ascending=False)
        selected_features = correlation[1:top_n+1].index.tolist()
    
    elif method == 'importance':
        # 使用隨機森林特徵重要性選擇特徵
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = feature_importance.head(top_n)['feature'].tolist()
    
    else:
        raise ValueError("方法必須是 'correlation' 或 'importance'")
    
    print(f"選擇了 {len(selected_features)} 個特徵:")
    for i, feature in enumerate(selected_features[:10], 1):
        print(f"{i}. {feature}")
    if len(selected_features) > 10:
        print(f"...以及 {len(selected_features) - 10} 個其他特徵")
    
    return selected_features

def load_and_preprocess_data(file_path):
    """
    載入和預處理資料
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
    
    # 使用進階特徵工程
    df = advanced_feature_engineering(df)
    
    # 特徵相關性
    correlation = df.corr()['Diabetes_binary'].sort_values(ascending=False)
    print("\n與目標變數的相關性:")
    print(correlation.head(10))
    
    return df

def apply_smote_oversampling(X_train, y_train):
    """
    使用SMOTE對少數類別進行過採樣
    """
    print("使用SMOTE過採樣技術平衡資料集...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 檢查過採樣後的類別分佈
    class_counts = np.bincount(y_resampled.astype(int))
    print(f"過採樣後的類別分佈: {class_counts}")
    
    return X_resampled, y_resampled

def create_learning_rate_scheduler(optimizer):
    """
    創建學習率調度器
    使用餘弦退火調度，在訓練過程中動態調整學習率
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,          # 第一次重啟的周期長度
        T_mult=2,        # 每次重啟後周期長度倍增
        eta_min=1e-6     # 最小學習率
    )
    return scheduler

def train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, patience=15):
    """
    改進的訓練函數，包含學習率調度和早停機制
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    lr_history = []
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            
            # 梯度歸零
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 添加L1正則化
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            # 反向傳播與優化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 計算正確率
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        # 在每個epoch結束後更新學習率
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            print(f"當前學習率: {current_lr:.2e}")
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                
                # 前向傳播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # 計算正確率
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停檢查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'早停觸發! 在第 {epoch+1} 輪停止訓練')
            break
    
    # 載入最佳模型
    if early_stopping.best_model is not None:
        model.load_state_dict(early_stopping.best_model)
    
    # 繪製學習率變化曲線
    if lr_history:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(lr_history)), lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.savefig(f"{output_dir}/learning_rate_schedule.png")
        plt.close()
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, criterion, device):
    """
    評估模型
    """
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            
            # 轉換為CPU並保存預測結果
            target_np = targets.cpu().numpy().flatten()
            output_np = outputs.cpu().numpy().flatten()
            pred_np = (output_np >= 0.5).astype(int)
            
            all_targets.extend(target_np)
            all_predictions.extend(pred_np)
            all_probabilities.extend(output_np)
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # 計算評估指標
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    
    print(f"\n測試損失: {test_loss:.4f}")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分數: {f1:.4f}")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # 繪製ROC曲線
    fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC曲線 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假陽性率')
    plt.ylabel('真陽性率')
    plt.title('ROC曲線')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    繪製訓練歷史
    """
    # 繪製損失曲線
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='訓練損失')
    plt.plot(val_losses, label='驗證損失')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.title('訓練和驗證損失')
    plt.legend()
    
    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='訓練準確率')
    plt.plot(val_accs, label='驗證準確率')
    plt.xlabel('Epoch')
    plt.ylabel('準確率')
    plt.title('訓練和驗證準確率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()

def main():
    # 載入和預處理資料
    file_path = 'diabetes\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    df = load_and_preprocess_data(file_path)
    
    # 特徵選擇
    selected_features = select_optimal_features(df, method='importance', top_n=30)
    
    # 分離特徵和目標變數
    X = df[selected_features]
    y = df['Diabetes_binary']
    
    # 分割資料
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"訓練集: {X_train.shape}, 驗證集: {X_val.shape}, 測試集: {X_test.shape}")
    
    # 使用SMOTE平衡訓練資料
    X_train_resampled, y_train_resampled = apply_smote_oversampling(X_train, y_train)
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 創建資料集和資料載入器
    batch_size = 64  # 嘗試較小的批次大小
    train_dataset = DiabetesDataset(X_train_scaled, y_train_resampled)
    val_dataset = DiabetesDataset(X_val_scaled, y_val.values)
    test_dataset = DiabetesDataset(X_test_scaled, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 定義模型參數
    input_size = X_train_scaled.shape[1]
    hidden_sizes = [256, 128, 64, 32]  # 更深的神經網絡
    
    # 初始化改進後的模型
    model = ImprovedDiabetesNet(input_size, hidden_sizes, dropout_rate=0.25).to(device)
    print(model)
    
    # 定義加權損失函數
    criterion = WeightedBCELoss(pos_weight=1.5)
    
    # 使用更好的優化器 - AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 創建學習率調度器
    scheduler = create_learning_rate_scheduler(optimizer)
    
    # 訓練模型
    print("\n開始訓練模型...")
    epochs = 200  # 增加訓練輪數
    model, train_losses, val_losses, train_accs, val_accs = train_model_with_scheduler(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, patience=20
    )
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 評估模型
    print("\n評估模型...")
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'selected_features': selected_features,
        'metrics': metrics
    }, f"{output_dir}/improved_diabetes_model.pth")
    print(f"模型已保存至: {output_dir}/improved_diabetes_model.pth")
    
    # 保存結果摘要
    with open(f"{output_dir}/results_summary.txt", "w") as f:
        f.write("糖尿病預測模型評估結果\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型類型: ImprovedDiabetesNet\n")
        f.write(f"隱藏層配置: {hidden_sizes}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"優化器: AdamW\n")
        f.write(f"初始學習率: {0.001}\n")
        f.write(f"特徵數量: {input_size}\n")
        f.write("\n性能指標:\n")
        f.write(f"準確率: {metrics['accuracy']:.4f}\n")
        f.write(f"精確率: {metrics['precision']:.4f}\n")
        f.write(f"召回率: {metrics['recall']:.4f}\n")
        f.write(f"F1分數: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
    
    # 打印主要指標
    print("\n模型效能摘要:")
    print(f"準確率: {metrics['accuracy']:.4f}")
    print(f"F1分數: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # 與基準模型比較
    print("\n與基準模型比較 (之前的最佳結果):")
    baseline_acc = 0.7464  # 從DL_pytorch.py的結果獲取
    baseline_f1 = 0.7546
    
    acc_improvement = (metrics['accuracy'] - baseline_acc) / baseline_acc * 100
    f1_improvement = (metrics['f1'] - baseline_f1) / baseline_f1 * 100
    
    print(f"準確率改進: {acc_improvement:.2f}%")
    print(f"F1分數改進: {f1_improvement:.2f}%")
    
    return model, metrics

if __name__ == "__main__":
    main()
    plt.close('all')  # 確保所有圖表都已關閉
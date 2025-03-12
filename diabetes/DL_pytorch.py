import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # 設定使用 Agg 非互動式後端
import matplotlib.pyplot as plt
import seaborn as sns

# 設定使用微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 如果還是有問題，可以嘗試這樣設定
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

# 建立輸出目錄
output_dir = 'diabetes/pytorch_models'
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

# 定義神經網絡模型
class DiabetesNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3):
        super(DiabetesNet, self).__init__()
        
        # 建立層列表
        layers = []
        
        # 輸入層到第一個隱藏層
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 建立所有隱藏層
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 輸出層
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        # 將所有層組合為一個序列模型
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    # 載入和預處理資料
def load_and_preprocess_data(file_path):
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
    print(correlation.head(10))
    
    return df

# 訓練神經網絡模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
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
            
            # 反向傳播與優化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 計算正確率
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
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
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 載入最佳模型
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs

# 評估模型
def evaluate_model(model, test_loader, criterion, device):
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
    plt.close('all')
    
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
    plt.close('all')
    
    return {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
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
    plt.close('all')  # 確保關閉所有圖形

def main():
    # 載入和預處理資料
    file_path = 'diabetes\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    df = load_and_preprocess_data(file_path)
    
    # 分離特徵和目標變數
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # 使用與GPU_ML.py類似的特徵選擇方法
    # 重要特徵列表 (參考GPU_ML.py的輸出結果)
    selected_features = [
        'Age_Health_Interaction', 'Risk_Score', 'GenHlth', 'Health_Score',
        'BMI_BP_Interaction', 'Age_BMI_Interaction', 'HighBP', 'Total_Risk_Factors',
        'BMI_Squared', 'BMI', 'BMI_Category', 'HighChol', 'Age', 'Income',
        'DiffWalk', 'Age_Group', 'Lifestyle_Score', 'Total_Healthy_Behaviors'
    ]
    
    X_selected = X[selected_features]
    
    # 分割資料
    X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"訓練集: {X_train.shape}, 驗證集: {X_val.shape}, 測試集: {X_test.shape}")
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 創建資料集和資料載入器
    train_dataset = DiabetesDataset(X_train_scaled, y_train.values)
    val_dataset = DiabetesDataset(X_val_scaled, y_val.values)
    test_dataset = DiabetesDataset(X_test_scaled, y_test.values)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 定義模型參數
    input_size = X_train_scaled.shape[1]
    hidden_sizes = [128, 64, 32]  # 隱藏層大小
    
    # 初始化模型
    model = DiabetesNet(input_size, hidden_sizes).to(device)
    print(model)
    
    # 定義損失函數和優化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # 訓練模型
    print("\n開始訓練模型...")
    epochs = 100
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, device
    )
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 評估模型
    print("\n評估模型...")
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    # 保存模型
    torch.save(model.state_dict(), f"{output_dir}/diabetes_model.pth")
    print(f"模型已保存至: {output_dir}/diabetes_model.pth")
    
    return model, metrics

if __name__ == "__main__":
    main()
    plt.close('all')  # 確保所有圖表都已關閉
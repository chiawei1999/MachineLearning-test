# 糖尿病預測模型 (Diabetes Prediction Model)

這個專案使用機器學習技術，基於健康指標資料集訓練模型以預測糖尿病風險。專案利用GPU加速處理與多種進階機器學習技術，提供高效且精確的預測結果。

## 專案概述

本專案利用 BRFSS 2015 健康指標數據集，開發與評估多種機器學習模型來預測個體是否患有糖尿病。各模型透過多種特徵工程技術與優化方法進行訓練，並提供詳細的效能比較分析。

## 資料集

使用了三個主要資料集：
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (5050分割版本)
- `diabetes_012_health_indicators_BRFSS2015.csv` (包含前期糖尿病分類)
- `diabetes_binary_health_indicators_BRFSS2015.csv` (二元分類完整版本)

每個資料集包含 22 個欄位，涵蓋多種健康指標，如血壓、膽固醇水平、BMI、生活習慣等特徵。

## 特徵工程

專案包含豐富的特徵工程與預處理步驟：

1. **基本特徵處理**：缺失值檢查與資料清理
2. **進階特徵創建**：
   - BMI分類（細緻化分類）
   - 年齡分組
   - 綜合健康評分
   - 風險因子綜合指標（權重優化）
   - 生活方式綜合指標
   - 特徵交互作用（BMI與高血壓、年齡與健康狀態等）
   - 非線性特徵（BMI平方項）

3. **特徵選擇**：使用 RFECV (Recursive Feature Elimination with Cross-Validation) 演算法，結合隨機森林進行特徵重要性評估

## 模型架構

專案實現並比較了多種機器學習模型：

1. **邏輯迴歸 (Logistic Regression)**：使用標準化特徵與平衡權重
2. **隨機森林 (Random Forest)**：使用200棵決策樹與平衡類別權重
3. **梯度提升樹 (Gradient Boosting)**：使用200個估計器與適中學習率
4. **XGBoost**：高效能梯度提升實現，使用直方圖加速
5. **加權投票集成 (Weighted Voting Ensemble)**：結合表現最佳的三個模型，根據各模型效能為其分配權重

## 模型效能

經過評估，各模型的預測效能如下：

| 模型               | 準確率   | 精確率   | 召回率   | F1分數   |
|--------------------|---------|---------|---------|---------|
| Gradient Boosting  | 0.7470  | 0.7284  | 0.7877  | 0.7569  |
| XGBoost            | 0.7457  | 0.7251  | 0.7912  | 0.7567  |
| Logistic Regression| 0.7442  | 0.7328  | 0.7686  | 0.7503  |
| Random Forest      | 0.7302  | 0.7146  | 0.7667  | 0.7397  |
| Weighted Voting    | 0.7467  | 0.7288  | 0.7860  | 0.7563  |

### 結果分析

- **梯度提升樹**為表現最佳的單一模型，準確率達 74.70%
- **XGBoost**在召回率方面表現最佳 (79.12%)，適合優先識別潛在病例
- **集成模型**在各項指標上表現均衡，準確率為 74.67%

## 主要功能

- **資料載入與預處理**：`load_and_preprocess_data()`
- **特徵選擇**：`enhanced_feature_selection()`
- **XGBoost 模型訓練**：`train_xgboost_model()`
- **多模型訓練與評估**：`train_and_evaluate_models()`
- **集成模型創建**：`create_voting_ensemble()`
- **視覺化與結果分析**：混淆矩陣、ROC曲線、特徵重要性等

## 使用方法

1. 確保安裝所需套件：
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

2. 放置資料集於正確路徑

3. 執行主程式：
```bash
python GPU_ML.py
```

## 輸出結果

程式將產生多種分析圖表與模型，儲存於 `diabetes/gpu_enhanced_models` 目錄：
- 特徵選擇結果
- 特徵重要性分析
- 各模型的混淆矩陣
- ROC曲線
- 模型比較結果
- 已訓練的模型檔案 (.pkl 與 .json 格式)

## 未來改進方向

1. 進一步最佳化模型超參數
2. 探索深度學習方法的應用
3. 開發更多組合特徵以提升預測效能
4. 優化GPU計算效率
5. 實現更全面的解釋性分析

## 授權資訊

本專案遵循 MIT 授權條款
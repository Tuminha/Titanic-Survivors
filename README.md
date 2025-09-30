# 🚢 Titanic Survival Prediction with Neural Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-red.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Titanic%20Dataset-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-XX.XX%25-green.svg)

**Predicting Titanic passenger survival using deep learning**

[🎯 View Project](#overview) • [📊 Results](#results) • [🚀 Quick Start](#quick-start) • [📈 Performance](#performance)

</div>

---

## 👨‍💻 **Author Information**

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-blue?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through CodeCademy • Building AI solutions step by step*

</div>

---

## 🎯 **Project Overview**

This project applies **neural networks** to predict Titanic passenger survival, following the same methodology as my previous hotel cancellation predictor. The goal is to reinforce deep learning concepts with a smaller, more manageable dataset.

### **🎓 Learning Objectives:**
- Master binary classification with neural networks
- Practice feature engineering on historical data
- Implement proper train/test evaluation
- Build production-ready ML pipeline

### **🏆 Key Achievements:**
- [x] **Data Loading**: Successfully loaded and analyzed Titanic dataset
- [ ] **Data Processing**: Handle missing values, feature engineering
- [ ] **Neural Architecture**: Custom PyTorch model design
- [ ] **Model Training**: Proper scaling, loss functions, optimization
- [ ] **Performance Analysis**: Comprehensive metrics and business interpretation

---

## 📊 **Dataset Information**

| Attribute | Description | Type | Missing Values |
|-----------|-------------|------|----------------|
| `PassengerId` | Unique identifier | Integer | 0 |
| `Survived` | Survival (0=No, 1=Yes) | Binary | 0 |
| `Pclass` | Ticket class (1st, 2nd, 3rd) | Ordinal | 0 |
| `Name` | Passenger name | String | 0 |
| `Sex` | Gender | Categorical | 0 |
| `Age` | Age in years | Numerical | ~20% |
| `SibSp` | Siblings/spouses aboard | Integer | 0 |
| `Parch` | Parents/children aboard | Integer | 0 |
| `Ticket` | Ticket number | String | 0 |
| `Fare` | Passenger fare | Numerical | 0 |
| `Cabin` | Cabin number | String | ~77% |
| `Embarked` | Port of embarkation | Categorical | ~0.2% |

**Total Samples**: 891 passengers | **Target Distribution**: ~38% survived

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn kaggle
```

### **Setup**
```bash
git clone https://github.com/Tuminha/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor
jupyter notebook titanic_survival.ipynb
```

---

## 📈 **Project Phases**

### Phase 1: Data Exploration ✅ COMPLETED
<details>
<summary><strong>🔍 Understand the Titanic Dataset</strong></summary>

- [x] **Task 1**: Download Kaggle Titanic dataset
- [x] **Task 2**: Exploratory data analysis and missing value assessment  
- [x] **Task 3**: Visualize survival patterns by passenger class, gender, age
- [x] **Task 4**: Correlation analysis and feature importance insights

**Key Findings:**
- ✅ **Survival Rate**: 38.4% overall survival rate
- ✅ **Gender Impact**: Women had 74% survival rate vs 19% for men
- ✅ **Class Hierarchy**: 1st class (63%) > 2nd class (47%) > 3rd class (24%)
- ✅ **Age Patterns**: Children (58%) > Young Adults (38%) > Seniors (29%) survival
- ✅ **Family Patterns**: Solo travelers (30%) vs small families (55-72%) survival
- ✅ **Strongest Predictors**: Pclass (-0.34), Fare (+0.26), Gender correlation
- ✅ **Missing Data**: Age (20%), Cabin (77%), Embarked (0.2%)

**Technical Achievements:**
- ✅ **Feature Engineering**: Created Family_Size, Family_Category, and Age_group features
- ✅ **Visualizations**: Generated correlation heatmaps and survival pattern charts
- ✅ **Statistical Analysis**: Comprehensive correlation matrix with key insights
- ✅ **Data Analysis**: Detailed survival rate breakdowns by all major features
- ✅ **Data Quality**: Identified missing value strategies for preprocessing

</details>

### Phase 2: Data Preprocessing ✅ COMPLETED
<details>
<summary><strong>🔧 Clean and Prepare Data for Neural Networks</strong></summary>

- [x] **Task 5**: Handle missing values (Age imputation, Embarked filling)
- [x] **Task 6**: Feature engineering (Title extraction, Family size)
- [x] **Task 7**: Encode categorical variables (Sex, Embarked one-hot)
- [x] **Task 8**: Feature scaling with StandardScaler

**Goals:**
- 🎯 **Missing Values**: Develop intelligent imputation strategies
- 🎯 **Feature Engineering**: Create new predictive features
- 🎯 **Categorical Encoding**: Convert text to numerical features
- 🎯 **Scaling**: Prepare data for neural network training

**Key Achievements:**
- ✅ **Missing Values**: Age imputed by class/gender medians, Embarked filled with mode, Cabin dropped
- ✅ **Feature Engineering**: Title extraction completed with 5 categories (Mr, Mrs, Miss, Master, High-Status)
- ✅ **Categorical Encoding**: All categorical variables encoded - Sex (label), Embarked (one-hot), Title_Group (one-hot), Family_Category (one-hot), Age_group (one-hot)
- ✅ **Data Cleanup**: Removed non-predictive columns (Name, Ticket, Title)
- ✅ **Feature Scaling**: StandardScaler applied to numerical features (Age, Fare, Pclass, SibSp, Parch, Family_Size)

**Technical Achievements:**
- ✅ **Complete Preprocessing Pipeline**: 22 features ready for neural network training
- ✅ **Professional Encoding Strategy**: Label encoding for binary, one-hot for multiple categories
- ✅ **Feature Standardization**: All numerical features normalized (mean=0, std=1)
- ✅ **Data Quality**: All features numerical, no missing values, proper scaling

</details>

### Phase 3: Model Preparation 🚧 IN PROGRESS
<details>
<summary><strong>⚙️ Prepare Data for PyTorch Neural Network</strong></summary>

- [x] **Task 9**: Import PyTorch libraries
- [x] **Task 10**: Create feature and target tensors
- [x] **Task 11**: Train/test split (80/20)
- [ ] **Task 12**: Verify data shapes and scaling

**Goals:**
- 🎯 **Tensor Creation**: Convert preprocessed data to PyTorch tensors
- 🎯 **Data Split**: 80/20 train/test maintaining class balance
- 🎯 **Pipeline Validation**: Ensure consistent scaling across splits

**Progress:**
- ✅ **PyTorch Setup**: Libraries imported and environment ready
- ✅ **Tensor Creation**: Features (891, 24) and target (891,) converted to PyTorch tensors
- ✅ **Feature Scaling**: StandardScaler applied to numerical features
- ✅ **Train/Test Split**: 80/20 split with stratify maintaining class balance
- 🎯 **Next**: Final verification of tensor shapes and data preparation

</details>

### Phase 4: Neural Network Classification
<details>
<summary><strong>🧠 Build and Train Survival Prediction Model</strong></summary>

- [ ] **Task 13**: Build neural network architecture
- [ ] **Task 14**: Define loss function and optimizer
- [ ] **Task 15**: Train model with progress tracking
- [ ] **Task 16**: Evaluate on test set
- [ ] **Task 17**: Calculate comprehensive metrics

**Target Architecture:**
- 🎯 **Input Layer**: Number of features after preprocessing
- 🎯 **Hidden Layers**: 16 → 8 nodes (simpler than hotel model)
- 🎯 **Output Layer**: 1 node with sigmoid activation
- 🎯 **Expected Performance**: 80%+ accuracy

</details>

---

## 🏆 **Results**

### **Model Performance**
```
Final Test Results:
├── Accuracy:  XX.XX%
├── Precision: XX.XX%
├── Recall:    XX.XX%
└── F1-Score:  XX.XX%
```

### **Business Interpretation**
- **Historical Insight**: Model identifies key survival factors
- **Feature Importance**: Gender, class, and age most predictive
- **Generalization**: Strong performance on unseen passengers

---

## 🛠️ **Technical Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Matplotlib, Seaborn | EDA and results visualization |
| **Machine Learning** | PyTorch | Neural network implementation |
| **Evaluation** | Scikit-learn | Model metrics and validation |
| **Version Control** | Git/GitHub | Project tracking and collaboration |

---

## 📝 **Learning Journey**

This project reinforces concepts from my previous **Hotel Cancellation Predictor** with a simpler dataset:

**Previous Project**: 119K hotel bookings → 82.65% accuracy
**Current Project**: 891 passengers → Target 80%+ accuracy

**Skills Reinforced**:
- [x] Data loading and exploratory data analysis
- [x] Statistical analysis and correlation interpretation
- [x] Data visualization with matplotlib and seaborn
- [x] Feature engineering and data transformation
- [x] Comprehensive survival rate analysis by multiple features
- [x] Age group categorization and family size analysis
- [x] Missing value imputation strategies and data preprocessing
- [ ] Neural network architecture design  
- [ ] Training loop implementation and debugging
- [ ] Model evaluation and business interpretation

---

## 🚀 **Next Steps**

- [ ] **Model Optimization**: Hyperparameter tuning, regularization
- [ ] **Feature Engineering**: Advanced feature creation
- [ ] **Ensemble Methods**: Combine multiple models
- [ ] **Deployment**: Create prediction API

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

*Building AI solutions one dataset at a time* 🚀

</div>

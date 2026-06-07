# 🎯 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![AWS](https://img.shields.io/badge/AWS-EC2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🎯 Customer Churn Prediction System

A production-ready machine learning platform that predicts customer churn 
with 80.45% accuracy using advanced ML techniques and cloud infrastructure.

## 🌟 Features

- **🔮 Real-Time Predictions**: Get churn risk scores instantly
- **📊 Interactive Dashboard**: Beautiful Streamlit interface for predictions
- **☁️ Cloud Deployment**: Running on AWS EC2 (always online)
- **📈 Production Monitoring**: CloudWatch alerts & metrics
- **🐳 Containerized**: Docker for easy deployment anywhere
- **🧪 Tested**: 25 unit tests ensuring reliability
- **📚 Well Documented**: Complete API & architecture docs

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.45% |
| Precision | 64.81% |
| Recall | 56.18% |
| F1 Score | 0.6020 |
| ROC AUC | 0.8445 |

## 🏗️ System Architecture
Data Pipeline → Model Training → REST API → AWS EC2
↓                ↓
Hyperparameter        FastAPI
Tuning (80%+)         Uvicorn
↓                ↓
Evaluation          Docker Container
↓
Streamlit Dashboard

## 📈 Live Demo

**🎬 Try it now:** 
[Streamlit Dashboard](https://churn-prediction-pipeline-maqkiputjuuh634q8gowtc.streamlit.app/)

**📊 API Endpoint (AWS EC2):**
- Health Check: `http://3.139.61.131:8000/health`
- API Docs: `http://3.139.61.131:8000/docs`

## 🛠️ Technologies Used

**Data & ML:**
- Pandas, NumPy
- Scikit-learn
- Logistic Regression

**Backend & API:**
- FastAPI
- Uvicorn
- Pydantic

**Frontend & Dashboard:**
- Streamlit
- Plotly
- Pandas

**DevOps & Cloud:**
- Docker
- AWS EC2
- CloudWatch

**Testing:**
- Pytest
- Coverage

## 📊 Key Results

### Model Performance
- Achieved 80.45% accuracy on test set
- 64.81% precision (minimize false positives)
- 56.18% recall (catch high-risk customers)
- ROC AUC of 0.8445 (excellent discrimination)

### Deployment
- ✅ Containerized with Docker
- ✅ Running on AWS EC2 24/7
- ✅ Free tier ($0/month)
- ✅ Publicly accessible
- ✅ Monitored with CloudWatch

### Testing
- ✅ 25 unit tests passing
- ✅ Integration tests
- ✅ API health checks
- ✅ 95%+ code coverage

## 📚 Documentation

- [Architecture Design](ARCHITECTURE.md) - System design & components
- [API Documentation](API_DOCUMENTATION.md) - Endpoints & examples
- [Results Report](RESULTS_REPORT.md) - Detailed metrics & analysis
- [Setup Guide](SETUP.md) - Installation & configuration

## 📝 Model Details

**Algorithm:** Logistic Regression
**Features:** 19 customer behavioral & demographic features
**Training Data:** 7,043 telecom customers
**Target Variable:** Customer churn (Yes/No)

**Key Features:**
- Customer demographics (age, family status)
- Service usage (internet, phone, streaming)
- Contract information (type, billing)
- Billing data (monthly & total charges)

## 🔄 Data Pipeline

1. **EDA & Exploration** → Understand data
2. **Data Cleaning** → Handle missing values, outliers
3. **Feature Engineering** → Create meaningful features
4. **Preprocessing** → Encode, scale, split
5. **Model Training** → Hyperparameter tuning
6. **Evaluation** → Performance metrics
7. **Deployment** → API & Dashboard

## 💡 Key Insights

- Month-to-month contracts have 42% churn vs 11% for 2-year
- Fiber optic users churn more (41%) vs DSL (27%)
- Electronic check payment method has 45% churn rate
- Longer tenure strongly correlates with retention
- Higher monthly charges correlate with churn risk

## 🎯 Business Value

This system enables:
✅ Identify at-risk customers proactively
✅ Prioritize retention efforts on high-risk segments
✅ Estimate potential revenue impact of churn
✅ Test retention strategies with A/B testing
✅ Monitor model performance over time

## 📈 Future Enhancements

- [ ] Real-time model monitoring dashboard
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] Customer segmentation analysis
- [ ] Causal inference (why do they churn?)
- [ ] Multi-model ensemble
- [ ] Explainability (SHAP values)

## 📧 Contact & Social

- [Github](https://github.com/zianafique)
- [LinkedIn](https://linkedin.com/in/afiquezian)
- Email: afiquezian@gmail.com

⭐ If you found this helpful, please star the repository!

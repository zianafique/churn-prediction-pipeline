"""
🎨 Professional Churn Prediction Dashboard
==========================================
A comprehensive customer churn risk assessment and prediction platform
designed for business teams to make data-driven retention decisions.

Features:
- Real-time churn probability predictions
- Risk stratification and business recommendations
- Customer profile analysis and comparison
- Interactive visualizations
- Export capabilities for business reporting
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Churn Risk Assessment Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM STYLING & THEME
# ============================================================================

st.markdown(
    """
    <style>
        /* Main container */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        /* Header styling */
        h1 {
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #764ba2;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Risk level colors */
        .risk-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ff0000;
            box-shadow: 0 4px 12px rgba(255,107,107,0.3);
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #ffa502 0%, #ff7a00 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ff7a00;
            box-shadow: 0 4px 12px rgba(255,165,2,0.3);
        }
        
        .risk-low {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
            box-shadow: 0 4px 12px rgba(46,204,113,0.3);
        }
        
        /* Info box */
        .info-box {
            background: #ecf0f1;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Success box */
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            border-radius: 5px;
            color: #155724;
        }
        
        /* Section divider */
        .divider {
            border-top: 2px solid #667eea;
            margin: 30px 0;
        }
        
        /* Button styling */
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-weight: bold !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_URL = "http://3.139.61.131:8000/predict"

# ============================================================================
# HEADER & DESCRIPTION
# ============================================================================

st.markdown("# 📊 Churn Risk Assessment Platform")

st.markdown(
    """
    <div class="info-box">
    <h3>🎯 Platform Overview</h3>
    <p>
    This enterprise-grade churn prediction platform leverages advanced machine learning 
    to identify customers at risk of leaving. Using 19 behavioral and demographic features, 
    our Logistic Regression model achieves <strong>80.45% accuracy</strong> in predicting 
    customer churn within the next billing cycle.
    </p>
    <p><strong>Key Capabilities:</strong></p>
    <ul>
        <li>🔮 <strong>Real-Time Predictions:</strong> Get churn risk scores instantly</li>
        <li>📊 <strong>Risk Stratification:</strong> Classify customers into actionable risk tiers</li>
        <li>💡 <strong>Actionable Insights:</strong> Receive data-driven retention recommendations</li>
        <li>👤 <strong>Customer Profile Analysis:</strong> Compare customer against population averages</li>
        <li>📈 <strong>Risk Metrics:</strong> Detailed probability and confidence scores</li>
    </ul>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_endpoint = st.text_input(
        "API Endpoint", value=API_URL, help="URL of the churn prediction API"
    )

    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.info("""
    1. Fill in customer information in the form
    2. Click 'Predict Churn Risk'
    3. Review prediction results and recommendations
    4. Compare customer profile with population statistics
    5. Export results for team communication
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.metric("Model Type", "Logistic Regression")
    st.metric("Accuracy", "80.45%")
    st.metric("Precision", "64.81%")
    st.metric("Recall", "56.18%")
    st.metric("ROC AUC", "0.8445")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

col1, col2 = st.columns([1.2, 1], gap="large")

# LEFT COLUMN: INPUT FORM
with col1:
    st.markdown("## 👤 Customer Information")

    # Demographics Section
    with st.expander("👥 Demographics", expanded=True):
        dem_col1, dem_col2 = st.columns(2)
        with dem_col1:
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer gender")
            senior = st.selectbox(
                "Senior Citizen", ["No", "Yes"], help="Is customer 65+?"
            )
            partner = st.selectbox(
                "Partner", ["Yes", "No"], help="Has spouse/partner on account?"
            )
        with dem_col2:
            dependents = st.selectbox(
                "Dependents", ["Yes", "No"], help="Has children/dependents?"
            )
            tenure = st.slider(
                "Tenure (Months)",
                0,
                72,
                24,
                help="How long customer has been with company",
            )

    # Services Section
    with st.expander("🔌 Services & Features", expanded=True):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        internet_services_col1, internet_services_col2 = st.columns(2)
        with internet_services_col1:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
        with internet_services_col2:
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

    # Contract & Payment Section
    with st.expander("💳 Contract & Payment", expanded=True):
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Month-to-month has highest churn risk",
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            help="Electronic check customers have highest churn",
        )

    # Charges Section
    with st.expander("💰 Billing Information", expanded=True):
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 0.0, 500.0, 65.5, help="Current monthly bill amount"
        )
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, 1570.0, help="Lifetime total charges"
        )

# RIGHT COLUMN: RESULTS & VISUALIZATIONS
with col2:
    st.markdown("## 🎯 Prediction Results")

    # Predict Button
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # Prepare customer data
        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        try:
            with st.spinner("🔄 Analyzing customer profile..."):
                response = requests.post(api_endpoint, json=customer_data, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prob = result["churn_probability"]
                    prediction = result["churn"]

                    # Determine risk level
                    if prob >= 0.7:
                        risk_class = "risk-high"
                        risk_label = "🔴 HIGH RISK"
                        risk_description = "Immediate Action Required"
                        risk_color = "#ff6b6b"
                    elif prob >= 0.5:
                        risk_class = "risk-medium"
                        risk_label = "🟡 MEDIUM RISK"
                        risk_description = "Proactive Engagement Needed"
                        risk_color = "#ffa502"
                    else:
                        risk_class = "risk-low"
                        risk_label = "🟢 LOW RISK"
                        risk_description = "Maintain Relationship"
                        risk_color = "#2ecc71"

                    # Display Risk Box
                    st.markdown(
                        f"""
                        <div class="{risk_class}">
                            <h2>{risk_label}</h2>
                            <p style="font-size: 18px; margin: 10px 0;"><strong>{risk_description}</strong></p>
                            <p style="font-size: 24px; font-weight: bold;">
                                Churn Probability: {prob * 100:.1f}%
                            </p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Key Metrics
                    st.markdown("### 📈 Prediction Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric(
                            "Churn Risk",
                            f"{prob * 100:.1f}%",
                            delta=None,
                            delta_color="inverse",
                        )

                    with metric_col2:
                        st.metric("Retention Likelihood", f"{(1 - prob) * 100:.1f}%")

                    with metric_col3:
                        st.metric(
                            "Model Confidence", f"{max(prob, 1 - prob) * 100:.1f}%"
                        )

                    st.markdown("---")

                    # Recommendation
                    st.markdown("### 💡 Business Recommendation")
                    st.markdown(
                        f"""
                        <div class="success-box">
                        <h4>Recommended Action Plan</h4>
                        {result["recommendation"]}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Risk Gauge Chart
                    st.markdown("### 📊 Risk Gauge")

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=prob * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Churn Probability (%)"},
                            delta={"reference": 50},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": risk_color},
                                "steps": [
                                    {"range": [0, 30], "color": "#d4edda"},
                                    {"range": [30, 70], "color": "#fff3cd"},
                                    {"range": [70, 100], "color": "#f8d7da"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 90,
                                },
                            },
                        )
                    )
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")

                    # Customer Profile Summary
                    st.markdown("### 👤 Customer Profile Summary")

                    profile_data = {
                        "Attribute": [
                            "Tenure",
                            "Contract Type",
                            "Monthly Bill",
                            "Internet Service",
                            "Streaming Services",
                        ],
                        "Value": [
                            f"{tenure} months ({tenure // 12}y {tenure % 12}m)",
                            contract,
                            f"${monthly_charges:.2f}",
                            internet if internet != "No" else "None",
                            "Yes"
                            if (streaming_tv == "Yes" or streaming_movies == "Yes")
                            else "No",
                        ],
                    }

                    profile_df = pd.DataFrame(profile_data)
                    st.dataframe(profile_df, use_container_width=True, hide_index=True)

                    # Download Results
                    st.markdown("### 📥 Export Results")

                    export_data = {
                        "Prediction Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Churn Prediction": prediction,
                        "Churn Probability": f"{prob * 100:.2f}%",
                        "Risk Level": risk_label,
                        **customer_data,
                    }

                    export_json = json.dumps(export_data, indent=2)

                    st.download_button(
                        label="📄 Download as JSON",
                        data=export_json,
                        file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.error(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot Connect to API")
            st.error(f"The API at {api_endpoint} is not responding. Please verify:")
            st.warning("• EC2 instance is running")
            st.warning("• Docker container is running")
            st.warning("• API endpoint URL is correct")
            st.warning("• Security group allows port 8000")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #667eea; padding: 20px;">
        <h3>📊 Churn Prediction Platform v1.0</h3>
        <p><strong>Model Performance:</strong> 80.45% Accuracy | 64.81% Precision | 56.18% Recall | 0.8445 ROC AUC</p>
        <p><strong>Technology Stack:</strong> Python | FastAPI | Scikit-learn | Streamlit | AWS</p>
    </div>
""",
    unsafe_allow_html=True,
)

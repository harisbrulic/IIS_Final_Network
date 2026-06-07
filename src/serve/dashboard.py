import os
import streamlit as st
import requests
import pandas as pd
import mlflow
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Network attack detection",
    layout="wide"
)

st.sidebar.title("Network attack detection")
page = st.sidebar.radio("Navigation", [
    "Live Monitor",
    "Admin Panel",
    "Metrics"
])

if page == "Live Monitor":
    st.title("Live Monitor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Please enter your data:")
        
        ip = st.text_input("IP address", "185.178.208.153")
        port = st.number_input("Port", value=80)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        num_packets = st.number_input("# of packets", value=100)
        packet_size = st.number_input("Size of packets", value=60)
        duration = st.number_input("Connection duration", value=1.0)
        confidence = st.slider(
            "Confidence (0=unknown, 100=definitely malicious)",
            0, 100, 0
        )
        
        if st.button("Analyse", type="primary"):
            
            protocol_num = {"TCP": 6, "UDP": 17, "ICMP": 1}[protocol]
            
            features = [
                20.0, float(protocol_num), 64.0, float(num_packets),
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0, 0, 0, 0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                float(num_packets * packet_size),
                float(packet_size), float(packet_size),
                float(packet_size), 0.0, float(packet_size),
                float(duration), float(num_packets), 0.0
            ]
            
            try:
                response = requests.post(f"{API_URL}/predict", json={
                    "features": features,
                    "ip": ip,
                    "port": int(port),
                    "confidence": float(confidence)
                })
                
                result = response.json()
                
                with col2:
                    st.subheader("Results")
                    
                    if result["is_anomaly"]:
                        st.error("ATTACK DETECTED!")
                        st.error(f"Attack type: {result['attack_type']}")
                        st.error(f"IP threat: {result['ip_threat']}")
                        st.metric(
                            "Reconstruction Error",
                            f"{result['reconstruction_error']:.2f}"
                        )
                        
                        try:
                            lime_response = requests.post(f"{API_URL}/explain", json={
                                "features": features
                            })
                            if lime_response.status_code == 200:
                                lime_data = lime_response.json()
                                st.subheader("LIME")
                                st.caption("Top features:")
                                lime_df = pd.DataFrame(
                                    list(lime_data["lime_values"].items()),
                                    columns=["Feature", "Value"]
                                ).sort_values("Value", ascending=False)
                                st.bar_chart(lime_df.set_index("Feature"))
                        except Exception as e:
                            st.warning(f"LIME not available: {e}")
                            
                        st.subheader("AI Explanation")
                        with st.spinner("Google Gemini"):
                            try:
                                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                                
                                gemini_response = gemini_model.generate_content(f"""
                                    Network attack detected:
                                    - Attack type: {result['attack_type']}
                                    - IP threat: {result['ip_threat']}
                                    - Source IP: {ip}
                                    - Port: {port}
                                    - Protocol: {protocol}
                                    - Packets/sec: {num_packets}
                                    
                                    Explain in simple terms and recommend actions.
                                    Keep under 100 words.
                                """)
                                
                                st.info(gemini_response.text)
                            except Exception as e:
                                st.warning(f"LLM not available: {e}")
                        
                    else:
                        st.success("No attack detected")
                        
                    
                    st.json(result)
            
            except Exception as e:
                st.error(f"Error: {e}")
    
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    
    if 'result' in locals():
        st.session_state.predictions.append({
            "IP": ip,
            "Port": port,
            "Attack": result.get("attack_type", "N/A"),
            "Anomaly": result.get("is_anomaly", False),
            "Error": f"{result.get('reconstruction_error', 0):.2f}"
        })
    
    if st.session_state.predictions:
        st.subheader("History")
        st.dataframe(pd.DataFrame(st.session_state.predictions))

elif page == "Admin Panel":
    st.title("Admin Panel")
    
    st.subheader("API Status")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success("API IS WORKING")
        else:
            st.error("API IS NOT WORKING")
    except:
        st.error("API IS NOT AVAILABLE")
    
    st.divider()
    
    st.subheader("Live Monitoring")
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total predictions", stats["total_predictions"])
        with col2:
            st.metric("Attacks", stats["total_attacks"])
        with col3:
            st.metric("Normal", stats["total_normal"])
        with col4:
            if stats["total_predictions"] > 0:
                rate = stats["total_attacks"] / stats["total_predictions"] * 100
                st.metric("Attack rate", f"{rate:.1f}%")
        
        if stats["last_prediction"]:
            st.info(f"Last prediction: {stats['last_prediction']}")
        
        if stats["attack_types"]:
            st.subheader("Attack types distribution")
            attack_df = pd.DataFrame(
                list(stats["attack_types"].items()),
                columns=["Attack", "Count"]
            )
            st.bar_chart(attack_df.set_index("Attack"))
            
    except Exception as e:
        st.error(f"Stats error: {e}")
    
    st.divider()
    
    st.subheader("Model Metrics from MLflow")
    try:
        client_mlflow = mlflow.tracking.MlflowClient()
        
        experiments = {
            "autoencoder": ["detection_rate", "final_loss", "threshold"],
            "xgboost": ["accuracy", "f1_score"],
            "ip_classifier": ["accuracy", "f1_score"]
        }
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (exp_name, metrics) in enumerate(experiments.items()):
            exp = client_mlflow.get_experiment_by_name(exp_name)
            if exp:
                runs = client_mlflow.search_runs(exp.experiment_id)
                if runs:
                    latest = runs[0]
                    with cols[i]:
                        st.subheader(f"{exp_name}")
                        for metric in metrics:
                            value = latest.data.metrics.get(metric, "N/A")
                            if isinstance(value, float):
                                st.metric(metric, f"{value:.4f}")
                            else:
                                st.metric(metric, value)
    except Exception as e:
        st.error(f"MLflow error: {e}")
    
    st.divider()
    
    st.subheader("Data Validation Report")
    ge_path = "gx/uncommitted/data_docs/local_site/index.html"
    if os.path.exists(ge_path):
        with open(ge_path) as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)
    else:
        st.warning("GE report not available")
    
    st.divider()
    
    st.subheader("Data Testing Report")
    evidently_path = "reports/data_testing_report.html"
    if os.path.exists(evidently_path):
        with open(evidently_path) as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)
    else:
        st.warning("Evidently report not available")

elif page == "Metrics":
    st.title("Model Metrics")
    
    try:
        client_mlflow = mlflow.tracking.MlflowClient()
        
        data = []
        for exp_name in ["autoencoder", "xgboost", "ip_classifier"]:
            exp = client_mlflow.get_experiment_by_name(exp_name)
            if exp:
                runs = client_mlflow.search_runs(exp.experiment_id)
                if runs:
                    latest = runs[0]
                    row = {
                        "Model": exp_name,
                        **latest.data.metrics
                    }
                    data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)
            
            if "accuracy" in df.columns:
                st.subheader("Accuracy comparison")
                acc_data = df[df["accuracy"].notna()][["Model", "accuracy"]]
                st.bar_chart(acc_data.set_index("Model"))
                
            if "detection_rate" in df.columns:
                st.subheader("Detection Rate")
                det_data = df[df["detection_rate"].notna()][["Model", "detection_rate"]]
                st.bar_chart(det_data.set_index("Model"))
                
    except Exception as e:
        st.error(f"Error: {e}")
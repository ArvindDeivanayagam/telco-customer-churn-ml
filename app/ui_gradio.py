import gradio as gr
import requests
import pandas as pd


API_URL = "http://127.0.0.1:8000/predict"


def predict_from_form(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    threshold,
):
    features = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    payload = {"features": features, "threshold": float(threshold)}
    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    out = r.json()

    return (
        out["churn_prediction"],
        float(out["churn_probability"]),
        out["run_id"],
    )


with gr.Blocks(title="Telco Customer Churn Predictor") as demo:
    gr.Markdown("## Telco Customer Churn Predictor (XGBoost + MLflow + FastAPI)")

    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(["Female", "Male"], value="Female", label="gender")
            SeniorCitizen = gr.Dropdown([0, 1], value=0, label="SeniorCitizen")
            Partner = gr.Dropdown(["Yes", "No"], value="Yes", label="Partner")
            Dependents = gr.Dropdown(["Yes", "No"], value="No", label="Dependents")

            tenure = gr.Number(value=5, label="tenure (months)")
            PhoneService = gr.Dropdown(["Yes", "No"], value="Yes", label="PhoneService")
            MultipleLines = gr.Dropdown(["No", "Yes", "No phone service"], value="No", label="MultipleLines")

            InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], value="Fiber optic", label="InternetService")
            OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="OnlineSecurity")
            OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="OnlineBackup")
            DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="DeviceProtection")
            TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="TechSupport")

            StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], value="Yes", label="StreamingTV")
            StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], value="No", label="StreamingMovies")

            Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract")
            PaperlessBilling = gr.Dropdown(["Yes", "No"], value="Yes", label="PaperlessBilling")
            PaymentMethod = gr.Dropdown(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                value="Electronic check",
                label="PaymentMethod",
            )

            MonthlyCharges = gr.Number(value=85.5, label="MonthlyCharges")
            TotalCharges = gr.Number(value=420.3, label="TotalCharges")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision threshold")

            btn = gr.Button("Predict churn")

        with gr.Column():
            pred_label = gr.Label(num_top_classes=2, label="Churn prediction (0=No, 1=Yes)")
            pred_proba = gr.Number(label="Churn probability")
            run_id = gr.Textbox(label="MLflow run_id")

    btn.click(
        fn=predict_from_form,
        inputs=[
            gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges, threshold
        ],
        outputs=[pred_label, pred_proba, run_id],
    )

demo.launch()

# AI-Powered Flood Risk Analytics Portal

An end-to-end Machine Learning web application that predicts local flood risks based on historical environmental parameters. The system utilizes a trained classification pipeline to deliver real-time risk inference via an intuitive user interface.

## 🚀 Tech Stack
* **Backend Framework:** Flask (Python)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Core Algorithm:** Random Forest Classifier (~99% Predictive Accuracy)
* **Frontend UI:** Bootstrap 5, FontAwesome, HTML5/CSS3 (Glassmorphism layout)

## 📊 Core Features
* **Predictive Features:** Tracks Annual Rainfall (mm), Monsoon Seasonal Index, and Cloud Visibility Index.
* **Instant Evaluation:** High-speed serialized inference using custom pre-fit preprocessing scalers (`.pkl`).
* **Visual Alerts:** Dynamically updating UI banners utilizing risk thresholds.

## 🛠️ Local Setup Instruction
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
📄 README.md (copy पूरा का पूरा)

\# 🏭 FactoryGuard-AI



FactoryGuard-AI is a Machine Learning based predictive maintenance project that detects potential machine failures using sensor data.  

The system uses a trained ML model and exposes predictions through a Flask REST API.



---



\## 🚀 Project Features

\- Sensor data analysis (temperature, vibration, pressure)

\- Machine failure prediction using ML

\- Trained model saved with Joblib

\- Flask API for real-time predictions

\- Clean project structure

\- GitHub ready



---



\## 📁 Project Structure





FactoryGuard-AI/

│

├── api/

│ └── app.py # Flask API

│

├── data/

│ └── sensor\_data\_v1.csv # Sensor dataset

│

├── models/

│ └── model.pkl # Trained ML model

│

├── notebooks/

│ ├── data\_analysis.ipynb

│ └── model\_training.ipynb

│

├── .gitignore

└── README.md





---



\## 🧠 Machine Learning Model

\- Algorithm: (Logistic Regression / RandomForest – as used)

\- Features:

&nbsp; - Temperature

&nbsp; - Vibration

&nbsp; - Pressure

\- Target:

&nbsp; - Failure (0 = No Failure, 1 = Failure)



---



\## ⚙️ How to Run the Project



\### 1️⃣ Clone Repository

```bash

git clone https://github.com/Mukesh3597/FactoryGuard-AI.git

cd FactoryGuard-AI



2️⃣ Install Dependencies

pip install flask numpy pandas scikit-learn joblib



3️⃣ Run Flask API

cd api

python app.py





API will start at:



http://127.0.0.1:5000



🔌 API Endpoint

🔹 Predict Failure



POST /predict



Request JSON:



{

&nbsp; "temperature": 72.5,

&nbsp; "vibration": 0.56,

&nbsp; "pressure": 28.4

}





Response JSON:



{

&nbsp; "prediction": 0

}



📊 Model Performance



PR-AUC Score: 0.0092



Dataset is highly imbalanced (failure is rare)



🛠️ Tools \& Technologies



Python



Pandas, NumPy



Scikit-learn



Flask



Joblib



Git \& GitHub



Jupyter Notebook



👤 Author



Mukesh

GitHub: Mukesh3597 


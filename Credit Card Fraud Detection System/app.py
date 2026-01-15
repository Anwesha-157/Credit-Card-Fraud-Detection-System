from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

app = Flask(__name__)
app.secret_key = "secret123"  # required for sessions

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy User Model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Example user list (can extend to DB later)
users = {"admin": User(id=1, username="admin", password="admin123")}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if str(user.id) == user_id:
            return user
    return None

# Folders
UPLOAD_FOLDER = 'upload'
RESULT_FOLDER = 'result'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global storage for results
results_data = {}

@app.route('/')
def index():
    return render_template("login.html")  # login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect('/dashboard')
        return "Invalid credentials! Try again."
    return render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if not results_data:
        return render_template("dashboard.html", message="No dataset uploaded yet.")

    return render_template("dashboard.html",
                           fraud_count=results_data['fraud_count'],
                           total_transactions=results_data['total_transactions'],
                           fraud_percentage=results_data['fraud_percentage'],
                           fraud_plot="fraud_agencies.png")

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if not file:
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Step 1: Load Data
        df = pd.read_csv(filepath)
        actual_cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
        df.columns = actual_cols

        expected_cols = ['transaction_date', 'agency_name', 'vendor', 'amount']
        if not all(col in actual_cols for col in expected_cols):
            return f"Required columns missing. Found: {actual_cols}", 400

        df = df[expected_cols]
        df.columns = ['TRANSACTION_DATE', 'AGENCY_NAME', 'MERCHANT_NAME', 'TRANSACTION_AMOUNT']

        # Raw preview
        raw_preview_html = df.head().to_html(classes="table table-bordered", index=False)

        # Step 2: Preprocessing
        preprocessing_info = []
        preprocessing_info.append(f"Initial Shape: {df.shape}")
        preprocessing_info.append("Missing Values:\n" + df.isnull().sum().to_string())

        df.dropna(inplace=True)
        df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'], errors='coerce')
        df.dropna(subset=['TRANSACTION_DATE'], inplace=True)

        preprocessing_info.append(f"After Cleaning Shape: {df.shape}")

        # Feature engineering
        df['Month'] = df['TRANSACTION_DATE'].dt.month
        df['DayOfWeek'] = df['TRANSACTION_DATE'].dt.dayofweek
        df['AGENCY_CODE'] = df['AGENCY_NAME'].astype('category').cat.codes
        df['MERCHANT_CODE'] = df['MERCHANT_NAME'].astype('category').cat.codes

        # Step 3: EDA
        eda_lines = []
        eda_lines.append("Top 5 Agencies:\n" + df['AGENCY_NAME'].value_counts().head().to_string())
        eda_lines.append("Top 5 Merchants:\n" + df['MERCHANT_NAME'].value_counts().head().to_string())
        eda_lines.append("Transaction Amount Stats:\n" + df['TRANSACTION_AMOUNT'].describe().to_string())
        eda_lines.append("Transactions per Month:\n" + df['Month'].value_counts().sort_index().to_string())

        eda_plot_filename = "eda_plot.png"
        plt.figure(figsize=(10, 5))
        sns.histplot(df['TRANSACTION_AMOUNT'], bins=100, color='skyblue')
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_FOLDER, eda_plot_filename))
        plt.close()

        # Step 4: Model
        features = df[['TRANSACTION_AMOUNT', 'AGENCY_CODE', 'MERCHANT_CODE', 'Month', 'DayOfWeek']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
        model.fit(features_scaled)
        df['Anomaly'] = model.predict(features_scaled)

        anomalies = df[df['Anomaly'] == -1]
        fraud_table_html = anomalies.to_html(classes="table table-striped", index=False)

        # Summary counts
        total_transactions = len(df)
        fraud_count = len(anomalies)
        fraud_percentage = round((fraud_count / total_transactions) * 100, 2)

        # Step 5: Fraud Agency Visualization
        fraud_plot_filename = "fraud_agencies.png"
        plt.figure(figsize=(10, 4))
        top_agencies = anomalies['AGENCY_NAME'].value_counts().head(8)
        sns.barplot(x=top_agencies.values, y=top_agencies.index, palette="rocket")
        plt.title("Top Fraud-Prone Agencies")
        plt.xlabel("Fraud Count")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_FOLDER, fraud_plot_filename))
        plt.close()

        # Save results
        result_path = os.path.join(RESULT_FOLDER, "fraud_report.csv")
        anomalies.to_csv(result_path, index=False)

        # Store for dashboard & results page
        results_data.update({
            "raw_preview": raw_preview_html,
            "preprocessing_summary": "\n\n".join(preprocessing_info),
            "eda_summary": "\n\n".join(eda_lines),
            "fraud_table": fraud_table_html,
            "eda_plot": eda_plot_filename,
            "fraud_plot": fraud_plot_filename,
            "fraud_count": fraud_count,
            "total_transactions": total_transactions,
            "fraud_percentage": fraud_percentage
        })

        return redirect('/results')

    except Exception as e:
        return f"Error while processing file: {e}"

@app.route('/results')
@login_required
def results():
    if not results_data:
        return "No results available. Please upload a dataset first.", 400

    return render_template("results.html",
                           raw_preview=results_data["raw_preview"],
                           preprocessing_summary=results_data["preprocessing_summary"],
                           eda_summary=results_data["eda_summary"],
                           eda_plot=results_data["eda_plot"],
                           fraud_plot=results_data["fraud_plot"],
                           fraud_count=results_data["fraud_count"],
                           total_transactions=results_data["total_transactions"],
                           fraud_percentage=results_data["fraud_percentage"],
                           table=results_data["fraud_table"])

@app.route('/download')
@login_required
def download():
    path = os.path.join(RESULT_FOLDER, "fraud_report.csv")
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 
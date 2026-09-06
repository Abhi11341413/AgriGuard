AgriGuard : Federated Edge AI for Crop Disease Detection

📖 Project Overview

AgriGuard is a privacy-first, offline-capable Edge AI architecture designed for rapid, on-device crop disease diagnosis. It evaluates high-resolution leaf images locally on the edge device to preserve farmer data privacy, utilizing an OpenCV HSV vision filter to reject invalid images and a lightweight MobileNetV2 architecture for inference. Rather than uploading heavy images to a centralized cloud, edge devices securely transmit mathematical weight updates to a Central Server for collaborative Federated Learning (FedAvg). Finally, it integrates a local XML document database with an Agentic LLM (Gemini) to provide hallucination-free, context-aware agricultural remedies.

🗂️ Repository Structure & Evaluation Documents

All required documentation for project evaluation can be found in the root directory:

Project_Description.md: Core overview of the agricultural problem statement and our decentralized solution.

Novelty_and_Experiments.md: Details our parameter-efficient MobileNetV2 approach, non-IID data distribution, and 20-round Federated Learning experiment.

AgriGuard_Report.docx: Full academic report including Objectives, System Description, Results, References, and Team Contributions.

Screenshots_and_Explanation.md: Visual breakdown of the Streamlit architecture, OpenCV validation, and XML database retrieval.

Demo_Video_Link.txt: Link to the full demonstration application video.

💻 Source Code Structure

The codebase is structured to separate the local edge logic from the federated cloud aggregator and the offline data resources:

AgriGuard_Submission/
├── data/                    <-- Distributed Non-IID Image Data (Simulation)
│   ├── client_abhishek/
│   ├── client_abid/
│   └── client_diwakar/
├── app.py                   <-- Streamlit Application (Edge UI & RAG Pipeline)
├── client.py                <-- Federated Edge Node Script
├── server.py                <-- Global Aggregator (Flower Server)
├── model.py                 <-- MobileNetV2 Edge AI Architecture
├── vision.py                <-- OpenCV HSV Validation Pipeline
├── setup_nosql.py           <-- Database Initialization Script
├── test_ai.py               <-- Inference Testing Script
├── agriguard_model.pth      <-- Aggregated FedAvg Global Model Weights
├── remedies.xml             <-- Offline XML Document Database
├── agri_data.json           <-- NoSQL Diagnostic Data mapping
└── requirements.txt         <-- Project Dependencies


🧠 A Note on agriguard_model.pth

Included in the root directory is agriguard_model.pth. This is the final, stabilized global model resulting from 20 rounds of Federated Learning across our simulated edge nodes. Because the full federated training process takes over 20 hours to converge on local hardware, these pre-trained weights are included so evaluators can instantly test the Streamlit UI, the OpenCV vision filter, and the XML document database retrieval without waiting for model compilation.

⚙️ Installation & Setup

Clone the repository and navigate to the project directory:

git clone <your-repo-link>
cd AgriGuard_Submission


Install the required dependencies:

pip install -r requirements.txt


Configure the Agentic AI (Gemini):
To test the AI Chatbot prescription feature, open app.py and insert a valid Gemini API key at the top of the file:

API_KEY = "YOUR_API_KEY_HERE"


How to Run the Project

Option A: Quick Evaluation (UI & RAG Testing)

If you only wish to evaluate the final Edge UI, the OpenCV Bouncer, and the XML document database integration, you do not need to run the federated servers.

streamlit run app.py


Upload a leaf image to test the diagnosis, or upload a picture of a face/desk to test the OpenCV Bouncer.

Option B: Full Federated Learning Simulation

To successfully demonstrate the decentralized Federated Learning loop, you must run the server and the clients simultaneously in separate terminals.

Terminal 1: Start the Global Aggregator (Central Server)

python server.py


(Wait for the server to indicate it is listening for minimum 3 connections).

Terminals 2, 3, and 4: Start the Edge Nodes
Open three separate terminals and run the clients with their isolated data directories to simulate different farmers:

# Terminal 2
python client.py data/client_abhishek

# Terminal 3
python client.py data/client_diwakar

# Terminal 4
python client.py data/client_abid


Watch Terminal 1 as the server aggregates the mathematical weight updates (FedAvg) over the training rounds without ever accessing the raw images in the data/ folders!


## 👥 Team
Developed by students at NIT Trichy:
* Abhishek
* Diwakar
* Abid" > README.md

git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Abhi11341413/Federated-Learning-based-Agriguard.git
git push -u origin main
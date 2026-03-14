AgriGuard 🌱

**Privacy-Preserving Crop Disease Diagnosis using Edge AI and Federated Learning**

AgriGuard is an offline-first, decentralized Artificial Intelligence system designed to diagnose crop diseases (specifically Early Blight in Tomatoes and Potatoes) for rural farmers. By utilizing **Federated Learning**, the model trains locally on edge devices, ensuring user data privacy and drastically reducing cloud bandwidth requirements.

## 🚀 Key Features
* **Federated Learning Network:** Utilizes the Flower (\`flwr\`) framework to train a global model without raw images ever leaving the farmer's local device.
* **Custom CNN Architecture:** Powered by \`SimpleLeafNet\`, a lightweight Convolutional Neural Network built from scratch in **PyTorch**, optimized for low-latency edge computing.
* **Offline-First Architecture:** Diagnoses are mapped to a local **XML document database** (\`remedies.xml\`). This allows instant retrieval of organic and chemical treatments entirely offline, with no internet ping required.
* **Interactive UI:** A clean, responsive frontend built with **Streamlit** for easy image uploading and real-time diagnosis.

## 🛠️ Technology Stack
* **Deep Learning:** PyTorch, Torchvision
* **Federated Learning:** Flower (\`flwr\`)
* **Frontend:** Streamlit
* **Data Storage:** XML Document Database

## ⚙️ How to Run the Project

**1. Start the Central Server:**
Initialize the global model aggregation server.

python server.py


**2. Start the Edge Clients (Open separate terminals):**
Simulate the federated edge devices training on local data.

python client.py data/client_abhishek
python client.py data/client_diwakar
python client.py data/client_abid


**3. Launch the Application UI:**
Run the offline mobile simulator to diagnose a new leaf image.

streamlit run app.py


## 👥 Team
Developed by students at NIT Trichy:
* Abhishek
* Diwakar
* Abid" > README.md

git init
git add .
git commit -m "Initial commit: AgriGuard V1 MVP with PyTorch and XML DB"
git branch -M main
git remote add origin https://github.com/Abhi11341413/AgriGuard.git
git push -u origin main
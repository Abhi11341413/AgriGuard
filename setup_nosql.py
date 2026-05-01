from tinydb import TinyDB

db = TinyDB('agri_data.json')
db.truncate() # Clear old data

remedies = [
    {"disease_id": 0, "disease_name": "Pepper Bell - Bacterial Spot", "organic_treatment": "Remove infected leaves immediately. Avoid overhead watering.", "chemical_treatment": "Apply a copper-based bactericide spray early in the season."},
    {"disease_id": 1, "disease_name": "Healthy", "organic_treatment": "Maintain optimal watering. Ensure good soil drainage.", "chemical_treatment": "None required."},
    {"disease_id": 2, "disease_name": "Potato - Early Blight", "organic_treatment": "Ensure proper spacing for air circulation. Apply Bacillus subtilis biofungicide.", "chemical_treatment": "Apply Chlorothalonil 75% WP."},
    {"disease_id": 3, "disease_name": "Potato - Late Blight", "organic_treatment": "Destroy infected plants immediately. Do not compost.", "chemical_treatment": "Apply Mancozeb or Copper fungicide immediately."},
    {"disease_id": 4, "disease_name": "Healthy", "organic_treatment": "Maintain optimal watering. Ensure good soil drainage.", "chemical_treatment": "None required."},
    {"disease_id": 5, "disease_name": "Tomato - Bacterial Spot", "organic_treatment": "Prune for airflow. Use drip irrigation instead of sprinklers.", "chemical_treatment": "Apply Copper fungicide mixed with Mancozeb."},
    {"disease_id": 6, "disease_name": "Tomato - Early Blight", "organic_treatment": "Prune infected lower leaves. Apply Neem oil extract.", "chemical_treatment": "Apply Copper Hydroxide fungicide."},
    {"disease_id": 7, "disease_name": "Tomato - Late Blight", "organic_treatment": "Pull and destroy infected plants. Keep foliage dry.", "chemical_treatment": "Apply protective fungicides like Chlorothalonil."},
    {"disease_id": 8, "disease_name": "Tomato - Leaf Mold", "organic_treatment": "Increase greenhouse ventilation and reduce humidity.", "chemical_treatment": "Apply Calcium polysulfide or Copper spray."},
    {"disease_id": 9, "disease_name": "Tomato - Septoria Leaf Spot", "organic_treatment": "Remove infected leaves. Mulch around the base to prevent soil splashing.", "chemical_treatment": "Apply fungicidal sprays containing Chlorothalonil."},
    {"disease_id": 10, "disease_name": "Tomato - Spider Mites", "organic_treatment": "Spray underside of leaves with strong water stream. Apply Insecticidal Soap.", "chemical_treatment": "Apply Miticides (e.g., Abamectin)."},
    {"disease_id": 11, "disease_name": "Tomato - Target Spot", "organic_treatment": "Improve airflow by pruning. Avoid high nitrogen fertilizers.", "chemical_treatment": "Apply targeted fungicides like Azoxystrobin."},
    {"disease_id": 12, "disease_name": "Tomato - Yellow Leaf Curl Virus", "organic_treatment": "Use reflective mulches to repel whiteflies. Remove infected plants.", "chemical_treatment": "Apply insecticides to control the whitefly population (the virus vector)."},
    {"disease_id": 13, "disease_name": "Tomato - Mosaic Virus", "organic_treatment": "There is no cure. Uproot and burn the plant. Wash hands and tools thoroughly.", "chemical_treatment": "None. Prevention through sanitation is the only method."},
    {"disease_id": 14, "disease_name": "Healthy", "organic_treatment": "Maintain current nutrient schedule. Monitor weekly.", "chemical_treatment": "None required."}
]

db.insert_multiple(remedies)
print("✅ NGDB Document Database successfully created with 15 classes: agri_data.json")
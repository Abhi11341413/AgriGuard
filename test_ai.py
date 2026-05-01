import google.generativeai as genai

# Your exact API key
genai.configure(api_key="AIzaSyDgZE0JoqOZXkpWmc_Lgn4274fd05wPc-Y")

print("Checking Google Servers for available models...\n")

try:
    for m in genai.list_models():
        # We only want models that can write text (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    print("\n✅ Test complete. Copy one of the exact names above!")
except Exception as e:
    print(f"🚨 API Key Error: {e}")
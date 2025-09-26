import os
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import random
import pickle
import time
import threading

# -------------------- SPEAK FUNCTION --------------------
def speak(text):
    """Speak text reliably without cutting off."""
    print("\n[AI DERMA]:", text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.2)
    engine.stop()

# -------------------- LISTEN FUNCTION --------------------
def listen(timeout=5, phrase_time_limit=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speak("Listening...")

        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please try again.")
            return None

    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't understand that. Please try again.")
        return None
    except sr.RequestError as e:
        speak(f"Speech recognition error: {e}")
        return None

# -------------------- TEXT FALLBACK --------------------
def get_text_input(prompt):
    speak(prompt)
    return input("Type your response: ").strip().lower()

# -------------------- LOAD MODEL --------------------
def load_trained_model(model_path='skin_model.pkl', scaler_path='scaler.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Trained model loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        print("❌ Trained model not found! Using fallback dataset method.")
        return None, None

# -------------------- PREDICT SKIN --------------------
def predict_skin_type(features, model=None, scaler=None):
    if model is not None and scaler is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        confidence = np.max(probability)
        print(f"Model prediction: {prediction} (confidence: {confidence:.2%})")
        return prediction
    else:
        return compare_with_original_dataset(features)

def compare_with_original_dataset(features):
    SKIN_DATASET = {
        'Oily': [[130, 12, 15, 350, 18, 20, 1.5],[125, 10, 14, 320, 22, 25, 1.2],[135, 13, 16, 380, 20, 18, 1.8]],
        'Dry': [[110, 8, 12, 280, 8, 3, 3.5],[105, 7, 11, 250, 6, 2, 4.0],[115, 9, 13, 300, 9, 4, 3.0]],
        'Combination': [[120, 10, 14, 320, 16, 12, 2.5],[125, 11, 15, 340, 18, 14, 2.0],[118, 9, 13, 310, 17, 13, 2.8]],
        'Normal': [[140, 9, 14, 400, 10, 8, 1.0],[145, 8, 15, 420, 9, 7, 0.8],[138, 10, 13, 380, 11, 9, 1.2]]
    }
    min_distance, closest_type = float('inf'), 'Normal'
    for skin_type, data_points in SKIN_DATASET.items():
        for point in data_points:
            distance = np.linalg.norm(np.array(features) - np.array(point))
            if distance < min_distance:
                min_distance, closest_type = distance, skin_type
    return closest_type

# -------------------- ANALYSIS --------------------
def analyze_skin(image, model=None, scaler=None):
    img = cv2.resize(image, (500, 500))
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    mean_L, mean_a, mean_b = np.mean(L), np.mean(a), np.mean(b)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # pores
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    pore_percentage = np.sum(cleaned == 255) / cleaned.size * 100

    # sebum
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sebum_mask = cv2.inRange(hsv, (0, 0, 200), (30, 80, 255))
    sebum_percentage = np.sum(sebum_mask == 255) / sebum_mask.size * 100

    # wrinkles
    edges = cv2.Canny(gray, 50, 150)
    wrinkle_factor = np.sum(edges > 0) / edges.size * 100

    features = [mean_L, mean_a, mean_b, texture_score,
                pore_percentage, sebum_percentage, wrinkle_factor]

    skin_type = predict_skin_type(features, model, scaler)

    # ---- Age estimation ----
    age_factor = (
        0.3 * (texture_score / 500) +
        0.25 * (pore_percentage / 30) +
        0.2 * (wrinkle_factor / 5) +
        0.15 * (mean_L / 255) +
        0.1 * (sebum_percentage / 20)
    )
    if age_factor < 0.25:
        age_est, age_conf = "18-25 years", "High confidence"
    elif age_factor < 0.4:
        age_est, age_conf = "25-35 years", "Medium confidence"
    elif age_factor < 0.6:
        age_est, age_conf = "35-45 years", "Medium confidence"
    else:
        age_est, age_conf = "45+ years", "High confidence"

    # ---- Concerns ----
    concerns = []
    if pore_percentage > 20: concerns.append("Large pores")
    elif pore_percentage < 5: concerns.append("Clogged pores")
    if sebum_percentage > 15: concerns.append("Excess oil")
    elif sebum_percentage < 5: concerns.append("Lack of moisture")
    if mean_L < 120: concerns.append("Dullness")
    if mean_L > 180: concerns.append("Hyperpigmentation")
    if wrinkle_factor > 3: concerns.append("Visible wrinkles")
    if texture_score < 200: concerns.append("Uneven texture")
    if not concerns: concerns.append("No major concerns")

    # ---- Recommendations ----
    recommendations = {
        "Oily": {"Cleansers": ["Foaming cleanser","Gel-based cleanser"],"Toners": ["Witch hazel toner","Niacinamide toner"],
                 "Serums": ["Niacinamide serum","Salicylic acid serum"],"Moisturizers": ["Oil-free moisturizer","Gel moisturizer"],
                 "Sunscreen": ["Matte sunscreen","Oil-free sunscreen"]},
        "Dry": {"Cleansers": ["Cream cleanser","Oil cleanser"],"Toners": ["Hydrating toner","HA toner"],
                "Serums": ["Hyaluronic acid serum","Ceramide serum"],"Moisturizers": ["Rich cream","Barrier repair cream"],
                "Sunscreen": ["Moisturizing sunscreen","Cream sunscreen"]},
        "Combination": {"Cleansers": ["Balancing cleanser","Low-pH cleanser"],"Toners": ["pH-balancing toner","Rose water toner"],
                        "Serums": ["Vitamin C serum","Niacinamide serum"],"Moisturizers": ["Lightweight lotion","Gel-cream hybrid"],
                        "Sunscreen": ["Non-comedogenic sunscreen","Mineral sunscreen"]},
        "Normal": {"Cleansers": ["Gentle cleanser","Micellar water"],"Toners": ["Hydrating mist","Antioxidant toner"],
                   "Serums": ["Antioxidant serum","Vitamin E serum"],"Moisturizers": ["Lightweight lotion","Gel moisturizer"],
                   "Sunscreen": ["Broad spectrum sunscreen","Tinted sunscreen"]}
    }

    skin_recs = recommendations.get(skin_type, recommendations["Normal"])
    personalized_recs = [
        f"Cleanser: {random.choice(skin_recs['Cleansers'])}",
        f"Toner: {random.choice(skin_recs['Toners'])}",
        f"Serum: {random.choice(skin_recs['Serums'])}",
        f"Moisturizer: {random.choice(skin_recs['Moisturizers'])}",
        f"Sunscreen: {random.choice(skin_recs['Sunscreen'])}"
    ]

    return {
        'skin_type': skin_type,
        'age_estimate': age_est,
        'age_confidence': age_conf,
        'brightness': mean_L,
        'redness': mean_a,
        'yellowness': mean_b,
        'texture': texture_score,
        'pores': pore_percentage,
        'sebum': sebum_percentage,
        'wrinkles': wrinkle_factor,
        'concerns': concerns,
        'recommendations': personalized_recs,
        'detailed_analysis': {
            'brightness_level': 'High' if mean_L > 150 else 'Medium' if mean_L > 120 else 'Low',
            'redness_level': 'High' if mean_a > 15 else 'Medium' if mean_a > 10 else 'Low',
            'texture_quality': 'Smooth' if texture_score > 350 else 'Normal' if texture_score > 250 else 'Rough',
            'pore_size': 'Large' if pore_percentage > 18 else 'Medium' if pore_percentage > 10 else 'Small',
            'oil_level': 'High' if sebum_percentage > 15 else 'Medium' if sebum_percentage > 8 else 'Low'
        }
    }

# -------------------- IMAGE INPUT --------------------
capture_signal = False
def voice_listener():
    global capture_signal
    while True:
        command = listen(timeout=3, phrase_time_limit=3)
        if command and "capture" in command:
            capture_signal = True
            break

def get_image():
    while True:
        speak("Say: 'upload image', 'default image', or 'use webcam'")
        command = listen()
        if command is None:
            choice = input("Choose option (1: upload, 2: default, 3: webcam): ").strip()
            command = "upload" if choice == "1" else "default" if choice == "2" else "webcam"

        if "upload" in command:
            speak("Please type the path of your image:")
            path = input("Enter image path: ").strip()
            if os.path.exists(path):
                return cv2.imread(path)
            else:
                speak("File not found. Try again.")

        elif "default" in command:
            img = np.full((500, 500, 3), 200, dtype=np.uint8)
            cv2.putText(img, "DEFAULT IMAGE", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return img

        elif "webcam" in command:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                speak("Could not access the webcam")
                return None

            speak("Say 'capture' when ready.")
            global capture_signal
            capture_signal = False

            listener_thread = threading.Thread(target=voice_listener, daemon=True)
            listener_thread.start()

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.imshow("Webcam - Press ESC to exit", frame)

                if capture_signal:
                    speak("Image captured successfully!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            return None

        else:
            speak("Command not recognized. Try again.")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    speak("Welcome to the Enhanced Voice Enabled Skin Analyzer")

    model, scaler = load_trained_model()

    speak("Please say your name")
    name, attempts = None, 0
    while name is None and attempts < 3:
        name = listen()
        attempts += 1

    if name is None:
        name = get_text_input("Voice recognition failed. Please type your name:")

    speak(f"Hello {name}, let's begin your comprehensive skin analysis.")

    image = get_image()
    if image is None:
        speak("No image was captured. Exiting.")
        exit()

    speak("Analyzing your skin now with enhanced AI algorithms...")
    results = analyze_skin(image, model, scaler)

    # ---- Speak results ----
    speak(f"Skin type detected: {results['skin_type']}")
    speak(f"Estimated age range: {results['age_estimate']} with {results['age_confidence']}")

    speak("Detailed skin analysis:")
    for key, value in results['detailed_analysis'].items():
        speak(f"{key.replace('_',' ').capitalize()}: {value}")

    speak("Detected skin concerns: " + ", ".join(results['concerns']))

    speak("Here are your personalized skincare recommendations:")
    for i, r in enumerate(results['recommendations'], 1):
        speak(f"Recommendation {i}: {r}")
        time.sleep(0.3)

    speak("Thank you for using the enhanced AI skin analyzer. Have a beautiful day!")

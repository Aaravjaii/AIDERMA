import os
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import random

# Initialize text-to-speech
engine = pyttsx3.init()
def speak(text):
    print("\n[Assistant]:", text)
    engine.say(text)
    engine.runAndWait()

# Initialize speech recognizer
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't understand that. Please try again.")
        return None

# Skin analysis dataset (small but effective)
SKIN_DATASET = {
    # Format: [mean_L, mean_a, mean_b, texture_score, pore_percentage, sebum_percentage, wrinkle_factor]
    'Oily': [
        [130, 12, 15, 350, 18, 20, 1.5],
        [125, 10, 14, 320, 22, 25, 1.2],
        [135, 13, 16, 380, 20, 18, 1.8]
    ],
    'Dry': [
        [110, 8, 12, 280, 8, 3, 3.5],
        [105, 7, 11, 250, 6, 2, 4.0],
        [115, 9, 13, 300, 9, 4, 3.0]
    ],
    'Combination': [
        [120, 10, 14, 320, 16, 12, 2.5],
        [125, 11, 15, 340, 18, 14, 2.0],
        [118, 9, 13, 310, 17, 13, 2.8]
    ],
    'Normal': [
        [140, 9, 14, 400, 10, 8, 1.0],
        [145, 8, 15, 420, 9, 7, 0.8],
        [138, 10, 13, 380, 11, 9, 1.2]
    ]
}

def compare_with_dataset(features):
    min_distance = float('inf')
    closest_type = 'Normal'
    
    for skin_type, data_points in SKIN_DATASET.items():
        for point in data_points:
            distance = np.linalg.norm(np.array(features) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                closest_type = skin_type
    return closest_type

def analyze_skin(image):
    img = cv2.resize(image, (500, 500))
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    mean_L = np.mean(L)
    mean_a = np.mean(a)
    mean_b = np.mean(b)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    pore_percentage = np.sum(cleaned == 255) / cleaned.size * 100

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sebum_mask = cv2.inRange(hsv, (0, 0, 200), (30, 80, 255))
    sebum_percentage = np.sum(sebum_mask == 255) / sebum_mask.size * 100

    edges = cv2.Canny(gray, 50, 150)
    wrinkle_factor = np.sum(edges > 0) / edges.size * 100

    # Compare with dataset for more accurate skin type detection
    features = [mean_L, mean_a, mean_b, texture_score, pore_percentage, sebum_percentage, wrinkle_factor]
    skin_type = compare_with_dataset(features)

    age_factor = (
        0.3 * (texture_score / 500) +
        0.25 * (pore_percentage / 30) +
        0.2 * (wrinkle_factor / 5) +
        0.15 * (mean_L / 255) +
        0.1 * (sebum_percentage / 20)
    )

    if age_factor < 0.25:
        age_est = "18-25 years"
        age_conf = "High confidence"
    elif age_factor < 0.4:
        age_est = "25-35 years"
        age_conf = "Medium confidence"
    elif age_factor < 0.6:
        age_est = "35-45 years"
        age_conf = "Medium confidence"
    else:
        age_est = "45+ years"
        age_conf = "High confidence"

    concerns = []
    if pore_percentage > 20:
        concerns.append("Large pores")
    elif pore_percentage < 5:
        concerns.append("Clogged pores")
    if sebum_percentage > 15:
        concerns.append("Excess oil")
    elif sebum_percentage < 5:
        concerns.append("Lack of moisture")
    if mean_L < 120:
        concerns.append("Dullness")
    if mean_L > 180:
        concerns.append("Hyperpigmentation")
    if wrinkle_factor > 3:
        concerns.append("Visible wrinkles")
    if texture_score < 200:
        concerns.append("Uneven texture")
    if not concerns:
        concerns.append("No major concerns")

    # Expanded recommendations database
    recommendations = {
        "Oily": {
            "Cleansers": ["Foaming cleanser", "Gel-based cleanser", "Salicylic acid cleanser"],
            "Toners": ["Witch hazel toner", "Niacinamide toner", "Tea tree toner"],
            "Serums": ["Niacinamide serum", "Salicylic acid serum", "Zinc serum"],
            "Moisturizers": ["Oil-free moisturizer", "Gel moisturizer", "Water-based moisturizer"],
            "Sunscreen": ["Matte sunscreen", "Oil-free sunscreen", "Mineral sunscreen"],
            "Treatments": ["Clay mask 2-3x/week", "BHA exfoliant 2x/week", "Oil-control primer"]
        },
        "Dry": {
            "Cleansers": ["Cream cleanser", "Oil cleanser", "Milky cleanser"],
            "Toners": ["Hydrating toner", "HA toner", "Essence toner"],
            "Serums": ["Hyaluronic acid serum", "Ceramide serum", "Squalane serum"],
            "Moisturizers": ["Rich cream", "Barrier repair cream", "Occlusive moisturizer"],
            "Sunscreen": ["Moisturizing sunscreen", "Cream sunscreen", "Tinted sunscreen"],
            "Treatments": ["Sleeping mask", "Facial oil", "Gentle exfoliation 1x/week"]
        },
        "Combination": {
            "Cleansers": ["Balancing cleanser", "BHA cleanser", "Low-pH cleanser"],
            "Toners": ["pH-balancing toner", "Centella toner", "Rose water toner"],
            "Serums": ["Vitamin C serum", "Niacinamide serum", "Peptide serum"],
            "Moisturizers": ["Lightweight lotion (oily zones)", "Cream (dry zones)", "Gel-cream hybrid"],
            "Sunscreen": ["Non-comedogenic sunscreen", "Lightweight sunscreen", "Mineral sunscreen"],
            "Treatments": ["Clay mask on T-zone", "Hydrating mask on cheeks", "AHA exfoliant 1-2x/week"]
        },
        "Normal": {
            "Cleansers": ["Gentle cleanser", "Micellar water", "Milk cleanser"],
            "Toners": ["Hydrating mist", "Antioxidant toner", "Soothing toner"],
            "Serums": ["Antioxidant serum", "Vitamin E serum", "Ferulic acid serum"],
            "Moisturizers": ["Lightweight lotion", "Gel moisturizer", "Balancing cream"],
            "Sunscreen": ["Broad spectrum sunscreen", "Lightweight sunscreen", "Tinted sunscreen"],
            "Treatments": ["Weekly exfoliation", "Sheet masks", "Facial massage"]
        }
    }

    # Select specific recommendations based on concerns
    personalized_recommendations = []
    skin_recs = recommendations[skin_type]
    
    # Always include basics
    personalized_recommendations.extend([
        f"Cleanser: {random.choice(skin_recs['Cleansers'])}",
        f"Toner: {random.choice(skin_recs['Toners'])}",
        f"Serum: {random.choice(skin_recs['Serums'])}",
        f"Moisturizer: {random.choice(skin_recs['Moisturizers'])}",
        f"Sunscreen: {random.choice(skin_recs['Sunscreen'])}"
    ])
    
    # Add targeted treatments based on concerns
    if "Large pores" in concerns:
        personalized_recommendations.append("Treatment: " + random.choice([
            "Niacinamide treatment", "Clay mask", "Pore-minimizing primer"
        ]))
    if "Excess oil" in concerns:
        personalized_recommendations.append("Treatment: " + random.choice([
            "Oil-control sheets", "Mattifying moisturizer", "Salicylic acid spot treatment"
        ]))
    if "Dullness" in concerns:
        personalized_recommendations.append("Treatment: " + random.choice([
            "Vitamin C booster", "Exfoliating treatment", "Brightening mask"
        ]))
    if "Visible wrinkles" in concerns:
        personalized_recommendations.append("Treatment: " + random.choice([
            "Retinol treatment", "Peptide serum", "Hyaluronic acid booster"
        ]))
    if "Uneven texture" in concerns:
        personalized_recommendations.append("Treatment: " + random.choice([
            "AHA/BHA exfoliant", "Microdermabrasion scrub", "Resurfacing treatment"
        ]))

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
        'recommendations': personalized_recommendations,
        'detailed_analysis': {
            'brightness_level': 'High' if mean_L > 150 else 'Medium' if mean_L > 120 else 'Low',
            'redness_level': 'High' if mean_a > 15 else 'Medium' if mean_a > 10 else 'Low',
            'texture_quality': 'Smooth' if texture_score > 350 else 'Normal' if texture_score > 250 else 'Rough',
            'pore_size': 'Large' if pore_percentage > 18 else 'Medium' if pore_percentage > 10 else 'Small',
            'oil_level': 'High' if sebum_percentage > 15 else 'Medium' if sebum_percentage > 8 else 'Low'
        }
    }

def get_image():
    while True:
        speak("Say one of the following: 'upload image', 'default image', or 'use webcam'")
        command = listen()
        if command is None:
            continue
        if "upload" in command:
            speak("Please type the path of your image:")
            path = input("Enter image path: ").strip()
            if os.path.exists(path):
                return cv2.imread(path)
            else:
                speak("File not found. Try again.")
        elif "default" in command:
            img = np.full((500, 500, 3), 200, dtype=np.uint8)
            cv2.putText(img, "DEFAULT IMAGE", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return img
        elif "webcam" in command:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                speak("Could not access the webcam")
                return None
            speak("Position your face and say 'capture' to take a photo")
            while True:
                ret, frame = cap.read()
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                command = listen()
                if command and "capture" in command:
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame
            cap.release()
            cv2.destroyAllWindows()
            return None
        else:
            speak("Command not recognized. Try again.")

if __name__ == "__main__":
    speak("Welcome to the Enhanced Voice Enabled Skin Analyzer")
    speak("Please say your name")
    name = None
    while name is None:
        name = listen()
    speak(f"Hello {name}, let's begin your comprehensive skin analysis.")

    image = get_image()
    if image is None:
        speak("No image was captured. Exiting.")
        exit()

    speak("Analyzing your skin now with enhanced algorithms...")
    results = analyze_skin(image)

    speak(f"Skin type detected: {results['skin_type']}")
    speak(f"Estimated age range: {results['age_estimate']} with {results['age_confidence']}")
    
    # Detailed analysis
    speak("Detailed skin analysis:")
    speak(f"Brightness level: {results['detailed_analysis']['brightness_level']}")
    speak(f"Redness level: {results['detailed_analysis']['redness_level']}")
    speak(f"Texture quality: {results['detailed_analysis']['texture_quality']}")
    speak(f"Pore size: {results['detailed_analysis']['pore_size']}")
    speak(f"Oil level: {results['detailed_analysis']['oil_level']}")
    
    speak("Detected skin concerns: " + ", ".join(results['concerns']))
    speak("Here are your personalized skincare recommendations:")
    for i, r in enumerate(results['recommendations'], 1):
        speak(f"{i}. {r}")

    speak("Thank you for using the enhanced skin analyzer. Have a beautiful day!")
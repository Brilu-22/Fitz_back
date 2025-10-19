import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, auth, firestore

import requests # For Edamam API
import google.generativeai as genai # For Google Gemini
import spotipy # For Spotify Web API
from spotipy.oauth2 import SpotifyClientCredentials # For client credentials flow


# --- 0. Load Environment Variables ---
# This must be called before accessing any os.getenv() calls
load_dotenv("fitz_backend/fitz.env")

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="AI Fitness Music App API",
    description="API for generating personalized workout plans, diet suggestions, and music playlists using AI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# --- 2. Firebase Initialization ---
# Retrieve the path to the Firebase service account key
FIREBASE_PRIVATE_KEY_PATH = os.getenv('FIREBASE_PRIVATE_KEY_PATH')

if not FIREBASE_PRIVATE_KEY_PATH:
    raise RuntimeError("FIREBASE_PRIVATE_KEY_PATH environment variable not set.")
if not os.path.exists(FIREBASE_PRIVATE_KEY_PATH):
    raise FileNotFoundError(f"Firebase private key file not found at: {FIREBASE_PRIVATE_KEY_PATH}")

try:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(FIREBASE_PRIVATE_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client() # Firestore client for database operations
    print("Firebase initialized successfully.")
except Exception as e:
    # Log the error and raise a RuntimeError to prevent the app from starting with a broken dependency
    print(f"ERROR: Failed to initialize Firebase: {e}")
    raise RuntimeError(f"Failed to initialize Firebase: {e}. Check FIREBASE_PRIVATE_KEY_PATH.")


# --- 3. API Key Configuration & Service Initialization ---

# Edamam
EDAMAM_APP_ID = os.getenv('EDAMAM_APP_ID')
EDAMAM_APP_KEY = os.getenv('EDAMAM_APP_KEY')
if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
    print("WARNING: Edamam API keys (EDAMAM_APP_ID, EDAMAM_APP_KEY) are missing. Nutrition analysis may not work.")

# Google Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("WARNING: Google Gemini API key (GEMINI_API_KEY) is missing. AI generation may not work.")
    gemini_model = None # Set to None if key is missing
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("Google Gemini model initialized.")

# Spotify
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
spotify = None # Initialize to None
if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
    print("WARNING: Spotify API keys (SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET) are missing. Spotify integration may not work.")
else:
    try:
        # Initialize Spotify using Client Credentials Flow for server-side access
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET
        )
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        print("Spotify client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Spotify: {e}. Check SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env")
        # spotify remains None if initialization fails


# --- 4. Pydantic Models for Request Body Validation ---

class RegisterRequest(BaseModel):
    email: str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6, example="securepassword123")
    name: str = Field(..., example="Jane Doe")

class LoginRequest(BaseModel):
    email: str = Field(..., example="user@example.com")
    password: str = Field(..., example="securepassword123")

class GeneratePlanRequest(BaseModel):
    user_uid: str = Field(..., example="someFirebaseUserUID")
    current_weight: float = Field(..., gt=0, example=70.5)
    target_weight: float = Field(..., gt=0, example=65.0)
    duration_weeks: int = Field(..., gt=0, le=52, example=8) # 1 to 52 weeks
    music_genre: str = Field(..., example="Electronic Dance Music")
    workout_intensity: Optional[str] = Field("moderate", example="high-energy") # Added for more specific music search

class NutritionIngredient(BaseModel):
    ingredient: str = Field(..., example="1 cup cooked rice")

# --- 5. API Endpoints ---

@app.get("/", summary="API Root / Health Check")
async def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"message": "AI Fitness Music App Backend is running!"}

@app.post("/register", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register_user(request_data: RegisterRequest):
    """
    Registers a new user with Firebase Authentication and stores basic profile in Firestore.
    """
    try:
        user = auth.create_user(email=request_data.email, password=request_data.password)
        await db.collection('users').document(user.uid).set({
            'name': request_data.name,
            'email': request_data.email,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        return {"message": "User registered successfully", "uid": user.uid}
    except Exception as e:
        if 'email-already-exists' in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered. Please use a different email or log in."
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration failed: {e}")

@app.post("/login", response_model=Dict[str, str], summary="Login (Firebase Admin SDK Verification)")
async def login_user(request_data: LoginRequest):
    """
    Simulates user login for backend verification.
    NOTE: For actual client-side login, use Firebase client SDKs and verify ID tokens on the backend.
    """
    try:
        # This only verifies if a user with the email exists.
        # It does NOT authenticate password via Admin SDK.
        # Real client apps should use Firebase client SDK for email/password auth
        # and then send the ID token to the backend for verification.
        user = auth.get_user_by_email(request_data.email)
        # In a real system, you'd likely generate a session cookie or a JWT for the client here
        # after verifying a Firebase ID token sent from the client.
        return {"message": "User found (client handles actual authentication)", "uid": user.uid, "email": user.email}
    except auth.UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found with this email.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Login verification failed: {e}")

@app.post("/generate_plan", response_model=Dict[str, Any], summary="Generate personalized fitness and music plan")
async def generate_plan(request_data: GeneratePlanRequest):
    """
    Generates a comprehensive fitness plan, diet suggestions, and Spotify playlist recommendations
    based on user goals and preferences using Google Gemini and Spotify APIs.
    """
    if not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model (Gemini) is not initialized. Please check GEMINI_API_KEY."
        )

    try:
        # --- Generate Workout Plan using Google Gemini ---
        workout_prompt = f"""
        You are an expert fitness coach. Create a detailed 1-week workout plan for a user aiming to go from {request_data.current_weight} kg to {request_data.target_weight} kg in {request_data.duration_weeks} weeks.
        The plan should include both cardio and strength training.
        For each exercise, provide:
        - Exercise Name
        - A very brief description (1 sentence)
        - Sets and Reps (e.g., 3 sets of 10-12 reps)
        - A short instruction on how to perform it safely.
        Include rest days. Also, specifically suggest 3 variations or progressions for glute bridge exercises.
        Format the plan clearly, day by day, using Markdown headings for readability.
        """
        gemini_workout_response = gemini_model.generate_content(workout_prompt)
        workout_plan = gemini_workout_response.text

        # --- Generate Diet Plan using Google Gemini ---
        diet_initial_prompt = f"""
        You are an expert nutritionist. Based on a user's goal to go from {request_data.current_weight} kg to {request_data.target_weight} kg in {request_data.duration_weeks} weeks,
        provide a general overview of recommended daily caloric intake and a balanced macronutrient breakdown (proteins, carbs, fats).
        Then, suggest a 3-day sample meal plan, including Breakfast, Lunch, Dinner, and 2 Snacks per day.
        For each meal, suggest specific healthy options.
        Include advice on water intake and general healthy eating habits.
        Format the diet plan clearly, day by day, using Markdown headings for readability.
        """
        gemini_diet_response = gemini_model.generate_content(diet_initial_prompt)
        diet_general_info = gemini_diet_response.text

        # --- Generate Music Playlist using Spotify ---
        playlist_suggestions: List[Dict[str, Any]] = []
        if spotify:
            try:
                # Use workout_intensity in the query for better results
                query = f"{request_data.music_genre} {request_data.workout_intensity} workout music"
                results = spotify.search(q=query, type='playlist', limit=5)

                for playlist in results.get('playlists', {}).get('items', []):
                    playlist_suggestions.append({
                        "name": playlist['name'],
                        "description": playlist.get('description', 'No description available'),
                        "spotify_url": playlist['external_urls']['spotify'],
                        "image_url": playlist['images'][0]['url'] if playlist['images'] else None,
                        "owner": playlist['owner']['display_name'],
                        "tracks_count": playlist['tracks']['total']
                    })

            except spotipy.exceptions.SpotifyException as se:
                print(f"Spotify API error during search: {se}")
                playlist_suggestions.append({"error": f"Spotify API error: {se}. Check credentials or network."})
            except Exception as se:
                print(f"Unexpected error during Spotify search: {se}")
                playlist_suggestions.append({"error": f"An unexpected Spotify error occurred: {se}"})
        else:
            playlist_suggestions.append({"error": "Spotify API not initialized. Check server logs for credential errors."})


        # --- Store the generated plan in Firestore ---
        await db.collection('users').document(request_data.user_uid).collection('plans').add({
            'current_weight': request_data.current_weight,
            'target_weight': request_data.target_weight,
            'duration_weeks': request_data.duration_weeks,
            'music_genre': request_data.music_genre,
            'workout_intensity': request_data.workout_intensity,
            'workout_plan': workout_plan,
            'diet_general_info': diet_general_info,
            'music_playlist_suggestions': playlist_suggestions,
            'generated_at': firestore.SERVER_TIMESTAMP
        })

        return {
            "message": "Plan generated and saved successfully",
            "workout_plan": workout_plan,
            "diet_plan_summary": diet_general_info,
            "music_playlist_suggestions": playlist_suggestions
        }

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        print(f"ERROR: Failed to generate plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred while generating the plan: {str(e)}"
        )

@app.post("/analyze_nutrition", summary="Analyze nutrition for specific ingredients using Edamam")
async def analyze_nutrition(ingredients: List[NutritionIngredient]):
    """
    Analyzes the nutritional content of a list of ingredients using the Edamam Nutrition Analysis API.
    """
    if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Edamam API keys are not configured. Nutrition analysis is unavailable."
        )

    edamam_nutrition_url = f"https://api.edamam.com/api/nutrition-details"
    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # Convert list of Pydantic models to list of strings
        ingredient_strings = [item.ingredient for item in ingredients]
        response = requests.post(edamam_nutrition_url, json={"ingredients": ingredient_strings}, params=params, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Edamam API request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Edamam API service is currently unavailable or returned an error: {e}"
        )
    except Exception as e:
        print(f"ERROR: Unexpected error during nutrition analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during nutrition analysis: {str(e)}"
        )

# --- 6. Run the Application ---
# To run this file:
# cd ai_fitness_music_app/backend
# source venv/bin/activate
# uvicorn main:app --reload --port 8000
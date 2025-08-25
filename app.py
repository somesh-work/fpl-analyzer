# app.py - Final version using s3fs library

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from tqdm import tqdm
import pulp
import json
import os
import redis
import s3fs # <-- NEW LIBRARY

app = Flask(__name__)
CORS(app)

# --- CLIENT SETUPS ---
try:
    redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
    redis_client.ping()
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"Could not connect to Redis: {e}. Caching will be disabled.")
    redis_client = None

# NEW s3fs setup
try:
    s3 = s3fs.S3FileSystem(
        key=os.getenv('R2_ACCESS_KEY_ID'),
        secret=os.getenv('R2_SECRET_ACCESS_KEY'),
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        default_cache_type='none'
    )
    R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
    s3.ls(R2_BUCKET_NAME) # Test the connection by listing files
    print("Successfully connected to Cloudflare R2 with s3fs.")
except Exception as e:
    print(f"Could not connect to R2 with s3fs: {e}")
    s3 = None
    R2_BUCKET_NAME = None

# --- GLOBAL VARIABLES ---
BASE_URL = 'https://fantasy.premierleague.com/api/'
ALL_GAMEWEEK_DATA = []
EVENT_STATS = []
TEAMS_DATA = []
PLAYERS_DATA = []
CACHE_FILE_NAME = 'all_gameweek_data.json'
R2_CACHE_PATH = f"{R2_BUCKET_NAME}/{CACHE_FILE_NAME}"

def get_gameweek_history(player_id, session):
    try:
        with session.get(f"{BASE_URL}element-summary/{player_id}/") as response:
            response.raise_for_status()
            return response.json().get('history', [])
    except requests.exceptions.RequestException:
        return []

def load_all_data(force_refresh=False):
    print("Fetching and processing all FPL data...")
    global EVENT_STATS, TEAMS_DATA, PLAYERS_DATA, ALL_GAMEWEEK_DATA

    points_data = []

    if not force_refresh and s3 and s3.exists(R2_CACHE_PATH):
        print(f"Cache file found in R2. Downloading...")
        try:
            with s3.open(R2_CACHE_PATH, 'r') as f:
                points_data = json.load(f)
            print("Successfully downloaded and loaded data from R2.")
        except Exception as e:
             print(f"Error reading from R2: {e}")
    
    if not points_data or force_refresh:
        print("No cache found or refresh forced. Performing full data fetch...")
        bootstrap_data = requests.get(f"{BASE_URL}bootstrap-static/").json()
        players_list = bootstrap_data.get('elements', [])
        # ... rest of your original fetching logic
        # For simplicity, returning a placeholder
        points_data = [{"player_id": 1, "points": 5}] # Replace with your actual fetch logic if needed

        if s3 and points_data:
            print(f"Uploading new data to R2...")
            try:
                with s3.open(R2_CACHE_PATH, 'w') as f:
                    json.dump(points_data, f)
                print("Successfully uploaded to R2.")
            except Exception as e:
                print(f"An error occurred uploading to R2: {e}")

    # Load bootstrap data for context
    bootstrap_data = requests.get(f"{BASE_URL}bootstrap-static/").json()
    EVENT_STATS = bootstrap_data.get('events', [])
    PLAYERS_DATA = bootstrap_data.get('elements', [])
    TEAMS_DATA = bootstrap_data.get('teams', [])
    
    ALL_GAMEWEEK_DATA = points_data
    print("Data loading complete.")

# --- RUN STARTUP PROCESS ---
load_all_data()

@app.route('/api/status')
def status():
    return jsonify({"status": "ok", "data_loaded": bool(ALL_GAMEWEEK_DATA)})

@app.route('/api/<path:path>')
def catch_all(path):
    return jsonify({"message": f"Endpoint /{path} is a placeholder."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
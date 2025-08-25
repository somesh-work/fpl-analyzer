# app.py - Final "Pandas-Free" version for Vercel

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from tqdm import tqdm
import pulp
import json
import os
import redis
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import statistics

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

try:
    endpoint_url = f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        region_name='weur',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        verify=False
    )
    print("Successfully connected to Cloudflare R2.")
except Exception as e:
    print(f"Could not connect to R2: {e}. Parquet file caching will be disabled.")
    s3_client = None

# --- GLOBAL VARIABLES ---
BASE_URL = 'https://fantasy.premierleague.com/api/'
ALL_GAMEWEEK_DATA = []
EVENT_STATS = []
TEAMS_DATA = []
PLAYERS_DATA = []
CACHE_FILE_NAME = 'all_gameweek_data.json'

def get_gameweek_history(player_id, session):
    try:
        with session.get(f"{BASE_URL}element-summary/{player_id}/") as response:
            response.raise_for_status()
            return response.json().get('history', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching history for player {player_id}: {e}")
        return []

def load_all_data(force_refresh=False):
    print("Fetching and processing all FPL data...")
    global EVENT_STATS, TEAMS_DATA, PLAYERS_DATA, ALL_GAMEWEEK_DATA

    R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
    points_data = []

    if not force_refresh and s3_client and R2_BUCKET_NAME:
        print(f"Trying to download from R2 bucket: {R2_BUCKET_NAME}...")
        try:
            obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=CACHE_FILE_NAME)
            points_data = json.loads(obj['Body'].read().decode('utf-8'))
            print("Successfully downloaded and loaded data from R2.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print("File not found in R2. Will perform a full fetch from API.")
            else:
                print(f"An error occurred downloading from R2: {e}")
    
    if not points_data or force_refresh:
        print("Performing full data fetch...")
        bootstrap_data = requests.get(f"{BASE_URL}bootstrap-static/").json()
        players_list = bootstrap_data.get('elements', [])
        teams_list = bootstrap_data.get('teams', [])
        positions_list = bootstrap_data.get('element_types', [])
        
        teams_map = {team['id']: team['name'] for team in teams_list}
        positions_map = {pos['id']: pos['singular_name_short'] for pos in positions_list}

        with requests.Session() as session:
            all_histories = []
            for player in tqdm(players_list, desc="Fetching Player Histories"):
                history_data = get_gameweek_history(player['id'], session)
                for history_item in history_data:
                    full_item = history_item.copy()
                    full_item['id'] = player['id']
                    full_item['web_name'] = player['web_name']
                    full_item['team_name'] = teams_map.get(player['team'])
                    full_item['position'] = positions_map.get(player['element_type'])
                    all_histories.append(full_item)
            points_data = all_histories

        if s3_client and R2_BUCKET_NAME and points_data:
            print(f"Uploading new data to R2 bucket...")
            try:
                s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=CACHE_FILE_NAME, Body=json.dumps(points_data))
                print("Successfully uploaded to R2.")
            except ClientError as e:
                print(f"An error occurred uploading to R2: {e}")

    bootstrap_data = requests.get(f"{BASE_URL}bootstrap-static/").json()
    EVENT_STATS = bootstrap_data.get('events', [])
    PLAYERS_DATA = bootstrap_data.get('elements', [])
    TEAMS_DATA = bootstrap_data.get('teams', [])
    
    ALL_GAMEWEEK_DATA = points_data
    print("Data loading complete.")

# Minimal placeholder for what was a complex function.
# You can rebuild the logic here using standard Python if desired.
def _get_optimal_team():
    print("Generating a basic optimal team without heavy libraries.")
    players = [p for p in PLAYERS_DATA if p.get('status') == 'a']
    
    # Use FPL's own expected points for simplicity
    for p in players:
        try:
            p['predicted_points'] = float(p.get('ep_next', 0.0))
        except (ValueError, TypeError):
            p['predicted_points'] = 0.0
        p['now_cost'] = p.get('now_cost', 0) / 10.0

    # PuLP Optimization
    prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)
    player_vars = {p['id']: pulp.LpVariable(f"player_{p['id']}", cat='Binary') for p in players}

    # Objective function
    prob += pulp.lpSum(player_vars[p['id']] * p['predicted_points'] for p in players)

    # Constraints
    prob += pulp.lpSum(player_vars[p['id']] * p['now_cost'] for p in players) <= 100.0
    prob += pulp.lpSum(player_vars.values()) == 15
    prob += pulp.lpSum(player_vars[p['id']] for p in players if p['element_type'] == 1) == 2
    prob += pulp.lpSum(player_vars[p['id']] for p in players if p['element_type'] == 2) == 5
    prob += pulp.lpSum(player_vars[p['id']] for p in players if p['element_type'] == 3) == 5
    prob += pulp.lpSum(player_vars[p['id']] for p in players if p['element_type'] == 4) == 3
    for team_id in [t['id'] for t in TEAMS_DATA]:
        prob += pulp.lpSum(player_vars[p['id']] for p in players if p['team'] == team_id) <= 3
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected_ids = {pid for pid, var in player_vars.items() if var.value() == 1}
    squad = [p for p in players if p['id'] in selected_ids]

    return squad if pulp.LpStatus[prob.status] == 'Optimal' else []

# --- RUN STARTUP PROCESS ---
load_all_data()

# --- API ENDPOINTS ---
@app.route('/api/predicted-team')
def get_predicted_team():
    squad = _get_optimal_team()
    if not squad:
        return jsonify({'error': 'Could not generate an optimal team.'}), 500

    # Format for frontend (simplified)
    team_map = {t['id']: t['name'] for t in TEAMS_DATA}
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    for p in squad:
        p['team_name'] = team_map.get(p['team'])
        p['position'] = pos_map.get(p['element_type'])
        p['cost'] = p['now_cost']
        # Add dummy keys that the frontend might expect
        p['status'] = 'Active' # Dummy status
        p['xGI'] = 0; p['ICT'] = 0; p['Recent PPG'] = 0; p['Avg FDR'] = 0

    totals = {
        'predicted_points': sum(p['predicted_points'] for p in squad),
        'cost': sum(p['now_cost'] for p in squad)
    }
    return jsonify({'team': squad, 'totals': totals})

@app.route('/api/<path:path>')
def catch_all(path):
    # This acts as a placeholder for all your other stats endpoints
    return jsonify({"message": f"Endpoint /{path} is not yet refactored in this lightweight version."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
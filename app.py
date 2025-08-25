# app.py - Final, fully functional lightweight version

import os
import json
import statistics
from functools import lru_cache
import redis
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request
from flask_cors import CORS
import pulp
import requests
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

# --- CLIENT SETUPS ---
try:
    redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
    redis_client.ping()
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"Could not connect to Redis: {e}")
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
    print(f"Could not connect to R2: {e}")
    s3_client = None

# --- GLOBAL VARIABLES & CACHING ---
BASE_URL = 'https://fantasy.premierleague.com/api/'
ALL_GAMEWEEK_DATA = []
BOOTSTRAP_DATA = {}
FIXTURES_DATA = []
CACHE_FILE_NAME = 'all_gameweek_data.json'
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
R2_CACHE_PATH = f"{R2_BUCKET_NAME}/{CACHE_FILE_NAME}" if R2_BUCKET_NAME else None

@lru_cache(maxsize=None)
def get_bootstrap_data():
    global BOOTSTRAP_DATA
    if not BOOTSTRAP_DATA:
        print("Fetching and caching bootstrap data.")
        BOOTSTRAP_DATA = requests.get(f"{BASE_URL}bootstrap-static/").json()
    return BOOTSTRAP_DATA

@lru_cache(maxsize=None)
def get_fixtures_data():
    global FIXTURES_DATA
    if not FIXTURES_DATA:
        print("Fetching and caching fixtures data.")
        FIXTURES_DATA = requests.get(f"{BASE_URL}fixtures/").json()
    return FIXTURES_DATA

def get_gameweek_history(player_id, session):
    try:
        with session.get(f"{BASE_URL}element-summary/{player_id}/") as response:
            response.raise_for_status()
            return response.json().get('history', [])
    except requests.exceptions.RequestException:
        return []

def load_all_data(force_refresh=False):
    print("Initializing FPL data load...")
    global ALL_GAMEWEEK_DATA
    
    points_data = []
    if not force_refresh and s3_client and R2_CACHE_PATH:
        try:
            if s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=CACHE_FILE_NAME):
                print(f"Cache file found in R2. Downloading...")
                obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=CACHE_FILE_NAME)
                points_data = json.loads(obj['Body'].read().decode('utf-8'))
                print("Successfully loaded data from R2.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print("Cache file not found in R2.")
            else:
                print(f"An R2 error occurred: {e}")

    if not points_data or force_refresh:
        print("Performing full data fetch from FPL API...")
        bootstrap = get_bootstrap_data()
        players_list = bootstrap.get('elements', [])
        teams_map = {t['id']: t['name'] for t in bootstrap.get('teams', [])}
        pos_map = {p['id']: p['singular_name_short'] for p in bootstrap.get('element_types', [])}
        
        with requests.Session() as session:
            all_histories = []
            for player in tqdm(players_list, desc="Fetching Player Histories"):
                history = get_gameweek_history(player['id'], session)
                for entry in history:
                    full_entry = entry.copy()
                    full_entry.update({
                        'id': player['id'],
                        'web_name': player['web_name'],
                        'team_name': teams_map.get(player['team']),
                        'position': pos_map.get(player['element_type'])
                    })
                    all_histories.append(full_entry)
            points_data = all_histories

        if s3_client and R2_CACHE_PATH and points_data:
            print("Uploading fresh data to R2...")
            try:
                s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=CACHE_FILE_NAME, Body=json.dumps(points_data))
                print("Successfully uploaded to R2.")
            except ClientError as e:
                print(f"Could not upload to R2: {e}")

    ALL_GAMEWEEK_DATA = points_data
    print("Data loading complete.")

# --- UTILITY FUNCTIONS (REPLACING PANDAS) ---
def get_filtered_gameweek_data(gw, lag):
    start_gw = gw - lag
    return [p for p in ALL_GAMEWEEK_DATA if start_gw <= p.get('round', 0) <= gw]

def group_and_aggregate(data, group_by_key, agg_fields):
    groups = {}
    for row in data:
        key = row.get(group_by_key)
        if key not in groups:
            groups[key] = {field: 0 for field in agg_fields}
            groups[key][group_by_key] = key
        for field in agg_fields:
            groups[key][field] += row.get(field, 0)
    return list(groups.values())

# --- RUN STARTUP PROCESS ---
load_all_data()

# --- API ENDPOINTS ---
@app.route('/api/latest-gameweek')
def get_latest_gameweek():
    events = get_bootstrap_data().get('events', [])
    finished_gws = [gw['id'] for gw in events if gw.get('is_finished')]
    return jsonify({'latest_gameweek': max(finished_gws) if finished_gws else 0})

@app.route('/api/fdr')
def get_fdr_data():
    bootstrap = get_bootstrap_data()
    teams = bootstrap.get('teams', [])
    fixtures = get_fixtures_data()
    team_map = {t['id']: t['short_name'] for t in teams}

    # Find the next gameweek
    events = bootstrap.get('events', [])
    next_gw_id = next((event['id'] for event in events if event.get('is_next')), None)
    if not next_gw_id:
        return jsonify([])

    fdr_results = []
    for team in teams:
        team_fixtures = []
        fdr_sum = 0
        for i in range(6): # Next 6 gameweeks
            gw_id = next_gw_id + i
            gw_fixtures = [f for f in fixtures if f.get('event') == gw_id and (f.get('team_h') == team['id'] or f.get('team_a') == team['id'])]
            if gw_fixtures:
                fixture = gw_fixtures[0]
                is_home = fixture['team_h'] == team['id']
                difficulty = fixture['team_h_difficulty'] if is_home else fixture['team_a_difficulty']
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                team_fixtures.append({
                    "opponent": team_map.get(opponent_id, '?'),
                    "fdr": difficulty,
                    "location": "H" if is_home else "A"
                })
                fdr_sum += difficulty
            else:
                team_fixtures.append(None)
        
        fdr_results.append({
            'name': team['name'],
            'fdr_sum': fdr_sum,
            'fixtures': {next_gw_id + i: fix for i, fix in enumerate(team_fixtures)}
        })

    fdr_results.sort(key=lambda x: x['fdr_sum'])
    return jsonify(fdr_results)

@app.route('/api/stats/<path:stat_path>')
def get_pl_stats(stat_path):
    # This single endpoint can handle all your simple stats tables
    gw = request.args.get('gw', type=int, default=38)
    lag = request.args.get('lag', type=int, default=37)
    
    data_slice = get_filtered_gameweek_data(gw, lag)
    
    player_stats = {}
    for p in data_slice:
        pid = p['id']
        if pid not in player_stats:
            player_stats[pid] = {
                'web_name': p['web_name'], 'team': p['team_name'], 'position': p['position'],
                'goals_scored': 0, 'assists': 0, 'goals_n_assists': 0, 'saves': 0, 'bps': 0,
                'expected_goals': 0, 'expected_assists': 0, 'expected_goal_involvements': 0,
                'ict_index': 0, 'total_points': 0
            }
        # Aggregate stats
        for key in player_stats[pid]:
            if key not in ['web_name', 'team', 'position']:
                player_stats[pid][key] += p.get(key, 0)

    player_list = list(player_stats.values())

    # Define what to return for each path
    stat_configs = {
        'pl-stats-1': {
            'most_goals': ('goals_scored', 'expected_goals'),
            'most_assists': ('assists', 'expected_assists'),
            'most_g_a': ('goals_n_assists', 'expected_goal_involvements'),
            'most_saves': ('saves', 'total_points')
        },
        'pl-stats-2': {
            'most_bps': ('bps', 'total_points'),
            'expected_goals': ('expected_goals', 'total_points'),
            'expected_assists': ('expected_assists', 'total_points'),
            'expected_gi': ('expected_goal_involvements', 'total_points')
        }
    }

    if stat_path in stat_configs:
        response = {}
        for key, sort_keys in stat_configs[stat_path].items():
            sorted_list = sorted(player_list, key=lambda x: x.get(sort_keys[0], 0), reverse=True)
            response[key] = sorted_list[:10]
        return jsonify(response)

    return jsonify({"error": "Stat path not found"}), 404


# Keep a simple predicted-team endpoint for now
@app.route('/api/predicted-team')
def get_predicted_team():
    # Placeholder using FPL's expected points
    players = get_bootstrap_data().get('elements', [])
    players.sort(key=lambda p: float(p.get('ep_next', 0)), reverse=True)
    
    team_map = {t['id']: t['name'] for t in get_bootstrap_data().get('teams', [])}
    pos_map = {p['id']: p['singular_name_short'] for p in get_bootstrap_data().get('element_types', [])}

    # Simplified team selection
    squad = players[:15]
    for p in squad:
        p['status'] = 'Active' # Dummy status
        p['team_name'] = team_map.get(p['team'])
        p['position'] = pos_map.get(p['element_type'])
        p['cost'] = p.get('now_cost', 0) / 10.0
        p['predicted_points'] = float(p.get('ep_next', 0))

    totals = {
        'cost': sum(p['cost'] for p in squad),
        'predicted_points': sum(p['predicted_points'] for p in squad)
    }
    return jsonify({"team": squad, "totals": totals})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
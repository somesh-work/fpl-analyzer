# app.py - Final Backend Code with Redis and R2 Caching

from flask import Flask, jsonify, request, send_file
from io import BytesIO
from flask_cors import CORS
import pandas as pd
import requests
from tqdm import tqdm
import warnings
import numpy as np
import pulp
from sklearn.preprocessing import MinMaxScaler
import json
import os
import redis
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

# Suppress future warnings from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app)

# --- REDIS CLIENT SETUP ---
try:
    # It will automatically use the REDIS_URL from your Render environment variables
    redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
    redis_client.ping() # Check the connection
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"Could not connect to Redis: {e}. Prediction caching will be disabled.")
    redis_client = None

# --- S3/R2 CLIENT SETUP ---
try:
    # This endpoint_url is specific to Cloudflare R2
    endpoint_url = f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        region_name='weur',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}) # <-- ADD THIS LINE
    )

    print("Successfully connected to Cloudflare R2.")
except Exception as e:
    print(f"Could not connect to R2: {e}. Parquet file caching will be disabled.")
    s3_client = None


# --- GLOBAL VARIABLES & DATA LOADING ---
BASE_URL = 'https://fantasy.premierleague.com/api/'
ALL_GAMEWEEK_DATA = pd.DataFrame()
EVENT_STATS = pd.DataFrame()
TEAMS_DF = pd.DataFrame()
FIXTURES_DF = pd.DataFrame()
PLAYERS_DF_FULL = pd.DataFrame()
PREDICTED_TEAM_HISTORY = []

# --- CACHE FILE PATHS ---
# We now only use a local path for the parquet file, which acts as a temporary location
ALL_DATA_CACHE_FILE = 'all_gameweek_data.parquet'

def get_gameweek_history(player_id):
    """Fetches all gameweek info for a given player_id for the current season."""
    try:
        r = requests.get(f"{BASE_URL}element-summary/{player_id}/").json()
        df = pd.json_normalize(r['history'])
        return df
    except Exception as e:
        # Return an empty DataFrame on error to avoid breaking the concatenation
        return pd.DataFrame()

def load_all_data(force_refresh=False):
    """
    Fetches all FPL data. It uses a local file first, then tries to download
    from Cloudflare R2. If both fail, it fetches from the live API and uploads
    the result to R2 for future starts.
    """
    print("Fetching and processing all FPL data for the current season...")
    global EVENT_STATS, TEAMS_DF, FIXTURES_DF, ALL_GAMEWEEK_DATA, PLAYERS_DF_FULL

    R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
    local_file_exists = os.path.exists(ALL_DATA_CACHE_FILE)
    points = pd.DataFrame()

    if not force_refresh and local_file_exists:
        print(f"Loading player history from local cache file: {ALL_DATA_CACHE_FILE}")
        points = pd.read_parquet(ALL_DATA_CACHE_FILE)
    elif not force_refresh and s3_client and R2_BUCKET_NAME:
        print(f"Local cache not found. Trying to download from R2 bucket: {R2_BUCKET_NAME}...")
        try:
            s3_client.download_file(R2_BUCKET_NAME, ALL_DATA_CACHE_FILE, ALL_DATA_CACHE_FILE)
            print("Successfully downloaded from R2. Loading into DataFrame.")
            points = pd.read_parquet(ALL_DATA_CACHE_FILE)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print("File not found in R2. Will perform a full fetch from API.")
            else:
                print(f"An error occurred downloading from R2: {e}")
    
    # If points DataFrame is still empty, we need to do a full fetch
    if points.empty or force_refresh:
        if force_refresh:
            print("Forcing data refresh. Fetching all player histories from API...")
        else:
            print("No cache found locally or in R2. Performing full fetch (this will take a few minutes)...")
        
        # --- ORIGINAL API FETCHING LOGIC ---
        bootstrap_data_full = requests.get(f"{BASE_URL}bootstrap-static/").json()
        players_df = pd.json_normalize(bootstrap_data_full['elements'])
        all_histories = [get_gameweek_history(player_id) for player_id in tqdm(players_df['id'], desc="Fetching Player Histories")]
        points = pd.concat(all_histories, ignore_index=True)

        if not points.empty:
            players_map = players_df[['id', 'web_name', 'team', 'element_type']]
            teams_map_df = pd.json_normalize(bootstrap_data_full['teams'])
            positions_map_df = pd.json_normalize(bootstrap_data_full['element_types'])
            
            points = players_map.merge(points, left_on='id', right_on='element')
            points = points.merge(
                teams_map_df[['id', 'name']].rename(columns={'id': 'team_id'}),
                left_on='team',
                right_on='team_id'
            ).drop(['team', 'team_id'], axis=1)
            points = points.merge(
                positions_map_df[['id', 'singular_name_short']].rename(columns={'id': 'position_id'}),
                left_on='element_type',
                right_on='position_id'
            ).drop(['element_type', 'position_id'], axis=1)
            points = points.rename(columns={'name': 'team', 'singular_name_short': 'position'})

            if 'goals_scored' in points.columns and 'assists' in points.columns:
                points['goals_n_assists'] = points['goals_scored'] + points['assists']
            
            # Save locally first
            points.to_parquet(ALL_DATA_CACHE_FILE)
            
            # --- NEW: UPLOAD TO R2 ---
            if s3_client and R2_BUCKET_NAME:
                print(f"Uploading new data to R2 bucket: {R2_BUCKET_NAME}...")
                try:
                    s3_client.upload_file(ALL_DATA_CACHE_FILE, R2_BUCKET_NAME, ALL_DATA_CACHE_FILE)
                    print("Successfully uploaded to R2.")
                except ClientError as e:
                    print(f"An error occurred uploading to R2: {e}")

    # This part needs to run regardless of cache to get latest player lists, events etc.
    bootstrap_data = requests.get(f"{BASE_URL}bootstrap-static/").json()
    events_df = pd.json_normalize(bootstrap_data['events'])
    EVENT_STATS = events_df[['id', 'average_entry_score', 'highest_score', 'finished']].rename(columns={'id': 'gameweek'})
    PLAYERS_DF_FULL = pd.json_normalize(bootstrap_data['elements'])
    TEAMS_DF = pd.json_normalize(bootstrap_data['teams'])
    fixtures_raw = requests.get(f"{BASE_URL}fixtures/").json()
    FIXTURES_DF = pd.json_normalize(fixtures_raw)
    FIXTURES_DF = FIXTURES_DF.dropna(subset=['event'])
    FIXTURES_DF['event'] = FIXTURES_DF['event'].astype(int)

    if not points.empty:
        numeric_cols = [
            'total_points', 'goals_scored', 'assists', 'ict_index', 'clean_sheets',
            'expected_goals', 'expected_assists', 'goals_n_assists', 'saves',
            'expected_goal_involvements', 'bps', 'expected_goals_conceded'
        ]
        for col in numeric_cols:
            if col in points.columns:
                points[col] = pd.to_numeric(points[col], errors='coerce')
    
    ALL_GAMEWEEK_DATA = points
    print("Data loading complete.")

def _get_player_predictions(gw_to_predict, custom_weights=None):
    """
    Analyzes all players and returns a DataFrame with their predicted points for a given gameweek.
    Accepts an optional dictionary of custom weights for the prediction playground.
    """
    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw_context = finished_gws['gameweek'].max() if not finished_gws.empty else 0
    fdr_start_gw = latest_gw_context + 1

    players = PLAYERS_DF_FULL.copy()
    numeric_stats = ['form', 'ict_index', 'expected_goals', 'expected_assists', 'expected_goals_conceded', 'now_cost', 'selected_by_percent']
    for stat in numeric_stats:
        players[stat] = pd.to_numeric(players[stat], errors='coerce')
    players[numeric_stats] = players[numeric_stats].fillna(0)
    players['xGI'] = players['expected_goals'] + players['expected_assists']
    
    team_stats = players.groupby('team')['expected_goals_conceded'].sum().reset_index().rename(columns={'expected_goals_conceded': 'team_xGC'})
    players = players.merge(team_stats, on='team')
    
    if latest_gw_context < 1:
        players['ppg_last_4'] = 0
    else:
        start_gw = max(1, latest_gw_context - 3)
        past_gws = ALL_GAMEWEEK_DATA.query("round >= @start_gw and round <= @latest_gw_context")
        recent_performance = past_gws.groupby('element')['total_points'].mean().reset_index().rename(columns={'total_points': 'ppg_last_4'})
        players = players.merge(recent_performance, left_on='id', right_on='element', how='left')
        players['ppg_last_4'] = players['ppg_last_4'].fillna(0)

    for col in ['xGI', 'ict_index', 'ppg_last_4']:
        players[f'raw_{col}'] = players[col]

    scaler = MinMaxScaler()
    metrics_to_normalize = ['xGI', 'ict_index', 'form', 'team_xGC', 'ppg_last_4']
    players[metrics_to_normalize] = scaler.fit_transform(players[metrics_to_normalize])
    
    n_upcoming_fixtures = 4
    upcoming_gws = range(fdr_start_gw, fdr_start_gw + n_upcoming_fixtures)
    upcoming_fixtures = FIXTURES_DF[FIXTURES_DF['event'].isin(upcoming_gws)]
    team_fdr = {}
    for team_id in TEAMS_DF['id']:
        team_fixtures = upcoming_fixtures[(upcoming_fixtures['team_h'] == team_id) | (upcoming_fixtures['team_a'] == team_id)]
        difficulties = [row['team_h_difficulty'] if row['team_h'] == team_id else row['team_a_difficulty'] for _, row in team_fixtures.iterrows()]
        avg_fdr = np.mean(difficulties) if difficulties else 3.0
        team_fdr[team_id] = avg_fdr
    team_fdr_df = pd.DataFrame(list(team_fdr.items()), columns=['team', 'avg_fdr'])
    players = players.merge(team_fdr_df, on='team')

    def fdr_to_multiplier(fdr, position_type):
        if position_type in [3, 4]: # MID, FWD
            if fdr <= 2.5: return 1.20;
            if fdr <= 3.0: return 1.05;
            if fdr <= 3.5: return 1.00;
            if fdr <= 4.0: return 0.90;
            return 0.80
        else: # GKP, DEF
            if fdr <= 2.2: return 1.25;
            if fdr <= 2.8: return 1.10;
            if fdr <= 3.4: return 1.00;
            if fdr <= 4.0: return 0.95;
            return 0.85
    
    players['fdr_multiplier'] = players.apply(lambda row: fdr_to_multiplier(row['avg_fdr'], row['element_type']), axis=1)

    if custom_weights:
        weights = custom_weights
    else:
        weights = {
            'xGI': 0.45, 'ict': 0.30, 'form': 0.10, 'ppg_last_4': 0.15,
            'def_xGC': 0.35, 'def_xGI': 0.25, 'def_ict': 0.25, 'def_ppg': 0.15,
            'gkp_xGC': 0.50, 'gkp_ppg': 0.50, 'fdr': 1.0
        }

    def calculate_predicted_points(player):
        pos = player['element_type']
        score = 0
        if pos in [3, 4]: # MID, FWD
            score = (weights['xGI'] * player['xGI'] + weights['ict'] * player['ict_index'] +
                     weights['form'] * player['form'] + weights['ppg_last_4'] * player['ppg_last_4'])
        elif pos == 2: # DEF
            score = (weights['def_xGC'] * (1 - player['team_xGC']) + weights['def_xGI'] * player['xGI'] +
                     weights['def_ict'] * player['ict_index'] + weights['def_ppg'] * player['ppg_last_4'])
        elif pos == 1: # GKP
            score = (weights['gkp_xGC'] * (1 - player['team_xGC']) + weights['gkp_ppg'] * player['ppg_last_4'])
        
        play_chance = player['chance_of_playing_next_round'] if pd.notnull(player['chance_of_playing_next_round']) else 100
        base_score = score * (play_chance / 100)
        
        fdr_weight = weights.get('fdr', 1.0)
        weighted_fdr_multiplier = player['fdr_multiplier'] ** fdr_weight
        
        return base_score * weighted_fdr_multiplier

    players['predicted_points'] = players.apply(calculate_predicted_points, axis=1)
    players['now_cost'] = players['now_cost'] / 10.0
    return players

def _get_optimal_team_for_gw(gw_to_predict, is_review=False, custom_weights=None):
    """
    Internal helper to generate an optimal team for a given gameweek.
    Returns the final 15-man DataFrame with all calculated stats.
    """
    players = _get_player_predictions(gw_to_predict, custom_weights=custom_weights)
    players_for_squad = players[players['now_cost'] > 0].copy()

    squad_prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    player_vars = {p['id']: pulp.LpVariable(f"player_{p['id']}", cat='Binary') for _, p in players_for_squad.iterrows()}
    squad_prob += pulp.lpSum([player_vars[p['id']] * (p['predicted_points'] + 1e-6) for _, p in players_for_squad.iterrows()])
    squad_prob += pulp.lpSum([player_vars[p['id']] * p['now_cost'] for _, p in players_for_squad.iterrows()]) <= 100.0
    squad_prob += pulp.lpSum(player_vars.values()) == 15
    squad_prob += pulp.lpSum([player_vars[p['id']] for _, p in players_for_squad.iterrows() if p['element_type'] == 1]) == 2
    squad_prob += pulp.lpSum([player_vars[p['id']] for _, p in players_for_squad.iterrows() if p['element_type'] == 2]) == 5
    squad_prob += pulp.lpSum([player_vars[p['id']] for _, p in players_for_squad.iterrows() if p['element_type'] == 3]) == 5
    squad_prob += pulp.lpSum([player_vars[p['id']] for _, p in players_for_squad.iterrows() if p['element_type'] == 4]) == 3
    for team_id in TEAMS_DF['id']:
        squad_prob += pulp.lpSum([player_vars[p['id']] for _, p in players_for_squad.iterrows() if p['team'] == team_id]) <= 3
    squad_prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[squad_prob.status] == 'Optimal':
        selected_player_ids = [pid for pid, var in player_vars.items() if var.value() == 1]
        squad_df = players_for_squad[players_for_squad['id'].isin(selected_player_ids)].copy()

        lineup_prob = pulp.LpProblem("FPL_Lineup_Optimization", pulp.LpMaximize)
        lineup_vars = {p['id']: pulp.LpVariable(f"lineup_{p['id']}", cat='Binary') for _, p in squad_df.iterrows()}
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] * p['predicted_points'] for _, p in squad_df.iterrows()])
        
        lineup_prob += pulp.lpSum(lineup_vars.values()) == 11
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 1]) == 1
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 2]) >= 3
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 2]) <= 5
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 3]) >= 2
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 3]) <= 5
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 4]) >= 1
        lineup_prob += pulp.lpSum([lineup_vars[p['id']] for _, p in squad_df.iterrows() if p['element_type'] == 4]) <= 3
        
        lineup_prob.solve(pulp.PULP_CBC_CMD(msg=0))
        active_player_ids = [pid for pid, var in lineup_vars.items() if var.value() == 1]
        
        squad_df['status'] = squad_df['id'].apply(lambda x: 'Active' if x in active_player_ids else 'Bench')
        return squad_df
    return None

def calculate_all_review_teams():
    print("Building predicted team history...")
    global PREDICTED_TEAM_HISTORY
    PREDICTED_TEAM_HISTORY = []
    
    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw = finished_gws['gameweek'].max() if not finished_gws.empty else 0
    
    for gw in tqdm(range(1, latest_gw + 1), desc="Building/Reviewing History"):
        squad_df = None
        redis_key = f"prediction:{gw}"

        if redis_client:
            cached_json = redis_client.get(redis_key)
            if cached_json:
                squad_for_gw_list = json.loads(cached_json)
                squad_df = pd.DataFrame(squad_for_gw_list)
        
        if squad_df is None:
            print(f"No Redis cache for GW{gw}. Generating retrospective team for history.")
            squad_df = _get_optimal_team_for_gw(gw, is_review=True)
            
            if squad_df is not None:
                cols_to_cache = ['id','status', 'web_name', 'team', 'element_type', 'now_cost', 'raw_xGI', 'raw_ict_index', 'raw_ppg_last_4', 'avg_fdr', 'predicted_points']
                squad_to_cache_list = squad_df[cols_to_cache].to_dict(orient='records')
                if redis_client:
                    redis_client.set(redis_key, json.dumps(squad_to_cache_list))

        if squad_df is not None:
            if 'now_cost' in squad_df.columns:
                squad_df.rename(columns={'now_cost': 'cost'}, inplace=True)

            active_players = squad_df[squad_df['status'] == 'Active'].copy()
            
            if not active_players.empty:
                active_player_ids = active_players['id'].tolist()
                
                captain = active_players.sort_values(by='predicted_points', ascending=False).iloc[0]
                captain_id = captain['id']

                actual_points_df = ALL_GAMEWEEK_DATA.query("round == @gw and element in @active_player_ids")[['element', 'total_points']]
                
                total_actual_points = actual_points_df['total_points'].sum()
                
                captain_actual_points_series = actual_points_df[actual_points_df['element'] == captain_id]['total_points']
                if not captain_actual_points_series.empty:
                    total_actual_points += captain_actual_points_series.iloc[0]
                
                PREDICTED_TEAM_HISTORY.append({
                    'gameweek': gw,
                    'predicted_team_points': int(total_actual_points)
                })

# --- RUN STARTUP PROCESS ---
load_all_data()
calculate_all_review_teams()

# --- API ENDPOINTS ---

def get_request_args():
    """Helper to parse and validate gameweek and lag from request arguments."""
    gw = request.args.get('gw', default=1, type=int)
    lag = request.args.get('lag', default=0, type=int)
    
    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw = finished_gws['gameweek'].max() if not finished_gws.empty else 0

    if latest_gw > 0:
        gw = min(gw, latest_gw)
        lag = min(lag, gw - 1) if lag >= 0 else gw - 1
    return gw, lag

@app.route('/api/predicted-team')
def get_predicted_team():
    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw = finished_gws['gameweek'].max() if not finished_gws.empty else 0
    context_gw = request.args.get('gw', default=latest_gw, type=int)

    if latest_gw > 0:
        context_gw = min(context_gw, latest_gw)
    
    next_gw = context_gw + 1
    squad_df = None
    redis_key = f"prediction:{next_gw}"

    if redis_client:
        cached_squad_json = redis_client.get(redis_key)
        if cached_squad_json:
            print(f"Loading predicted team for GW{next_gw} from Redis cache.")
            cached_squad_list = json.loads(cached_squad_json)
            squad_df = pd.DataFrame(cached_squad_list)
            if 'now_cost' in squad_df.columns:
                squad_df.rename(columns={'now_cost': 'cost'}, inplace=True)

    if squad_df is None:
        print(f"No Redis cache for GW{next_gw}. Generating a new prediction.")
        squad_df = _get_optimal_team_for_gw(next_gw, is_review=True)
        if squad_df is not None:
            cols_to_cache = ['id','status', 'web_name', 'team', 'element_type', 'now_cost', 'raw_xGI', 'raw_ict_index', 'raw_ppg_last_4', 'avg_fdr', 'predicted_points']
            squad_to_cache_list = squad_df[cols_to_cache].to_dict(orient='records')
            
            if redis_client:
                redis_client.set(redis_key, json.dumps(squad_to_cache_list), ex=604800)
                print(f"Saved prediction for GW{next_gw} to Redis cache.")
            
            squad_df.rename(columns={'now_cost': 'cost'}, inplace=True)

    if squad_df is not None:
        squad_df = squad_df.sort_values(by=['status', 'element_type'], ascending=[False, True])
        squad_df = squad_df.merge(TEAMS_DF[['id', 'name']], left_on='team', right_on='id', suffixes=('', '_team'))
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        squad_df['position'] = squad_df['element_type'].map(pos_map)
        
        output_cols = ['status', 'web_name', 'name', 'position', 'cost', 'raw_xGI', 'raw_ict_index', 'raw_ppg_last_4', 'avg_fdr', 'predicted_points']
        final_df = squad_df[output_cols]
        final_df = final_df.rename(columns={'name': 'team_name', 'raw_xGI': 'xGI', 'raw_ict_index': 'ICT', 'raw_ppg_last_4': 'Recent PPG', 'avg_fdr': 'Avg FDR'})
        
        totals = {
            'predicted_points': round(squad_df[squad_df['status'] == 'Active']['predicted_points'].sum(), 2),
            'cost': round(squad_df['cost'].sum(), 1)
        }
        return jsonify({'team': final_df.fillna(0).to_dict(orient='records'), 'totals': totals})
    else:
        return jsonify({'error': 'Could not find an optimal team.'}), 500

# --- The rest of your API endpoints remain unchanged ---
# (They don't use the prediction cache, so they are fine)

@app.route('/api/prediction-playground', methods=['POST'])
def prediction_playground():
    custom_weights = request.json
    
    if not custom_weights:
        return jsonify({'error': 'No weights provided.'}), 400

    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw = finished_gws['gameweek'].max() if not finished_gws.empty else 0
    context_gw = latest_gw

    squad_df = _get_optimal_team_for_gw(context_gw + 1, is_review=True, custom_weights=custom_weights)
    
    if squad_df is not None:
        squad_df = squad_df.sort_values(by=['status', 'element_type'], ascending=[False, True])
        squad_df = squad_df.merge(TEAMS_DF[['id', 'name']], left_on='team', right_on='id', suffixes=('', '_team'))
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        squad_df['position'] = squad_df['element_type'].map(pos_map)
        
        output_cols = ['status', 'web_name', 'name', 'position', 'now_cost', 'predicted_points']
        final_df = squad_df[output_cols]
        final_df = final_df.rename(columns={'name': 'team_name', 'now_cost': 'cost'})
        
        totals = {
            'predicted_points': round(squad_df[squad_df['status'] == 'Active']['predicted_points'].sum(), 2),
            'cost': round(squad_df['now_cost'].sum(), 1)
        }
        return jsonify({'team': final_df.fillna(0).to_dict(orient='records'), 'totals': totals})
    else:
        return jsonify({'error': 'Could not find an optimal team with the given weights.'}), 500

@app.route('/api/review-team')
def get_review_team():
    gw_to_review = request.args.get('gw', type=int)
    if not gw_to_review:
        return jsonify({'error': 'Gameweek parameter is required.'}), 400

    squad_df = None
    redis_key = f"prediction:{gw_to_review}"
    
    if redis_client:
        cached_squad_json = redis_client.get(redis_key)
        if cached_squad_json:
            print(f"Loading GW{gw_to_review} prediction from Redis cache.")
            cached_squad_list = json.loads(cached_squad_json)
            squad_df = pd.DataFrame(cached_squad_list)
            squad_df.rename(columns={'now_cost': 'cost'}, inplace=True)
    
    if squad_df is None:
        print(f"No cache for GW{gw_to_review}. Generating a retrospective prediction.")
        squad_df = _get_optimal_team_for_gw(gw_to_review, is_review=True)
        if squad_df is None:
            return jsonify({'error': 'Could not generate a retrospective team for the review.'}), 500
        # NOTE: We don't save to cache here to keep this endpoint read-only for history
        squad_df.rename(columns={'now_cost': 'cost'}, inplace=True)

    player_ids = squad_df['id'].tolist()
    actual_points_df = ALL_GAMEWEEK_DATA.query("round == @gw_to_review and element in @player_ids")[['element', 'total_points']].rename(columns={'total_points': 'actual_points', 'element': 'id'})
    squad_df = squad_df.merge(actual_points_df, on='id', how='left')
    squad_df['actual_points'] = squad_df['actual_points'].fillna(0).astype(int)

    squad_df = squad_df.merge(TEAMS_DF[['id', 'name']], left_on='team', right_on='id', suffixes=('', '_team'))
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    squad_df['position'] = squad_df['element_type'].map(pos_map)
    squad_df = squad_df.sort_values(by=['status', 'element_type'], ascending=[False, True])
    
    final_df = squad_df.rename(columns={'name': 'team_name', 'raw_xGI': 'xGI', 'raw_ict_index': 'ICT', 'raw_ppg_last_4': 'Recent PPG', 'avg_fdr': 'Avg FDR'})
    
    active_team = final_df[final_df['status'] == 'Active'].copy()
    active_team_sorted = active_team.sort_values(by='predicted_points', ascending=False)
    
    captain = active_team_sorted.iloc[0] if not active_team_sorted.empty else None
    vice_captain = active_team_sorted.iloc[1] if len(active_team_sorted) > 1 else None

    captaincy = {
        'captain': captain['web_name'] if captain is not None else None,
        'viceCaptain': vice_captain['web_name'] if vice_captain is not None else None
    }

    total_actual_points = active_team['actual_points'].sum()
    if captain is not None:
        total_actual_points += captain['actual_points']

    totals = {
        'predicted_points': round(active_team['predicted_points'].sum(), 2),
        'actual_points': int(total_actual_points),
        'cost': round(final_df['cost'].sum(), 1)
    }
    
    output_cols = ['status', 'web_name', 'team_name', 'position', 'cost', 'xGI', 'ICT', 'Recent PPG', 'Avg FDR', 'predicted_points', 'actual_points']
    
    return jsonify({
        'team': final_df[output_cols].fillna(0).to_dict(orient='records'), 
        'totals': totals,
        'captaincy': captaincy
    })

@app.route('/api/predicted-history')
def get_predicted_history():
    return jsonify(PREDICTED_TEAM_HISTORY)

@app.route('/api/latest-gameweek')
def get_latest_gameweek():
    if not EVENT_STATS.empty:
        finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
        if not finished_gws.empty:
            return jsonify({'latest_gameweek': int(finished_gws['gameweek'].max())})
        return jsonify({'latest_gameweek': 0})
    return jsonify({'error': 'Data not loaded'}), 404

# Your other endpoints like manager-team, fdr, stats, etc.
# can remain exactly as they were. I'm including them for completeness.

@app.route('/api/fdr')
def get_fdr_data():
    gw, lag = get_request_args()
    finished_gws = EVENT_STATS[EVENT_STATS['finished'] == True]
    latest_gw = finished_gws['gameweek'].max() if not finished_gws.empty else 0
    start_gw = gw - lag
    stats_df = ALL_GAMEWEEK_DATA.query("round <= @gw and round >= @start_gw")
    if stats_df.empty:
        return jsonify([])
    top_attack_teams = stats_df.query("position in ['MID', 'FWD']").groupby('team')['total_points'].sum().nlargest(10).index.tolist()
    top_defence_teams = stats_df.query("position in ['DEF', 'GKP']").groupby('team')['total_points'].sum().nlargest(10).index.tolist()
    team_stats = stats_df.groupby('team').agg(xgi_sum=('expected_goal_involvements', 'sum'), points_sum=('total_points', 'sum')).reset_index()
    team_map = TEAMS_DF.set_index('id')['name'].to_dict()
    upcoming_gws = range(latest_gw + 1, latest_gw + 7)
    fixtures_subset = FIXTURES_DF[FIXTURES_DF['event'].isin(upcoming_gws)]
    fdr_results = []
    for team_id, team_name in team_map.items():
        team_display_name = f"{team_name}*" if team_name in top_attack_teams else team_name
        current_team_stats = team_stats[team_stats['team'] == team_name]
        xgi_sum = float(current_team_stats.iloc[0]['xgi_sum']) if not current_team_stats.empty else 0.0
        points_sum = int(current_team_stats.iloc[0]['points_sum']) if not current_team_stats.empty else 0
        team_fixtures_data = {'name': team_display_name, 'fixtures': {}, 'fdr_sum': 0, 'xgi_sum': xgi_sum, 'points_sum': points_sum}
        for gw in upcoming_gws:
            fixture = fixtures_subset[(fixtures_subset['event'] == gw) & ((fixtures_subset['team_h'] == team_id) | (fixtures_subset['team_a'] == team_id))]
            if not fixture.empty:
                fixture_info = fixture.iloc[0]
                if fixture_info['team_h'] == team_id:
                    opponent_id, difficulty, location = fixture_info['team_a'], fixture_info['team_h_difficulty'], 'H'
                else:
                    opponent_id, difficulty, location = fixture_info['team_h'], fixture_info['team_a_difficulty'], 'A'
                opponent_name = team_map.get(opponent_id, 'N/A')
                opponent_display_name = f"{opponent_name}+" if opponent_name in top_defence_teams else opponent_name
                team_fixtures_data['fixtures'][gw] = {'opponent': opponent_display_name, 'fdr': int(difficulty), 'location': location}
                team_fixtures_data['fdr_sum'] += int(difficulty)
            else:
                team_fixtures_data['fixtures'][gw] = None
        fdr_results.append(team_fixtures_data)
    fdr_results.sort(key=lambda x: (x['fdr_sum'], -x['xgi_sum'], -x['points_sum']))
    return jsonify(fdr_results)

@app.route('/api/manager-team')
def get_manager_team_stats():
    manager_id, gw = request.args.get('manager_id', type=int), request.args.get('gw', type=int)
    if not manager_id or not gw:
        return jsonify({'error': 'Manager ID and Gameweek are required'}), 400
    if ALL_GAMEWEEK_DATA.empty:
        return jsonify({'error': 'Manager stats are not available until the season starts.'}), 404
    try:
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gw}/picks/').json()
        if 'picks' not in res:
            return jsonify({'error': 'Could not find team for this manager/gameweek.'}), 404
        my_team_df = pd.DataFrame(res['picks'])
        my_team_df['status'] = my_team_df['position'].apply(lambda x: 'Active' if x <= 11 else 'Bench')
        my_team_df = my_team_df.merge(PLAYERS_DF_FULL[['id', 'web_name']], left_on='element', right_on='id')
        team_player_ids = my_team_df['element'].tolist()
        gw_data = ALL_GAMEWEEK_DATA[(ALL_GAMEWEEK_DATA['element'].isin(team_player_ids)) & (ALL_GAMEWEEK_DATA['round'] == gw)]
        final_team_df = my_team_df.merge(gw_data, on='element', how='left', suffixes=('_pick', ''))
        return jsonify(final_team_df.fillna(0).to_dict(orient='records'))
    except Exception as e:
        print(f"Error fetching manager team: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/api/manager-history')
def get_manager_history():
    manager_id = request.args.get('manager_id', type=int)
    if not manager_id:
        return jsonify({'error': 'Manager ID is required'}), 400
    if EVENT_STATS.empty:
        return jsonify({'error': 'Manager history is not available.'}), 404
    try:
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{manager_id}/history/').json()
        if 'current' not in res:
            return jsonify({'error': 'Could not find history for this manager ID.'}), 404
        manager_history_df = pd.DataFrame(res['current'])[['event', 'points']].rename(columns={'event': 'gameweek', 'points': 'your_points'})
        merged_df = manager_history_df.merge(EVENT_STATS, on='gameweek')
        result_df = merged_df.rename(columns={'average_entry_score': 'average_points', 'highest_score': 'max_points'})
        return jsonify(result_df.fillna(0).to_dict(orient='records'))
    except Exception as e:
        print(f"Error fetching manager history: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/api/transfer-suggestions')
def get_transfer_suggestions():
    manager_id = request.args.get('manager_id', type=int)
    gw = request.args.get('gw', type=int)
    if not manager_id or not gw:
        return jsonify({'error': 'Manager ID and Gameweek are required'}), 400

    try:
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gw}/picks/').json()
        if 'picks' not in res:
            return jsonify({'error': 'Could not find team for this manager/gameweek.'}), 404
        
        user_squad_df = pd.DataFrame(res['picks'])
        user_player_ids = user_squad_df['element'].tolist()

        all_predictions = _get_player_predictions(gw + 1)
        
        all_predictions = all_predictions.merge(TEAMS_DF[['id', 'name']], left_on='team', right_on='id', suffixes=('', '_team'))
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        all_predictions['position'] = all_predictions['element_type'].map(pos_map)
        
        owned_players = all_predictions[all_predictions['id'].isin(user_player_ids)].copy()
        unowned_players = all_predictions[~all_predictions['id'].isin(user_player_ids)].copy()

        keep_suggestions = owned_players[owned_players['element_type'] != 1].sort_values(by='predicted_points', ascending=False).head(5)
        sell_suggestions = owned_players[owned_players['element_type'] != 1].sort_values(by='predicted_points', ascending=True).head(5)
        buy_suggestions = unowned_players[unowned_players['element_type'] != 1].sort_values(by='predicted_points', ascending=False).head(5)
        
        differentials = unowned_players[(unowned_players['element_type'] != 1) & (unowned_players['selected_by_percent'] < 3)].sort_values(by='predicted_points', ascending=False).head(5)

        output_cols = ['web_name', 'name', 'position', 'now_cost', 'predicted_points', 'avg_fdr', 'selected_by_percent']
        
        return jsonify({
            'keep': keep_suggestions[output_cols].rename(columns={'name': 'team_name', 'now_cost': 'cost'}).to_dict(orient='records'),
            'sell': sell_suggestions[output_cols].rename(columns={'name': 'team_name', 'now_cost': 'cost'}).to_dict(orient='records'),
            'buy': buy_suggestions[output_cols].rename(columns={'name': 'team_name', 'now_cost': 'cost'}).to_dict(orient='records'),
            'differentials': differentials[output_cols].rename(columns={'name': 'team_name', 'now_cost': 'cost'}).to_dict(orient='records')
        })

    except Exception as e:
        print(f"Error generating transfer suggestions: {e}")
        return jsonify({'error': 'An internal error occurred while generating suggestions.'}), 500

@app.route('/api/force-refresh', methods=['POST'])
def force_refresh():
    try:
        # We don't need to delete the R2 file, just the local one to trigger a refresh
        if os.path.exists(ALL_DATA_CACHE_FILE):
            os.remove(ALL_DATA_CACHE_FILE)
            print(f"Deleted local cache file: {ALL_DATA_CACHE_FILE}")
        
        # Reload all data, forcing a fresh download and upload to R2
        load_all_data(force_refresh=True)
        
        # Recalculate the historical predictions with the new data
        calculate_all_review_teams()
        
        return jsonify({'message': 'Data refresh successful. All player data has been updated.'})
    except Exception as e:
        print(f"Error during data refresh: {e}")
        return jsonify({'error': 'An error occurred during the data refresh process.'}), 500

def get_top_performers(df_slice, group_by_cols, agg_dict, sort_by_cols, head_n=10, ascending=False):
    """Generic helper function to get top performers from a dataframe slice."""
    if df_slice.empty:
        return pd.DataFrame()
    
    for col in agg_dict.values():
        if col[0] not in df_slice.columns:
            return pd.DataFrame(columns=group_by_cols + list(agg_dict.keys()))

    result = (
        df_slice.groupby(group_by_cols)
        .agg(**agg_dict)
        .reset_index()
        .sort_values(by=sort_by_cols, ascending=ascending)
        .head(head_n)
    )
    return result.fillna(0)


@app.route('/api/stats/pl-stats-1')
def get_pl_stats_1():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({'most_goals': [], 'most_assists': [], 'most_g_a': [], 'most_saves': []})
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")
    
    common_group = ['web_name', 'team', 'position']
    most_goals = get_top_performers(df_slice, common_group, {'goals_scored': ('goals_scored', 'sum'), 'expected_goals': ('expected_goals', 'sum')}, ['goals_scored', 'expected_goals'])
    most_assists = get_top_performers(df_slice, common_group, {'assists': ('assists', 'sum'), 'expected_assists': ('expected_assists', 'sum')}, ['assists', 'expected_assists'])
    most_g_a = get_top_performers(df_slice, common_group, {'goals_n_assists': ('goals_n_assists', 'sum'), 'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, ['goals_n_assists', 'expected_goal_involvements'])
    most_saves = get_top_performers(df_slice, common_group, {'saves': ('saves', 'sum'), 'total_points': ('total_points', 'sum')}, ['saves', 'total_points'])

    return jsonify({
        'most_goals': most_goals.to_dict(orient='records'), 
        'most_assists': most_assists.to_dict(orient='records'), 
        'most_g_a': most_g_a.to_dict(orient='records'), 
        'most_saves': most_saves.to_dict(orient='records')
    })

@app.route('/api/stats/pl-stats-2')
def get_pl_stats_2():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({'most_bps': [], 'expected_goals': [], 'expected_assists': [], 'expected_gi': []})
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")

    common_group = ['web_name', 'team', 'position']
    most_bps = get_top_performers(df_slice, common_group, {'bps': ('bps', 'sum'), 'total_points': ('total_points', 'sum')}, ['bps', 'total_points'])
    expected_goals = get_top_performers(df_slice, common_group, {'expected_goals': ('expected_goals', 'sum'), 'total_points': ('total_points', 'sum')}, ['expected_goals', 'total_points'])
    expected_assists = get_top_performers(df_slice, common_group, {'expected_assists': ('expected_assists', 'sum'), 'total_points': ('total_points', 'sum')}, ['expected_assists', 'total_points'])
    expected_gi = get_top_performers(df_slice, common_group, {'expected_goal_involvements': ('expected_goal_involvements', 'sum'), 'total_points': ('total_points', 'sum')}, ['expected_goal_involvements', 'total_points'])

    return jsonify({
        'most_bps': most_bps.to_dict(orient='records'), 
        'expected_goals': expected_goals.to_dict(orient='records'), 
        'expected_assists': expected_assists.to_dict(orient='records'), 
        'expected_gi': expected_gi.to_dict(orient='records')
    })

@app.route('/api/stats/best-players-by-position')
def get_best_players():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({'top_forwards': [], 'top_midfielders': [], 'top_defenders': [], 'top_goalkeepers': []})
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")

    def get_top_by_pos(pos):
        pos_df = df_slice.query("position==@pos")
        return get_top_performers(pos_df, ['web_name', 'team', 'position'], {'total_points': ('total_points', 'sum'), 'ict_index': ('ict_index', 'sum')}, ['total_points', 'ict_index'])

    return jsonify({
        'top_forwards': get_top_by_pos('FWD').to_dict(orient='records'), 
        'top_midfielders': get_top_by_pos('MID').to_dict(orient='records'), 
        'top_defenders': get_top_by_pos('DEF').to_dict(orient='records'), 
        'top_goalkeepers': get_top_by_pos('GKP').to_dict(orient='records')
    })
    
@app.route('/api/stats/best-teams')
def get_best_teams():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({'top_attack': [], 'worst_attack': [], 'top_defence': [], 'worst_defence': []})
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")

    attack_df = df_slice.query("position in ['MID', 'FWD']")
    defence_df = df_slice.query("position in ['DEF', 'GKP']")

    top_attack = get_top_performers(attack_df, ['team'], {'total_points': ('total_points', 'sum'), 'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, ['total_points'])
    worst_attack = get_top_performers(attack_df, ['team'], {'total_points': ('total_points', 'sum'), 'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, ['total_points'], ascending=True)
    top_defence = get_top_performers(defence_df, ['team'], {'total_points': ('total_points', 'sum'), 'expected_goals_conceded': ('expected_goals_conceded', 'sum')}, ['total_points'])
    worst_defence = get_top_performers(defence_df, ['team'], {'total_points': ('total_points', 'sum'), 'expected_goals_conceded': ('expected_goals_conceded', 'sum')}, ['total_points'], ascending=True)

    return jsonify({
        'top_attack': top_attack.to_dict(orient='records'), 
        'worst_attack': worst_attack.to_dict(orient='records'), 
        'top_defence': top_defence.to_dict(orient='records'), 
        'worst_defence': worst_defence.to_dict(orient='records')
    })

@app.route('/api/stats/team-ict-xgi')
def get_team_ict_xgi():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({'top_attack_ict': [], 'top_defence_ict': [], 'top_attack_xgi': [], 'top_defence_xgc': []})
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")

    attack_df = df_slice.query("position in ['MID', 'FWD']")
    defence_df = df_slice.query("position in ['DEF', 'GKP']")

    top_attack_ict = get_top_performers(attack_df, ['team'], {'ict_index': ('ict_index', 'sum')}, ['ict_index'])
    top_defence_ict = get_top_performers(defence_df, ['team'], {'ict_index': ('ict_index', 'sum')}, ['ict_index'])
    top_attack_xgi = get_top_performers(attack_df, ['team'], {'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, ['expected_goal_involvements'])
    top_defence_xgc = get_top_performers(defence_df, ['team'], {'expected_goals_conceded': ('expected_goals_conceded', 'sum')}, ['expected_goals_conceded'], ascending=True)

    return jsonify({
        'top_attack_ict': top_attack_ict.to_dict(orient='records'),
        'top_defence_ict': top_defence_ict.to_dict(orient='records'),
        'top_attack_xgi': top_attack_xgi.to_dict(orient='records'),
        'top_defence_xgc': top_defence_xgc.to_dict(orient='records')
    })

@app.route('/api/stats/player-ict-by-position')
def get_player_ict_by_position():
    gw, lag = get_request_args()
    df = ALL_GAMEWEEK_DATA
    if df.empty: return jsonify({
        'top_gkp_ict': [], 
        'top_def_ict': [], 
        'top_mid_ict': [], 
        'top_fwd_ict': []
    })
    start_gw = gw - lag
    df_slice = df.query("round <= @gw and round >= @start_gw")

    def get_top_ict_by_pos(pos):
        pos_df = df_slice.query("position==@pos")
        return get_top_performers(pos_df, ['web_name', 'team', 'position'], {'ict_index': ('ict_index', 'sum'), 'total_points': ('total_points', 'sum')}, ['ict_index', 'total_points'])

    return jsonify({
        'top_gkp_ict': get_top_ict_by_pos('GKP').to_dict(orient='records'), 
        'top_def_ict': get_top_ict_by_pos('DEF').to_dict(orient='records'), 
        'top_mid_ict': get_top_ict_by_pos('MID').to_dict(orient='records'), 
        'top_fwd_ict': get_top_ict_by_pos('FWD').to_dict(orient='records')
    })

@app.route('/api/download-stats')
def download_stats():
    gw, lag = get_request_args()
    start_gw = gw - lag
    df_slice = ALL_GAMEWEEK_DATA.query("round <= @gw and round >= @start_gw")

    try:
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        sheet_name = 'FPL_Stats'
        startrow = 0

        # --- Player Stats ---
        player_stats_to_generate = [
            {'title': 'Most Goals', 'agg': {'goals_scored': ('goals_scored', 'sum'), 'expected_goals': ('expected_goals', 'sum')}, 'sort': ['goals_scored', 'expected_goals']},
            {'title': 'Most Assists', 'agg': {'assists': ('assists', 'sum'), 'expected_assists': ('expected_assists', 'sum')}, 'sort': ['assists', 'expected_assists']},
            {'title': 'Most G+A', 'agg': {'goals_n_assists': ('goals_n_assists', 'sum'), 'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, 'sort': ['goals_n_assists', 'expected_goal_involvements']},
            {'title': 'Most Saves', 'agg': {'saves': ('saves', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['saves', 'total_points']},
            {'title': 'Most BPS', 'agg': {'bps': ('bps', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['bps', 'total_points']},
            {'title': 'Highest ICT', 'agg': {'ict_index': ('ict_index', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['ict_index', 'total_points']},
            {'title': 'Highest xG', 'agg': {'expected_goals': ('expected_goals', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['expected_goals', 'total_points']},
            {'title': 'Highest xA', 'agg': {'expected_assists': ('expected_assists', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['expected_assists', 'total_points']},
            {'title': 'Highest xGI', 'agg': {'expected_goal_involvements': ('expected_goal_involvements', 'sum'), 'total_points': ('total_points', 'sum')}, 'sort': ['expected_goal_involvements', 'total_points']}
        ]
        
        common_group = ['web_name', 'team', 'position']
        for stat in player_stats_to_generate:
            df = get_top_performers(df_slice, common_group, stat['agg'], stat['sort'])
            if not df.empty:
                pd.DataFrame([stat['title']]).to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, index=False)
                startrow += len(df) + 4

        # --- Player ICT by Position ---
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = df_slice.query("position==@pos")
            df = get_top_performers(pos_df, ['web_name', 'team', 'position'], {'ict_index': ('ict_index', 'sum'), 'total_points': ('total_points', 'sum')}, ['ict_index', 'total_points'])
            if not df.empty:
                title = f'Top {pos} (ICT)'
                pd.DataFrame([title]).to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, index=False)
                startrow += len(df) + 4

        # --- Team Stats ---
        attack_df = df_slice.query("position in ['MID', 'FWD']")
        defence_df = df_slice.query("position in ['DEF', 'GKP']")
        
        team_stats_to_generate = [
            {'title': 'Top Attack (Points)', 'df': attack_df, 'agg': {'total_points': ('total_points', 'sum'), 'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, 'sort': ['total_points']},
            {'title': 'Top Defence (Points)', 'df': defence_df, 'agg': {'total_points': ('total_points', 'sum'), 'expected_goals_conceded': ('expected_goals_conceded', 'sum')}, 'sort': ['total_points']},
            {'title': 'Top Attack (ICT)', 'df': attack_df, 'agg': {'ict_index': ('ict_index', 'sum')}, 'sort': ['ict_index']},
            {'title': 'Top Defence (ICT)', 'df': defence_df, 'agg': {'ict_index': ('ict_index', 'sum')}, 'sort': ['ict_index']},
            {'title': 'Top Attack (xGI)', 'df': attack_df, 'agg': {'expected_goal_involvements': ('expected_goal_involvements', 'sum')}, 'sort': ['expected_goal_involvements']},
            {'title': 'Best Defence (xGC)', 'df': defence_df, 'agg': {'expected_goals_conceded': ('expected_goals_conceded', 'sum')}, 'sort': ['expected_goals_conceded'], 'ascending': True},
        ]

        for stat in team_stats_to_generate:
            df = get_top_performers(stat['df'], ['team'], stat['agg'], stat['sort'], ascending=stat.get('ascending', False))
            if not df.empty:
                pd.DataFrame([stat['title']]).to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, index=False)
                startrow += len(df) + 4

        writer.close()
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name=f'fpl_stats_gw{gw}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"Error generating Excel file: {e}")
        return jsonify({'error': 'Failed to generate Excel file.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1927, debug=True)
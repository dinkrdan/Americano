from flask import Flask, render_template, request, jsonify, session
import os
import random
from collections import defaultdict
import itertools

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

class TournamentGenerator:
    def __init__(self):
        self.partnership_history = defaultdict(int)
        self.opponent_history = defaultdict(int)
        
    def create_default_players(self, total_players):
        """Create balanced default players with mixed genders and ratings"""
        # Fixed name lists - corrected gender assignments
        first_names_male = ['Alex', 'Ben', 'Charlie', 'Dan', 'Ethan', 'Gavin', 'Henry', 'Ian', 'Jack', 'Kyle', 'Liam', 'Mason', 'Noah', 'Owen', 'Paul']
        first_names_female = ['Alice', 'Beth', 'Clara', 'Diana', 'Eve', 'Fiona', 'Grace', 'Holly', 'Iris', 'Julia', 'Kate', 'Luna', 'Maya', 'Nina', 'Olivia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Taylor', 'Wilson', 'Moore', 'Jackson', 'White']
        
        players = []
        
        # Calculate how many of each gender to create
        num_female = total_players // 2
        num_male = total_players - num_female
        
        # Create female players
        for i in range(num_female):
            rating = round(random.uniform(3.0, 4.5), 1)
            
            # Generate unique name
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                first_name = random.choice(first_names_female)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                # Check if name is unique
                if not any(p['name'] == full_name for p in players):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: use index to ensure uniqueness
                full_name = f"{random.choice(first_names_female)} {random.choice(last_names)}{i+1}"
                first_name = full_name.split()[0]
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name,
                'gender': 'F',
                'rating': rating
            })
        
        # Create male players
        for i in range(num_male):
            rating = round(random.uniform(3.0, 4.5), 1)
            
            # Generate unique name
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                first_name = random.choice(first_names_male)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                # Check if name is unique
                if not any(p['name'] == full_name for p in players):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: use index to ensure uniqueness
                full_name = f"{random.choice(first_names_male)} {random.choice(last_names)}{i+1}"
                first_name = full_name.split()[0]
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name,
                'gender': 'M',
                'rating': rating
            })
        
        return players

    def validate_round_assignments(self, matches):
        """Validate that no player appears more than once in a round"""
        all_players_in_round = []
        
        for match in matches:
            team_a, team_b = match
            for player in team_a + team_b:
                player_id = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                all_players_in_round.append(player_id)
        
        # Check for duplicates
        seen = set()
        duplicates = set()
        for player_id in all_players_in_round:
            if player_id in seen:
                duplicates.add(player_id)
            seen.add(player_id)
        
        if duplicates:
            print(f"ERROR: Found duplicate player assignments in round: {duplicates}")
            return False, duplicates
        
        print(f"VALIDATION: Round assignments valid - {len(all_players_in_round)} unique players assigned")
        return True, None

    def generate_enhanced_tournament(self, courts, players_list, rounds, skip_players=None, avoid_mm_vs_ff=True, use_rating_balance=True, rating_factor=3):
        """Enhanced tournament generation with skip players and advanced options"""
        if skip_players is None:
            skip_players = []
        
        print(f"DEBUG: generate_enhanced_tournament called with {courts} courts, {len(players_list)} players")
        
        # Filter out skipped players
        available_players = [p for p in players_list if p.get('name', f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()) not in skip_players]
        
        total_courts = courts
        players_per_round = total_courts * 4
        
        print(f"DEBUG: Need {players_per_round} players for {total_courts} courts, have {len(available_players)} available")
        
        if len(available_players) < players_per_round:
            return {"error": f"Not enough players available. Need {players_per_round}, have {len(available_players)}"}
        
        # Multiple attempts to generate valid assignments
        max_attempts = 10
        for attempt in range(max_attempts):
            print(f"DEBUG: Generation attempt {attempt + 1}")
            
            # Select players for this round
            round_players = available_players[:players_per_round].copy()  # Make a copy to avoid modifying original
            sit_outs = [p.get('name', f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()) for p in available_players[players_per_round:]]
            
            # Shuffle players for randomness
            random.shuffle(round_players)
            
            print(f"DEBUG: Selected {len(round_players)} players for round")
            print(f"DEBUG: Players: {[p.get('name', p.get('firstName', 'Unknown')) for p in round_players]}")
            
            # Generate matches for this round
            matches = []
            
            # Assign players to courts using non-overlapping slices
            for court in range(total_courts):
                start_idx = court * 4
                team_a = round_players[start_idx:start_idx + 2]
                team_b = round_players[start_idx + 2:start_idx + 4]
                
                print(f"DEBUG: Court {court + 1} - Team A: {[p.get('name', p.get('firstName', 'Unknown')) for p in team_a]}")
                print(f"DEBUG: Court {court + 1} - Team B: {[p.get('name', p.get('firstName', 'Unknown')) for p in team_b]}")
                
                # Apply gender balancing if requested
                if avoid_mm_vs_ff:
                    team_a_genders = [p.get('gender', 'M') for p in team_a]
                    team_b_genders = [p.get('gender', 'M') for p in team_b]
                    
                    # If one team is all male and other all female, try to swap within the match
                    if (all(g == 'M' for g in team_a_genders) and all(g == 'F' for g in team_b_genders)) or \
                       (all(g == 'F' for g in team_a_genders) and all(g == 'M' for g in team_b_genders)):
                        # Swap one player from each team
                        team_a[1], team_b[1] = team_b[1], team_a[1]
                        print(f"DEBUG: Applied gender balancing swap for court {court + 1}")
                
                # Apply rating balancing if requested (only within the same match)
                if use_rating_balance:
                    team_a_rating = sum(p.get('rating', 3.5) for p in team_a)
                    team_b_rating = sum(p.get('rating', 3.5) for p in team_b)
                    
                    rating_diff = abs(team_a_rating - team_b_rating)
                    if rating_diff > 1.0:  # Threshold for rebalancing
                        # Try swapping players within this match to balance
                        best_diff = rating_diff
                        best_swap = None
                        
                        for i in range(2):
                            for j in range(2):
                                # Create temporary teams for testing
                                temp_team_a = team_a.copy()
                                temp_team_b = team_b.copy()
                                temp_team_a[i], temp_team_b[j] = temp_team_b[j], temp_team_a[i]
                                
                                new_a_rating = sum(p.get('rating', 3.5) for p in temp_team_a)
                                new_b_rating = sum(p.get('rating', 3.5) for p in temp_team_b)
                                new_diff = abs(new_a_rating - new_b_rating)
                                
                                if new_diff < best_diff:
                                    best_diff = new_diff
                                    best_swap = (i, j)
                        
                        if best_swap:
                            i, j = best_swap
                            team_a[i], team_b[j] = team_b[j], team_a[i]
                            print(f"DEBUG: Applied rating balancing swap for court {court + 1}")
                
                matches.append([team_a, team_b])
            
            # Validate the assignments
            is_valid, duplicates = self.validate_round_assignments(matches)
            
            if is_valid:
                print(f"DEBUG: Successfully generated valid round on attempt {attempt + 1}")
                return {
                    "matches": matches,
                    "sit_outs": sit_outs
                }
            else:
                print(f"DEBUG: Attempt {attempt + 1} failed - found duplicates: {duplicates}")
                # Continue to next attempt
                continue
        
        # If we get here, all attempts failed
        print(f"ERROR: Failed to generate valid round after {max_attempts} attempts")
        return {"error": f"Failed to generate valid player assignments after {max_attempts} attempts"}

    def generate_simple_tournament(self, courts, players_list, rounds):
        """
        Generate complete tournament schedule
        """
        print(f"DEBUG: Generating tournament with {len(players_list)} players, {courts} courts, {rounds} rounds")
        
        # Validate input
        players_per_round = courts * 4
        if len(players_list) < players_per_round:
            return {"error": f"Not enough players. Need at least {players_per_round} players for {courts} courts."}
        
        # Initialize partnership history
        self.partnership_history = defaultdict(int)
        
        schedule = []
        for round_num in range(rounds):
            print(f"DEBUG: Generating round {round_num + 1} of {rounds}")
            round_data = {
                "round": round_num + 1,
                "matches": [],
                "sit_outs": []
            }
            
            # For initial tournament generation, no skips
            result = self.generate_enhanced_tournament(courts, players_list, 1, skip_players=[])
            
            if 'error' in result:
                print(f"DEBUG: Error generating round {round_num + 1}: {result['error']}")
                return {"error": result['error']}
            
            round_data["matches"] = result["matches"]
            round_data["sit_outs"] = result["sit_outs"]
            
            schedule.append(round_data)
            print(f"DEBUG: Round {round_num + 1} added to schedule")
        
        print(f"DEBUG: Tournament generation complete - {len(schedule)} rounds in schedule")
        return {
            "success": True,
            "schedule": schedule,
            "players": players_list
        }

# Global tournament generator instance
tournament_gen = TournamentGenerator()

# Health check endpoint for Railway deployment
@app.route('/api/test')
def health_check():
    return jsonify({"status": "healthy", "message": "DiNKR Tournament System is running"}), 200

# Debug route to check template files
@app.route('/debug')
def debug():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug Test</title>
        <style>body { font-family: Arial; padding: 20px; background: white; }</style>
    </head>
    <body>
        <h1>Debug Test Page</h1>
        <p>If you can see this, Flask template rendering works!</p>
        <p>Template folder exists: ''' + str(os.path.exists(os.path.join(app.root_path, 'templates'))) + '''</p>
        <p>Index.html exists: ''' + str(os.path.exists(os.path.join(app.root_path, 'templates', 'index.html'))) + '''</p>
        <p>App root path: ''' + str(app.root_path) + '''</p>
        <p>Current working directory: ''' + str(os.getcwd()) + '''</p>
    </body>
    </html>
    '''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_tournament', methods=['POST'])
def generate_tournament():
    try:
        data = request.json
        courts = data.get('courts', 2)
        rounds = data.get('rounds', 6)
        use_defaults = data.get('useDefaults', True)
        avoid_mm_vs_ff = data.get('avoidMMvsFF', True)
        use_rating_balance = data.get('useRatingBalance', True)
        rating_factor = data.get('ratingFactor', 3)
        round_duration = data.get('roundDuration', 13)
        total_players = data.get('totalPlayers', 8)
        
        print(f"DEBUG: Tournament request - courts: {courts}, rounds: {rounds}, players: {total_players}")
        print(f"DEBUG: Advanced options - avoid MM/FF: {avoid_mm_vs_ff}, rating balance: {use_rating_balance}")
        
        # Store configuration in session
        session['config'] = {
            'courts': courts,
            'rounds': rounds,
            'avoidMMvsFF': avoid_mm_vs_ff,
            'useRatingBalance': use_rating_balance,
            'ratingFactor': rating_factor,
            'roundDuration': round_duration,
            'totalPlayers': total_players
        }
        
        if use_defaults:
            players_list = tournament_gen.create_default_players(total_players)
            print(f"DEBUG: Generated {len(players_list)} default players")
            for i, player in enumerate(players_list):
                print(f"  Player {i+1}: {player.get('name')} ({player.get('gender')}, {player.get('rating')})")
        else:
            players_list = data.get('players', [])
            if len(players_list) != total_players:
                return jsonify({"error": f"Expected {total_players} players, got {len(players_list)}"}), 400
        
        result = tournament_gen.generate_simple_tournament(courts, players_list, rounds)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Store tournament and initialize session
        session['tournament'] = result
        session['current_round'] = 0
        session['scores'] = {}
        session.modified = True
        
        print(f"DEBUG: Tournament generated successfully with {len(result['schedule'])} rounds")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_tournament_state', methods=['GET'])
def get_tournament_state():
    try:
        tournament = session.get('tournament')
        current_round = session.get('current_round', 0)
        scores = session.get('scores', {})
        
        return jsonify({
            "tournament": tournament,
            "current_round": current_round,
            "scores": scores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_score', methods=['POST'])
def update_score():
    try:
        data = request.json
        round_index = data['roundIndex']
        match_index = data['matchIndex']
        team = data['team']
        score = data['score']
        
        # Initialize scores structure if needed
        if 'scores' not in session:
            session['scores'] = {}
        
        if str(round_index) not in session['scores']:
            session['scores'][str(round_index)] = {}
        
        if str(match_index) not in session['scores'][str(round_index)]:
            session['scores'][str(round_index)][str(match_index)] = {}
        
        # Update the score
        session['scores'][str(round_index)][str(match_index)][team] = score
        session.modified = True
        
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"ERROR updating score: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/apply_player_switches', methods=['POST'])
def apply_player_switches():
    try:
        data = request.json
        switches = data.get('switches', [])
        round_index = data.get('roundIndex', session.get('current_round', 0))
        
        tournament = session.get('tournament', {})
        
        if not tournament or 'schedule' not in tournament:
            return jsonify({"error": "No tournament data found"}), 400
        
        if round_index >= len(tournament['schedule']):
            return jsonify({"error": "Invalid round index"}), 400
        
        # Apply all switches to the specified round
        current_matches = tournament['schedule'][round_index]['matches']
        
        # Create a mapping of player names to player objects for quick lookup
        player_map = {}
        for player in tournament['players']:
            # Handle both name formats
            if 'name' in player and player['name']:
                player_map[player['name']] = player
            elif 'firstName' in player:
                full_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                if full_name:
                    player_map[full_name] = player
                    player_map[player['firstName']] = player  # Also map by first name only
        
        print(f"DEBUG: Available players in map: {list(player_map.keys())}")
        print(f"DEBUG: Applying {len(switches)} switches: {switches}")
        
        # Apply each switch
        switches_applied = 0
        for switch in switches:
            old_player_name = switch['oldPlayer']
            new_player_name = switch['newPlayer']
            
            print(f"DEBUG: Switching {old_player_name} -> {new_player_name}")
            
            if new_player_name not in player_map:
                print(f"ERROR: New player '{new_player_name}' not found in player map")
                print(f"Available players: {list(player_map.keys())}")
                continue
                
            new_player = player_map[new_player_name]
            
            # Find and replace the player in matches
            switch_applied = False
            for match_idx, match in enumerate(current_matches):
                team_a, team_b = match
                
                # Check team A
                for i, player in enumerate(team_a):
                    player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                    if player_name == old_player_name:
                        team_a[i] = new_player
                        switch_applied = True
                        switches_applied += 1
                        print(f"DEBUG: Replaced {old_player_name} with {new_player_name} in Team A of match {match_idx}")
                        break
                
                if switch_applied:
                    break
                
                # Check team B
                for i, player in enumerate(team_b):
                    player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                    if player_name == old_player_name:
                        team_b[i] = new_player
                        switch_applied = True
                        switches_applied += 1
                        print(f"DEBUG: Replaced {old_player_name} with {new_player_name} in Team B of match {match_idx}")
                        break
                
                if switch_applied:
                    break
            
            if not switch_applied:
                print(f"WARNING: Could not find player {old_player_name} to replace")
        
        # After applying switches, validate the round
        is_valid, duplicates = tournament_gen.validate_round_assignments(current_matches)
        if not is_valid:
            return jsonify({"error": f"Player switches would create duplicate assignments: {list(duplicates)}"}), 400
        
        if switches_applied == 0:
            return jsonify({"error": "No switches were applied. Check player names match exactly."}), 400
        
        # Update sit-outs based on new assignments
        all_playing = set()
        for match in current_matches:
            team_a, team_b = match
            for player in team_a + team_b:
                player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                all_playing.add(player_name)
        
        all_players = set()
        for player in tournament['players']:
            player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
            all_players.add(player_name)
        
        sitting_out = list(all_players - all_playing)
        tournament['schedule'][round_index]['sit_outs'] = sitting_out
        
        print(f"DEBUG: Updated sit-outs: {sitting_out}")
        
        # Update session
        session['tournament'] = tournament
        session.modified = True
        
        print(f"DEBUG: Successfully applied {switches_applied} out of {len(switches)} switches")
        return jsonify({"success": True, "applied_switches": switches_applied, "total_switches": len(switches)})
        
    except Exception as e:
        print(f"ERROR in apply_player_switches: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to apply switches: {str(e)}"}), 500

@app.route('/api/advance_round', methods=['POST'])
def advance_round():
    try:
        data = request.json
        skip_players = data.get('skipPlayers', [])
        player_switches = data.get('playerSwitches', [])
        
        current = session.get('current_round', 0)
        tournament = session.get('tournament', {})
        config = session.get('config', {})
        
        print(f"DEBUG: advance_round called - current_round={current}")
        print(f"DEBUG: tournament has {len(tournament.get('schedule', []))} rounds in schedule")
        print(f"DEBUG: skip_players={skip_players}, switches={len(player_switches)}")
        
        total_rounds = len(tournament.get('schedule', []))
        next_round = current + 1
        
        print(f"DEBUG: next_round={next_round}, total_rounds={total_rounds}")
        
        if next_round >= total_rounds:
            print(f"DEBUG: Tournament completed - {next_round} >= {total_rounds}")
            return jsonify({"completed": True})
        
        # Generate matches for next round considering skips
        result = tournament_gen.generate_enhanced_tournament(
            courts=config.get('courts', 2),
            players_list=tournament['players'],
            rounds=1,
            skip_players=skip_players,
            avoid_mm_vs_ff=config.get('avoidMMvsFF', True),
            use_rating_balance=config.get('useRatingBalance', True),
            rating_factor=config.get('ratingFactor', 3)
        )
        
        if 'error' in result:
            return jsonify({"error": result['error']}), 400
        
        # Update the tournament schedule for the next round
        next_round_data = {
            "round": next_round + 1,
            "matches": result["matches"],
            "sit_outs": result["sit_outs"]
        }
        
        # Replace the next round in the schedule
        tournament['schedule'][next_round] = next_round_data
        print(f"DEBUG: Updated round {next_round + 1} in schedule")
        
        # Apply any manual player switches for the next round
        if player_switches:
            print(f"DEBUG: Applying {len(player_switches)} switches to round {next_round}")
            # Apply switches to the newly generated round
            current_matches = tournament['schedule'][next_round]['matches']
            player_map = {}
            for player in tournament['players']:
                if 'name' in player and player['name']:
                    player_map[player['name']] = player
                elif 'firstName' in player:
                    full_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                    if full_name:
                        player_map[full_name] = player
                        player_map[player['firstName']] = player
            
            for switch in player_switches:
                old_player_name = switch['oldPlayer']
                new_player_name = switch['newPlayer']
                
                if new_player_name not in player_map:
                    continue
                    
                new_player = player_map[new_player_name]
                
                # Find and replace the player in matches
                for match in current_matches:
                    team_a, team_b = match
                    
                    # Check team A
                    for i, player in enumerate(team_a):
                        player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                        if player_name == old_player_name:
                            team_a[i] = new_player
                            break
                    
                    # Check team B
                    for i, player in enumerate(team_b):
                        player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                        if player_name == old_player_name:
                            team_b[i] = new_player
                            break
            
            # Validate after applying switches
            is_valid, duplicates = tournament_gen.validate_round_assignments(current_matches)
            if not is_valid:
                return jsonify({"error": f"Round advancement failed: duplicate player assignments {list(duplicates)}"}), 400
        
        # Update session
        session['tournament'] = tournament
        session['current_round'] = next_round
        session.modified = True
        
        print(f"DEBUG: Successfully advanced to round {next_round + 1}")
        print(f"DEBUG: Current round set to {next_round}")
        print(f"DEBUG: Sitting out: {result['sit_outs']}")
        
        return jsonify({"success": True, "round": next_round})
        
    except Exception as e:
        print(f"ERROR in advance_round: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/calculate_results', methods=['GET'])
def calculate_results():
    try:
        tournament = session.get('tournament', {})
        scores = session.get('scores', {})
        
        if not tournament or 'players' not in tournament:
            return jsonify({"error": "No tournament data found"}), 400
        
        # Calculate stats for each player
        player_stats = {}
        
        # Initialize player stats
        for player in tournament['players']:
            player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
            player_stats[player_name] = {
                'name': player_name,
                'firstName': player.get('firstName', ''),
                'lastName': player.get('lastName', ''),
                'gender': player.get('gender', 'M'),
                'rating': player.get('rating', 3.5),
                'totalScore': 0,
                'wins': 0,
                'losses': 0,
                'matchesPlayed': 0
            }
        
        # Process each round's scores
        for round_index, round_scores in scores.items():
            round_data = tournament['schedule'][int(round_index)]
            
            for match_index, match_scores in round_scores.items():
                match = round_data['matches'][int(match_index)]
                team_a, team_b = match
                
                if 'teamA' in match_scores and 'teamB' in match_scores:
                    score_a = int(match_scores['teamA']) if match_scores['teamA'] else 0
                    score_b = int(match_scores['teamB']) if match_scores['teamB'] else 0
                    
                    # Add scores to each player
                    for player in team_a:
                        player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                        if player_name in player_stats:
                            player_stats[player_name]['totalScore'] += score_a
                            player_stats[player_name]['matchesPlayed'] += 1
                            if score_a > score_b:
                                player_stats[player_name]['wins'] += 1
                            else:
                                player_stats[player_name]['losses'] += 1
                    
                    for player in team_b:
                        player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                        if player_name in player_stats:
                            player_stats[player_name]['totalScore'] += score_b
                            player_stats[player_name]['matchesPlayed'] += 1
                            if score_b > score_a:
                                player_stats[player_name]['wins'] += 1
                            else:
                                player_stats[player_name]['losses'] += 1
        
        # Convert to list and return
        results = list(player_stats.values())
        return jsonify(results)
        
    except Exception as e:
        print(f"ERROR in calculate_results: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
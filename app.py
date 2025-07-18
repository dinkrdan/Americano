from flask import Flask, render_template, request, jsonify, session
import os
import random
import itertools
from collections import defaultdict, Counter
import math

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

class AdvancedTournamentGenerator:
    def __init__(self):
        # Advanced tracking structures
        self.partnership_count = defaultdict(int)
        self.opponent_count = defaultdict(int)
        self.matchups_count = defaultdict(int)
        self.sit_out_history = []
        self.sit_out_count = defaultdict(int)
        self.partners = defaultdict(set)
        self.opponents = defaultdict(set)
        self.opponent_pairs = defaultdict(set)
        
        # Gender mapping for sophisticated MM vs FF logic
        self.gender_map = {}
        
    def create_default_players(self, total_players):
        """Create balanced default players with mixed genders and ratings"""
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
            
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                first_name = random.choice(first_names_female)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                if not any(p['name'] == full_name for p in players):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                full_name = f"{random.choice(first_names_female)} {random.choice(last_names)}{i+1}"
                first_name = full_name.split()[0]
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name,
                'gender': 'F',
                'rating': rating
            })
            self.gender_map[full_name] = 'F'
        
        # Create male players
        for i in range(num_male):
            rating = round(random.uniform(3.0, 4.5), 1)
            
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                first_name = random.choice(first_names_male)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                if not any(p['name'] == full_name for p in players):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                full_name = f"{random.choice(first_names_male)} {random.choice(last_names)}{i+1}"
                first_name = full_name.split()[0]
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name,
                'gender': 'M',
                'rating': rating
            })
            self.gender_map[full_name] = 'M'
        
        return players

    def initialize_tracking(self, players):
        """Initialize advanced tracking structures"""
        player_names = [p['name'] for p in players]
        
        # Initialize partnership tracking
        self.partnership_count = {frozenset(pair): 0 for pair in itertools.combinations(player_names, 2)}
        self.opponent_count = {frozenset(pair): 0 for pair in itertools.combinations(player_names, 2)}
        self.matchups_count = defaultdict(int)
        self.sit_out_count = {p: 0 for p in player_names}
        
        # Initialize advanced tracking
        self.partners = {p: set() for p in player_names}
        self.opponents = {p: set() for p in player_names}
        self.opponent_pairs = {p: set() for p in player_names}
        
        # Build gender map
        for player in players:
            self.gender_map[player['name']] = player['gender']

    def all_partnerships_covered(self):
        """Check if all possible partnerships have been tried"""
        return all(count > 0 for count in self.partnership_count.values())

    def all_opponents_covered(self):
        """Check if all possible opponent pairings have been tried"""
        return all(count > 0 for count in self.opponent_count.values())

    def can_partner(self, a, b):
        """Check if two players can partner together"""
        pair = frozenset((a, b))
        if self.partnership_count[pair] == 0:
            return True
        else:
            return self.all_partnerships_covered()

    def can_face(self, a, b):
        """Check if two players can face each other"""
        pair = frozenset((a, b))
        if self.opponent_count[pair] == 0:
            return True
        else:
            return self.all_opponents_covered()

    def matchup_key(self, teamA, teamB):
        """Generate a unique key for a team matchup"""
        tA = frozenset(teamA)
        tB = frozenset(teamB)
        return frozenset((tA, tB))

    def times_matchup_played(self, teamA, teamB):
        """Count how many times this exact matchup has been played"""
        key = self.matchup_key(teamA, teamB)
        return self.matchups_count.get(key, 0)

    def rating_difference_penalty(self, teamA, teamB, ratings):
        """Calculate the absolute difference in combined team ratings"""
        teamA_rating = sum(ratings.get(p, 3.5) for p in teamA)
        teamB_rating = sum(ratings.get(p, 3.5) for p in teamB)
        diff = abs(teamA_rating - teamB_rating)
        return diff

    def is_mm_vs_ff(self, teamA, teamB):
        """Check if this is an all-male vs all-female matchup"""
        teamA_genders = [self.gender_map.get(p, 'M') for p in teamA]
        teamB_genders = [self.gender_map.get(p, 'M') for p in teamB]

        is_teamA_all_m = all(g == 'M' for g in teamA_genders)
        is_teamA_all_f = all(g == 'F' for g in teamA_genders)
        is_teamB_all_m = all(g == 'M' for g in teamB_genders)
        is_teamB_all_f = all(g == 'F' for g in teamB_genders)

        return (is_teamA_all_m and is_teamB_all_f) or (is_teamA_all_f and is_teamB_all_m)

    def has_consecutive_opponents(self, teamA, teamB, round_index):
        """Check if any player would face the same opponent in consecutive rounds"""
        if round_index == 0 or len(self.sit_out_history) < round_index:
            return False
            
        # Get last round's matches from sit_out_history structure
        # This is a simplified check - in full implementation would need complete match history
        for playerA in teamA:
            for playerB in teamB:
                # Check if these players faced each other in the previous round
                # For now, we'll use a simplified heuristic
                if len(self.opponents[playerA]) > 0 and playerB in self.opponents[playerA]:
                    # They've faced before - check if it was recent
                    # This could be enhanced with more detailed round tracking
                    return True
        return False

    def sitting_constraints_ok(self, outs, round_index):
        """Check if sitting out these players violates constraints"""
        # Avoid back-to-back sit-outs or 2 of 3 consecutive rounds
        if round_index > 0 and len(self.sit_out_history) > round_index - 1:
            last_round_out = self.sit_out_history[round_index - 1]
            for p in outs:
                if p in last_round_out:
                    return False
                    
        if round_index >= 2 and len(self.sit_out_history) > round_index - 2:
            for p in outs:
                count_in_3 = 0
                if len(self.sit_out_history) > round_index - 1 and p in self.sit_out_history[round_index - 1]:
                    count_in_3 += 1
                if len(self.sit_out_history) > round_index - 2 and p in self.sit_out_history[round_index - 2]:
                    count_in_3 += 1
                count_in_3 += 1  # Current round
                if count_in_3 >= 2:
                    return False
        return True

    def select_sit_outs(self, players, courts, round_index):
        """Intelligently select players to sit out this round"""
        extra = len(players) - 4 * courts
        if extra <= 0:
            return []

        player_names = [p['name'] for p in players]
        
        # Sort players by current sit-outs (ascending) + random tie-break
        sorted_players_by_sit = sorted(player_names, key=lambda p: (self.sit_out_count[p], random.random()))
        candidate_combos = list(itertools.combinations(sorted_players_by_sit, extra))
        random.shuffle(candidate_combos)

        valid_combos = []
        for combo in candidate_combos:
            if self.sitting_constraints_ok(combo, round_index):
                valid_combos.append(combo)

        if not valid_combos:
            # No valid combos => fallback to first candidate
            return list(candidate_combos[0]) if candidate_combos else []

        # Evaluate each valid combo by the new distribution's balance
        best_combo = None
        best_score = float('inf')

        for combo in valid_combos:
            # Simulate new distribution
            temp_sit_out_count = self.sit_out_count.copy()
            for p in combo:
                temp_sit_out_count[p] += 1

            # Score = difference between max and min sit-out counts
            difference = max(temp_sit_out_count.values()) - min(temp_sit_out_count.values())

            if difference < best_score:
                best_score = difference
                best_combo = combo

        return list(best_combo) if best_combo else []

    def generate_round_matches(self, players, courts, round_index, avoid_mm_vs_ff=True, use_rating_balance=True, rating_factor=3):
        """COMPLETELY FIXED: Generate matches using improved algorithm that prevents duplicates"""
        
        print(f"DEBUG: Starting round {round_index + 1} generation with {len(players)} players, {courts} courts")
        
        # Get players sitting out
        sit_outs = self.select_sit_outs(players, courts, round_index)
        round_players = [p for p in players if p['name'] not in sit_outs]
        
        expected_playing = 4 * courts
        actual_playing = len(round_players)
        
        print(f"DEBUG: Expected playing: {expected_playing}, Actual playing: {actual_playing}")
        print(f"DEBUG: Sitting out: {sit_outs}")
        
        if actual_playing != expected_playing:
            return None, f"Invalid number of playing players: {actual_playing}, expected {expected_playing}"

        # Extract ratings for quick lookup
        ratings = {p['name']: p['rating'] for p in players}
        player_names = [p['name'] for p in round_players]
        
        print(f"DEBUG: Playing players: {player_names}")
        
        # COMPLETELY REWRITTEN: Use a more robust algorithm
        matches = []
        used_players = set()
        
        def find_best_match(available_players):
            """Find the best 4-player combination for a match"""
            if len(available_players) < 4:
                return None, None, float('inf')
            
            best_combo = None
            best_teams = None
            best_score = float('inf')
            
            # Try all combinations of 4 players
            for combo in itertools.combinations(available_players, 4):
                # Try all ways to split into 2 teams
                for teamA in itertools.combinations(combo, 2):
                    teamB = tuple(p for p in combo if p not in teamA)
                    
                    # Calculate score for this match configuration
                    match_score = 0
                    
                    # Partnership constraints
                    if not self.can_partner(teamA[0], teamA[1]):
                        match_score += 10
                    if not self.can_partner(teamB[0], teamB[1]):
                        match_score += 10
                    
                    # Opponent constraints
                    for a in teamA:
                        for b in teamB:
                            if not self.can_face(a, b):
                                match_score += 2
                    
                    # Rating balance
                    if use_rating_balance:
                        rating_diff = self.rating_difference_penalty(teamA, teamB, ratings)
                        match_score += rating_diff * rating_factor
                    
                    # Gender balance
                    if avoid_mm_vs_ff and self.is_mm_vs_ff(teamA, teamB):
                        rating_diff = self.rating_difference_penalty(teamA, teamB, ratings)
                        if rating_diff >= 0.5:
                            match_score += 3
                        else:
                            match_score += 1
                    
                    # Repeated matchup penalty
                    match_score += self.times_matchup_played(teamA, teamB)
                    
                    if match_score < best_score:
                        best_score = match_score
                        best_combo = combo
                        best_teams = (teamA, teamB)
            
            return best_combo, best_teams, best_score
        
        # Generate matches sequentially
        remaining_players = player_names.copy()
        
        for court_num in range(courts):
            print(f"DEBUG: Generating match for court {court_num + 1}, remaining players: {len(remaining_players)}")
            
            if len(remaining_players) < 4:
                return None, f"Not enough players remaining for court {court_num + 1}: {len(remaining_players)}"
            
            # Find best match from remaining players
            combo, teams, score = find_best_match(remaining_players)
            
            if combo is None:
                return None, f"Could not form match for court {court_num + 1}"
            
            teamA, teamB = teams
            matches.append((teamA, teamB))
            
            # Remove all 4 players from remaining pool
            for player in combo:
                remaining_players.remove(player)
                used_players.add(player)
            
            print(f"DEBUG: Court {court_num + 1}: {teamA} vs {teamB}")
            print(f"DEBUG: Remaining players after court {court_num + 1}: {len(remaining_players)}")
        
        # CRITICAL: Final validation
        all_assigned = []
        for teamA, teamB in matches:
            all_assigned.extend(teamA)
            all_assigned.extend(teamB)
        
        if len(all_assigned) != len(set(all_assigned)):
            duplicates = [p for p in all_assigned if all_assigned.count(p) > 1]
            return None, f"CRITICAL ERROR: Duplicate assignments detected: {duplicates}"
        
        if len(all_assigned) != expected_playing:
            return None, f"Wrong number of players assigned: {len(all_assigned)} vs {expected_playing}"
        
        print(f"DEBUG: Successfully generated {len(matches)} matches with no duplicates")

        # Update tracking data
        if len(self.sit_out_history) <= round_index:
            self.sit_out_history.extend([[] for _ in range(round_index + 1 - len(self.sit_out_history))])
        
        self.sit_out_history[round_index] = list(sit_outs)
        for s in sit_outs:
            self.sit_out_count[s] += 1

        # Update partnership and opponent tracking
        for (teamA, teamB) in matches:
            # Partnership tracking
            pairA = frozenset(teamA)
            pairB = frozenset(teamB)
            self.partnership_count[pairA] += 1
            self.partnership_count[pairB] += 1
            
            # Partners tracking
            self.partners[teamA[0]].add(teamA[1])
            self.partners[teamA[1]].add(teamA[0])
            self.partners[teamB[0]].add(teamB[1])
            self.partners[teamB[1]].add(teamB[0])
            
            # Opponent tracking
            for a in teamA:
                for b in teamB:
                    opp_key = frozenset((a, b))
                    self.opponent_count[opp_key] += 1
                    self.opponents[a].add(b)
                    self.opponents[b].add(a)
            
            # Opponent pairs tracking
            opp_pair_B = frozenset(teamB)
            opp_pair_A = frozenset(teamA)
            for p in teamA:
                self.opponent_pairs[p].add(opp_pair_B)
            for p in teamB:
                self.opponent_pairs[p].add(opp_pair_A)
            
            # Matchup tracking
            mk = self.matchup_key(teamA, teamB)
            self.matchups_count[mk] += 1

        # Convert tuples back to player objects for return
        matches_with_objects = []
        player_lookup = {p['name']: p for p in players}
        
        for teamA_names, teamB_names in matches:
            teamA_objects = [player_lookup[name] for name in teamA_names]
            teamB_objects = [player_lookup[name] for name in teamB_names]
            matches_with_objects.append([teamA_objects, teamB_objects])

        sit_out_names = list(sit_outs)
        
        return matches_with_objects, sit_out_names

    def generate_complete_tournament(self, courts, players_list, rounds, avoid_mm_vs_ff=True, use_rating_balance=True, rating_factor=3):
        """Generate a complete tournament with sophisticated algorithms"""
        
        self.initialize_tracking(players_list)
        
        schedule = []
        for round_num in range(rounds):
            print(f"DEBUG: Generating round {round_num + 1} of {rounds}")
            
            matches, sit_outs = self.generate_round_matches(
                players_list, courts, round_num, avoid_mm_vs_ff, use_rating_balance, rating_factor
            )
            
            if matches is None:
                return {"error": sit_outs}  # sit_outs contains error message in this case
            
            round_data = {
                "round": round_num + 1,
                "matches": matches,
                "sit_outs": sit_outs
            }
            
            schedule.append(round_data)
            print(f"DEBUG: Round {round_num + 1} completed successfully")
        
        return {
            "success": True,
            "schedule": schedule,
            "players": players_list
        }

    def calculate_dinkr_performance_analysis(self, players, scores, schedule):
        """Calculate sophisticated DiNKR performance analysis"""
        
        results = []
        
        for player in players:
            player_name = player.get('name') or f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
            player_rating = player.get('rating', 3.5)
            
            total_score = 0
            wins = 0
            losses = 0
            matches_played = 0
            predicted_total = 0
            
            # Analyze each match this player participated in
            for round_index, round_data in enumerate(schedule):
                if str(round_index) not in scores:
                    continue
                    
                round_scores = scores[str(round_index)]
                
                for match_index, match in enumerate(round_data['matches']):
                    if str(match_index) not in round_scores:
                        continue
                        
                    teamA, teamB = match
                    match_scores = round_scores[str(match_index)]
                    
                    if 'teamA' not in match_scores or 'teamB' not in match_scores:
                        continue
                    
                    try:
                        score_a = int(match_scores['teamA']) if match_scores['teamA'] else 0
                        score_b = int(match_scores['teamB']) if match_scores['teamB'] else 0
                    except (ValueError, TypeError):
                        continue
                    
                    # Check if player is in this match
                    player_team = None
                    teammate_rating = 3.5
                    opponent_ratings = []
                    
                    for p in teamA:
                        p_name = p.get('name') or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                        if p_name == player_name:
                            player_team = 'A'
                            # Find teammate rating
                            for teammate in teamA:
                                teammate_name = teammate.get('name') or f"{teammate.get('firstName', '')} {teammate.get('lastName', '')}".strip()
                                if teammate_name != player_name:
                                    teammate_rating = teammate.get('rating', 3.5)
                            # Get opponent ratings
                            opponent_ratings = [p.get('rating', 3.5) for p in teamB]
                            break
                    
                    if player_team is None:
                        for p in teamB:
                            p_name = p.get('name') or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                            if p_name == player_name:
                                player_team = 'B'
                                # Find teammate rating
                                for teammate in teamB:
                                    teammate_name = teammate.get('name') or f"{teammate.get('firstName', '')} {teammate.get('lastName', '')}".strip()
                                    if teammate_name != player_name:
                                        teammate_rating = teammate.get('rating', 3.5)
                                # Get opponent ratings
                                opponent_ratings = [p.get('rating', 3.5) for p in teamA]
                                break
                    
                    if player_team is not None:
                        matches_played += 1
                        
                        # Calculate predicted score using DiNKR algorithm
                        team_rating = player_rating + teammate_rating
                        opponent_team_rating = sum(opponent_ratings)
                        predicted_score = self.calculate_predicted_score(team_rating, opponent_team_rating)
                        predicted_total += predicted_score
                        
                        if player_team == 'A':
                            total_score += score_a
                            if score_a > score_b:
                                wins += 1
                            else:
                                losses += 1
                        else:
                            total_score += score_b
                            if score_b > score_a:
                                wins += 1
                            else:
                                losses += 1
            
            # Calculate performance metrics
            avg_score = total_score / matches_played if matches_played > 0 else 0
            win_rate = (wins / matches_played * 100) if matches_played > 0 else 0
            predicted_avg = predicted_total / matches_played if matches_played > 0 else 0
            
            # DiNKR Performance Index
            performance_index = avg_score / predicted_avg if predicted_avg > 0 else 1.0
            
            # Performance categories
            if performance_index > 1.15:
                performance_label = 'Exceptional'
            elif performance_index > 1.05:
                performance_label = 'Exceeded'
            elif performance_index > 0.95:
                performance_label = 'Met'
            elif performance_index > 0.85:
                performance_label = 'Below'
            else:
                performance_label = 'Underperformed'
            
            results.append({
                'name': player_name,
                'firstName': player.get('firstName', ''),
                'lastName': player.get('lastName', ''),
                'gender': player.get('gender', 'M'),
                'rating': player_rating,
                'totalScore': total_score,
                'wins': wins,
                'losses': losses,
                'matchesPlayed': matches_played,
                'avgScore': round(avg_score, 1),
                'winRate': round(win_rate),
                'predictedAvg': round(predicted_avg, 1),
                'performanceIndex': round(performance_index, 3),
                'performanceLabel': performance_label
            })
        
        return results

    def calculate_predicted_score(self, team_rating, opponent_rating, sx=0.15):
        """Calculate predicted score using DiNKR algorithm"""
        rating_diff = abs(team_rating - opponent_rating)
        
        if team_rating > opponent_rating:
            # Team is favored
            raw_opponent_score = 11 - (rating_diff / sx)
            opponent_score = max(0, round(raw_opponent_score))
            return 11
        elif opponent_rating > team_rating:
            # Team is underdog
            raw_team_score = 11 - (rating_diff / sx)
            team_score = max(0, round(raw_team_score))
            return team_score
        else:
            # Even match
            return 10.5  # Average between 10 and 11

# Global tournament generator instance
tournament_gen = AdvancedTournamentGenerator()

@app.route('/api/test')
def health_check():
    return jsonify({"status": "healthy", "message": "DiNKR Tournament System is running"}), 200

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
        print(f"DEBUG: Options - avoid MM/FF: {avoid_mm_vs_ff}, rating balance: {use_rating_balance}, factor: {rating_factor}")
        
        # Validate player count
        min_players = courts * 4
        if total_players < min_players:
            return jsonify({"error": f"Need at least {min_players} players for {courts} courts"}), 400
        
        # Reset tournament generator for new tournament
        global tournament_gen
        tournament_gen = AdvancedTournamentGenerator()
        
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
        else:
            players_list = data.get('players', [])
            if len(players_list) != total_players:
                return jsonify({"error": f"Expected {total_players} players, got {len(players_list)}"}), 400
            
            # Build gender map for custom players
            for player in players_list:
                tournament_gen.gender_map[player['name']] = player['gender']
        
        result = tournament_gen.generate_complete_tournament(
            courts, players_list, rounds, avoid_mm_vs_ff, use_rating_balance, rating_factor
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Store tournament and initialize session
        session['tournament'] = result
        session['current_round'] = 0
        session['scores'] = {}
        session.modified = True
        
        print(f"DEBUG: Tournament generated with {len(result['schedule'])} rounds")
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
        
        if not tournament or 'schedule' not in tournament or 'players' not in tournament:
            return jsonify({"error": "Invalid tournament state"}), 400
        
        return jsonify({
            "tournament": tournament,
            "current_round": current_round,
            "scores": scores
        })
    except Exception as e:
        print(f"ERROR in get_tournament_state: {str(e)}")
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
            if 'name' in player and player['name']:
                player_map[player['name']] = player
            elif 'firstName' in player:
                full_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                if full_name:
                    player_map[full_name] = player
                    player_map[player['firstName']] = player
        
        # Apply each switch
        switches_applied = 0
        for switch in switches:
            old_player_name = switch['oldPlayer']
            new_player_name = switch['newPlayer']
            
            if new_player_name not in player_map:
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
                        break
                
                if switch_applied:
                    break
        
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
        
        # Update session
        session['tournament'] = tournament
        session.modified = True
        
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
        
        # Enhanced validation and error handling
        if not tournament or 'schedule' not in tournament or 'players' not in tournament:
            return jsonify({"error": "No valid tournament data found"}), 400
            
        total_rounds = len(tournament.get('schedule', []))
        next_round = current + 1
        
        if next_round >= total_rounds:
            return jsonify({"completed": True})
        
        # Preserve the global tournament_gen instance with existing state
        global tournament_gen
        
        # Validate that tournament_gen has proper state
        if not hasattr(tournament_gen, 'partnership_count') or not tournament_gen.partnership_count:
            print("WARNING: Tournament generator state lost, reconstructing...")
            # Reconstruct state from session data
            tournament_gen.initialize_tracking(tournament['players'])
        
        # Get current configuration
        players_list = tournament['players']
        courts = config.get('courts', 2)
        avoid_mm_vs_ff = config.get('avoidMMvsFF', True)
        use_rating_balance = config.get('useRatingBalance', True)
        rating_factor = config.get('ratingFactor', 3)
        
        # Filter out skipped players for this round
        available_players = [p for p in players_list if p.get('name') not in skip_players]
        
        print(f"DEBUG: Advancing to round {next_round + 1} with {len(available_players)} available players")
        
        matches, sit_outs_result = tournament_gen.generate_round_matches(
            available_players, courts, next_round, avoid_mm_vs_ff, use_rating_balance, rating_factor
        )
        
        if matches is None:
            print(f"ERROR: Failed to generate round {next_round + 1}: {sit_outs_result}")
            return jsonify({"error": sit_outs_result}), 400
        
        # Add manually skipped players to sit outs
        all_sit_outs = list(set(list(sit_outs_result) + skip_players))
        
        # Update the tournament schedule for the next round
        next_round_data = {
            "round": next_round + 1,
            "matches": matches,
            "sit_outs": all_sit_outs
        }
        
        tournament['schedule'][next_round] = next_round_data
        
        # Apply any manual player switches
        if player_switches:
            current_matches = tournament['schedule'][next_round]['matches']
            player_map = {p['name']: p for p in tournament['players']}
            
            for switch in player_switches:
                old_player_name = switch['oldPlayer']
                new_player_name = switch['newPlayer']
                
                if new_player_name not in player_map:
                    continue
                    
                new_player = player_map[new_player_name]
                
                for match in current_matches:
                    team_a, team_b = match
                    
                    for i, player in enumerate(team_a):
                        if player.get('name') == old_player_name:
                            team_a[i] = new_player
                            break
                    
                    for i, player in enumerate(team_b):
                        if player.get('name') == old_player_name:
                            team_b[i] = new_player
                            break
        
        # Update session with validation
        session['tournament'] = tournament
        session['current_round'] = next_round
        session.modified = True
        
        print(f"DEBUG: Successfully advanced to round {next_round + 1}")
        
        return jsonify({"success": True, "round": next_round})
        
    except Exception as e:
        print(f"ERROR in advance_round: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Round advancement failed: {str(e)}"}), 500

@app.route('/api/calculate_results', methods=['GET'])
def calculate_results():
    try:
        tournament = session.get('tournament', {})
        scores = session.get('scores', {})
        
        if not tournament or 'players' not in tournament:
            return jsonify({"error": "No tournament data found"}), 400
        
        # Use sophisticated DiNKR analysis
        results = tournament_gen.calculate_dinkr_performance_analysis(
            tournament['players'], scores, tournament['schedule']
        )
        
        return jsonify(results)
        
    except Exception as e:
        print(f"ERROR in calculate_results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

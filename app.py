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
        self.courts = 2  # Store courts for validation
        
    def create_default_players(self, total_players):
        """Create balanced default players with mixed genders and ratings"""
        first_names_male = ['Alex', 'Ben', 'Charlie', 'Dan', 'Ethan', 'Gavin', 'Henry', 'Ian', 'Jack', 'Kyle', 'Liam', 'Mason', 'Noah', 'Owen', 'Paul']
        first_names_female = ['Alice', 'Beth', 'Clara', 'Diana', 'Eve', 'Fiona', 'Grace', 'Holly', 'Iris', 'Julia', 'Kate', 'Luna', 'Maya', 'Nina', 'Olivia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Taylor', 'Wilson', 'Moore', 'Jackson', 'White']
        
        players = []
        used_names = set()
        
        # Calculate how many of each gender to create
        num_female = total_players // 2
        num_male = total_players - num_female
        
        # Create female players
        for i in range(num_female):
            rating = round(random.uniform(3.0, 4.5), 1)
            
            # Generate unique name
            attempts = 0
            while attempts < 1000:  # Increased attempts
                first_name = random.choice(first_names_female)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                if full_name not in used_names:
                    used_names.add(full_name)
                    break
                attempts += 1
            
            # Fallback with guaranteed uniqueness
            if attempts >= 1000:
                full_name = f"Player_F_{i+1}"
                first_name = f"Player_F_{i+1}"
                last_name = ""
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name.split()[-1] if last_name else "",
                'gender': 'F',
                'rating': rating
            })
            self.gender_map[full_name] = 'F'
        
        # Create male players
        for i in range(num_male):
            rating = round(random.uniform(3.0, 4.5), 1)
            
            # Generate unique name
            attempts = 0
            while attempts < 1000:
                first_name = random.choice(first_names_male)
                last_name = random.choice(last_names)
                full_name = f"{first_name} {last_name}"
                
                if full_name not in used_names:
                    used_names.add(full_name)
                    break
                attempts += 1
            
            # Fallback with guaranteed uniqueness
            if attempts >= 1000:
                full_name = f"Player_M_{i+1}"
                first_name = f"Player_M_{i+1}"
                last_name = ""
            
            players.append({
                'name': full_name,
                'firstName': first_name,
                'lastName': last_name.split()[-1] if last_name else "",
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
            
        for playerA in teamA:
            for playerB in teamB:
                if len(self.opponents[playerA]) > 0 and playerB in self.opponents[playerA]:
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

    def select_sit_outs(self, players, courts, round_index, manual_sit_outs=None):
        """Intelligently select players to sit out this round"""
        if manual_sit_outs is None:
            manual_sit_outs = []
            
        extra = len(players) - 4 * courts
        if extra <= 0:
            return manual_sit_outs

        player_names = [p['name'] for p in players]
        available_for_auto_sitout = [p for p in player_names if p not in manual_sit_outs]
        
        # Calculate how many more sit-outs we need
        additional_sitouts_needed = extra - len(manual_sit_outs)
        if additional_sitouts_needed <= 0:
            return manual_sit_outs
        
        # Sort available players by current sit-outs (ascending) + random tie-break
        sorted_players_by_sit = sorted(available_for_auto_sitout, key=lambda p: (self.sit_out_count[p], random.random()))
        candidate_combos = list(itertools.combinations(sorted_players_by_sit, additional_sitouts_needed))
        random.shuffle(candidate_combos)

        valid_combos = []
        for combo in candidate_combos:
            total_sitouts = list(manual_sit_outs) + list(combo)
            if self.sitting_constraints_ok(total_sitouts, round_index):
                valid_combos.append(combo)

        if not valid_combos:
            # No valid combos => fallback to first candidate
            auto_sitouts = list(candidate_combos[0]) if candidate_combos else []
            return list(manual_sit_outs) + auto_sitouts

        # Evaluate each valid combo by the new distribution's balance
        best_combo = None
        best_score = float('inf')

        for combo in valid_combos:
            # Simulate new distribution
            temp_sit_out_count = self.sit_out_count.copy()
            for p in list(manual_sit_outs) + list(combo):
                temp_sit_out_count[p] += 1

            # Score = difference between max and min sit-out counts
            difference = max(temp_sit_out_count.values()) - min(temp_sit_out_count.values())

            if difference < best_score:
                best_score = difference
                best_combo = combo

        auto_sitouts = list(best_combo) if best_combo else []
        return list(manual_sit_outs) + auto_sitouts

    def generate_round_matches(self, players, courts, round_index, avoid_mm_vs_ff=True, use_rating_balance=True, rating_factor=3, manual_sit_outs=None, only_mixed_doubles=False, player_swaps=None):
        """COMPLETELY REWRITTEN: Generate matches with bulletproof duplicate prevention"""
        
        if manual_sit_outs is None:
            manual_sit_outs = []
        if player_swaps is None:
            player_swaps = []
            
        print(f"DEBUG: Starting round {round_index + 1} generation with {len(players)} players, {courts} courts")
        print(f"DEBUG: Manual sit-outs: {manual_sit_outs}")
        
        # Store courts for validation
        self.courts = courts
        
        # Get players sitting out (including manual sit-outs)
        sit_outs = self.select_sit_outs(players, courts, round_index, manual_sit_outs)
        round_players = [p for p in players if p['name'] not in sit_outs]
        
        expected_playing = 4 * courts
        actual_playing = len(round_players)
        
        print(f"DEBUG: Expected playing: {expected_playing}, Actual playing: {actual_playing}")
        print(f"DEBUG: Final sitting out: {sit_outs}")
        print(f"DEBUG: Playing players: {[p['name'] for p in round_players]}")
        
        if actual_playing != expected_playing:
            return None, f"Invalid number of playing players: {actual_playing}, expected {expected_playing}"

        # Extract ratings for quick lookup
        ratings = {p['name']: p['rating'] for p in players}
        available_players = [p['name'] for p in round_players]
        
        # BULLETPROOF ALGORITHM: Sequential court assignment with strict validation
        matches = []
        used_players = set()
        
        def score_match_quality(teamA, teamB):
            """Score a potential match (lower is better)"""
            score = 0
            violations = []

            # Partnership constraints (heavily weighted)
            if not self.can_partner(teamA[0], teamA[1]):
                score += 10
                violations.append(f"repeated_partnership_{teamA[0]}_{teamA[1]}")
            if not self.can_partner(teamB[0], teamB[1]):
                score += 10
                violations.append(f"repeated_partnership_{teamB[0]}_{teamB[1]}")

            # NEW: Strict Mixed Doubles Logic
            if only_mixed_doubles:
                teamA_genders = {self.gender_map.get(p) for p in teamA}
                teamB_genders = {self.gender_map.get(p) for p in teamB}
                if teamA_genders != {'M', 'F'}:
                    score += 1000 # Heavy penalty to avoid non-mixed teams
                    violations.append(f"invalid_mixed_team_{teamA[0]}_{teamA[1]}")
                if teamB_genders != {'M', 'F'}:
                    score += 1000 # Heavy penalty to avoid non-mixed teams
                    violations.append(f"invalid_mixed_team_{teamB[0]}_{teamB[1]}")
            
            # Opponent constraints
            for a in teamA:
                for b in teamB:
                    if not self.can_face(a, b):
                        score += 2
            
            # Rating balance
            if use_rating_balance:
                rating_diff = self.rating_difference_penalty(teamA, teamB, ratings)
                score += rating_diff * rating_factor
            
            # Gender balance (avoid MM vs FF unless necessary)
            if avoid_mm_vs_ff and self.is_mm_vs_ff(teamA, teamB):
                rating_diff = self.rating_difference_penalty(teamA, teamB, ratings)
                if rating_diff >= 0.5:
                    score += 3
                else:
                    score += 1
            
            # Repeated matchup penalty
            score += self.times_matchup_played(teamA, teamB)
            
            # Consecutive opponent penalty
            if self.has_consecutive_opponents(teamA, teamB, round_index):
                score += 5
            
            return score, violations
        
        # Generate matches one court at a time
        for court_num in range(courts):
            print(f"DEBUG: Generating match for court {court_num + 1}")
            print(f"DEBUG: Available players: {available_players}")
            print(f"DEBUG: Used players so far: {used_players}")
            
            if len(available_players) < 4:
                return None, f"Not enough players remaining for court {court_num + 1}: {len(available_players)}"
            
            # Try all possible 4-player combinations from remaining players
            best_score = float('inf')
            best_match = None
            
            for combo in itertools.combinations(available_players, 4):
                # Ensure no player is already used
                if any(player in used_players for player in combo):
                    continue
                
                # Try all ways to split into 2 teams of 2
                for teamA in itertools.combinations(combo, 2):
                    teamB = tuple(p for p in combo if p not in teamA)
                    
                    # Score this match configuration
                    score, violations = score_match_quality(teamA, teamB)
                    
                    if score < best_score:
                        best_score = score
                        best_match = (combo, teamA, teamB, violations)
            
            if best_match is None:
                return None, f"Could not form valid match for court {court_num + 1}"
            
            combo, teamA, teamB, violations = best_match
            matches.append((teamA, teamB, {"violations": violations}))
            
            # CRITICAL: Remove all 4 players from available pool
            for player in combo:
                if player in available_players:
                    available_players.remove(player)
                used_players.add(player)
            
            print(f"DEBUG: Court {court_num + 1}: {teamA} vs {teamB}")
            print(f"DEBUG: Remaining players after court {court_num + 1}: {available_players}")
            print(f"DEBUG: Used players after court {court_num + 1}: {used_players}")
        
        # FINAL VALIDATION: Ensure no duplicate assignments
        all_assigned_players = []
        for teamA, teamB, _ in matches:
            all_assigned_players.extend(teamA)
            all_assigned_players.extend(teamB)
        
        print(f"DEBUG: All assigned players: {all_assigned_players}")
        print(f"DEBUG: Unique assigned players: {set(all_assigned_players)}")
        
        if len(all_assigned_players) != len(set(all_assigned_players)):
            duplicates = [p for p in all_assigned_players if all_assigned_players.count(p) > 1]
            return None, f"CRITICAL ERROR: Duplicate player assignments detected: {duplicates}"
        
        if len(all_assigned_players) != expected_playing:
            return None, f"Wrong number of players assigned: {len(all_assigned_players)} vs {expected_playing}"
        
        print(f"DEBUG: Successfully generated {len(matches)} matches with no duplicates")

        # Update tracking data
        if len(self.sit_out_history) <= round_index:
            self.sit_out_history.extend([[] for _ in range(round_index + 1 - len(self.sit_out_history))])
        
        self.sit_out_history[round_index] = list(sit_outs)
        for s in sit_outs:
            self.sit_out_count[s] += 1

        # Update partnership and opponent tracking
        for (teamA, teamB, _) in matches:
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

        # Convert player names back to player objects for return
        matches_with_objects = []
        player_lookup = {p['name']: p for p in players}
        
        for teamA_names, teamB_names, violations in matches:
            teamA_objects = [player_lookup[name] for name in teamA_names]
            teamB_objects = [player_lookup[name] for name in teamB_names]
            matches_with_objects.append([teamA_objects, teamB_objects, violations])

        sit_out_names = list(sit_outs)
        
        return matches_with_objects, sit_out_names

    def generate_complete_tournament(self, courts, players_list, rounds, avoid_mm_vs_ff=True, use_rating_balance=True, rating_factor=3, only_mixed_doubles=False):
        """Generate a complete tournament with sophisticated algorithms"""
        
        self.initialize_tracking(players_list)
        self.courts = courts
        
        schedule = []
        for round_num in range(rounds):
            print(f"DEBUG: Generating round {round_num + 1} of {rounds}")
            
            matches, sit_outs = self.generate_round_matches(
                players_list, courts, round_num, avoid_mm_vs_ff, use_rating_balance, rating_factor, only_mixed_doubles=only_mixed_doubles
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
                        
                    teamA, teamB, _ = match
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
                    
                    if player_team is not in [None, '']:
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
        only_mixed_doubles = data.get('onlyMixedDoubles', False)
        rating_factor = data.get('ratingFactor', 3)
        round_duration = data.get('roundDuration', 13)
        
        print(f"DEBUG: Tournament request - courts: {courts}, rounds: {rounds}")
        print(f"DEBUG: Options - avoid MM/FF: {avoid_mm_vs_ff}, rating balance: {use_rating_balance}, factor: {rating_factor}, only mixed: {only_mixed_doubles}")
        
        # Reset tournament generator for new tournament
        global tournament_gen
        tournament_gen = AdvancedTournamentGenerator()
        
        if use_defaults:
            total_players = data.get('totalPlayers', 8)
            min_players = courts * 4
            if total_players < min_players:
                return jsonify({"error": f"Need at least {min_players} players for {courts} courts"}), 400
            players_list = tournament_gen.create_default_players(total_players)
            print(f"DEBUG: Generated {len(players_list)} default players")
        else:
            players_list = data.get('players', [])
            total_players = len(players_list)
            min_players = courts * 4
            if total_players < min_players:
                return jsonify({"error": f"Not enough players for {courts} courts. Need {min_players}, got {total_players}."}), 400
            
            # Build gender map for custom players
            for player in players_list:
                tournament_gen.gender_map[player['name']] = player['gender']
        
        session['config'] = {
            'courts': courts,
            'rounds': rounds,
            'avoidMMvsFF': avoid_mm_vs_ff,
            'useRatingBalance': use_rating_balance,
            'onlyMixedDoubles': only_mixed_doubles,
            'ratingFactor': rating_factor,
            'roundDuration': round_duration,
            'totalPlayers': total_players
        }

        result = tournament_gen.generate_complete_tournament(
            courts, players_list, rounds, avoid_mm_vs_ff, use_rating_balance, rating_factor, only_mixed_doubles
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
            set_final_results()
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
        only_mixed_doubles = config.get('onlyMixedDoubles', False)
        rating_factor = config.get('ratingFactor', 3)
        
        # Filter out skipped players for this round
        available_players = [p for p in players_list if p.get('name') not in skip_players]
        
        print(f"DEBUG: Advancing to round {next_round + 1} with {len(available_players)} available players")
        print(f"DEBUG: Skip players: {skip_players}")
        
        # Apply player switches to the schedule for the *next* round
        # This is a conceptual change: we regenerate the next round with the switches applied
        # For simplicity, we'll let the user make changes and then we regenerate the *current* round view
        # before they hit "Start Round". The advance_round will just move the pointer.
        
        session['current_round'] = next_round
        session.modified = True
        
        print(f"DEBUG: Successfully advanced to round {next_round + 1}")
        
        return jsonify({"success": True, "round": next_round})
        
    except Exception as e:
        print(f"ERROR in advance_round: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Round advancement failed: {str(e)}"}), 500

def set_final_results():
    """Helper to calculate and store final results in session."""
    tournament = session.get('tournament', {})
    scores = session.get('scores', {})
    if 'players' not in tournament:
        return

    results = tournament_gen.calculate_dinkr_performance_analysis(
        tournament['players'], scores, tournament['schedule']
    )
    
    male_winners = sorted([p for p in results if p['gender'] == 'M'], key=lambda x: x['totalScore'], reverse=True)
    female_winners = sorted([p for p in results if p['gender'] == 'F'], key=lambda x: x['totalScore'], reverse=True)
    
    session['final_results'] = {
        "dinkrAnalysis": results,
        "pointWinners": {
            "male": male_winners,
            "female": female_winners
        }
    }
    session.modified = True

@app.route('/api/calculate_results', methods=['GET'])
def calculate_results():
    try:
        if 'final_results' in session:
            return jsonify(session['final_results'])

        set_final_results()
        
        if 'final_results' not in session:
            return jsonify({"error": "Could not calculate results"}), 400

        return jsonify(session['final_results'])
        
    except Exception as e:
        print(f"ERROR in calculate_results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Additional API endpoints for enhanced functionality

@app.route('/api/validate_tournament', methods=['POST'])
def validate_tournament():
    """Validate tournament configuration before generation"""
    try:
        data = request.json
        courts = data.get('courts', 2)
        total_players = data.get('totalPlayers', 8)
        rounds = data.get('rounds', 6)
        
        # Validation checks
        min_players = courts * 4
        max_reasonable_rounds = total_players * 2  # Rough estimate
        
        issues = []
        
        if total_players < min_players:
            issues.append(f"Need at least {min_players} players for {courts} courts")
        
        if rounds > max_reasonable_rounds:
            issues.append(f"Too many rounds ({rounds}) for {total_players} players. Consider {max_reasonable_rounds//2}-{max_reasonable_rounds} rounds.")
        
        if total_players > 50:
            issues.append("Large tournaments (50+ players) may take longer to generate")
            
        return jsonify({
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": {
                "optimal_rounds": min(rounds, total_players),
                "estimated_duration": f"{rounds * data.get('roundDuration', 13)} minutes"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_results', methods=['GET'])
def export_results():
    """Export tournament results as CSV"""
    try:
        results_data = session.get('final_results', {})
        
        if not results_data or 'dinkrAnalysis' not in results_data:
            return jsonify({"error": "No results data found"}), 400
        
        results = results_data['dinkrAnalysis']
        
        # Format as CSV
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            'Name', 'Gender', 'Rating', 'Total Score', 'Wins', 'Losses', 
            'Matches Played', 'Average Score', 'Win Rate %', 
            'Performance Index', 'Performance Label'
        ])
        
        # Data rows
        for player in sorted(results, key=lambda x: x['totalScore'], reverse=True):
            writer.writerow([
                player['name'], player['gender'], player['rating'],
                player['totalScore'], player['wins'], player['losses'],
                player['matchesPlayed'], player['avgScore'], player['winRate'],
                player['performanceIndex'], player['performanceLabel']
            ])
        
        output.seek(0)
        
        return jsonify({
            "csv_data": output.getvalue(),
            "filename": f"dinkr_tournament_results.csv"
        })
        
    except Exception as e:
        print(f"ERROR in export_results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tournament_stats', methods=['GET'])
def tournament_stats():
    """Get tournament statistics and health metrics"""
    try:
        tournament = session.get('tournament', {})
        current_round = session.get('current_round', 0)
        scores = session.get('scores', {})
        
        if not tournament:
            return jsonify({"error": "No tournament data"}), 400
        
        # Calculate statistics
        total_rounds = len(tournament.get('schedule', []))
        total_players = len(tournament.get('players', []))
        completed_rounds = current_round
        
        # Partnership distribution
        partnership_stats = {}
        for player in tournament['players']:
            player_name = player['name']
            partner_count = len(tournament_gen.partners.get(player_name, set()))
            partnership_stats[player_name] = partner_count
        
        # Sitting out distribution
        sitout_stats = {}
        for player in tournament['players']:
            player_name = player['name']
            sitout_count = tournament_gen.sit_out_count.get(player_name, 0)
            sitout_stats[player_name] = sitout_count
        
        return jsonify({
            "tournament_progress": {
                "current_round": current_round + 1,
                "total_rounds": total_rounds,
                "completion_percentage": round((completed_rounds / total_rounds) * 100, 1) if total_rounds > 0 else 0
            },
            "player_stats": {
                "total_players": total_players,
                "players_per_round": tournament_gen.courts * 4 if hasattr(tournament_gen, 'courts') else 0
            },
            "partnership_distribution": {
                "min_partners": min(partnership_stats.values()) if partnership_stats else 0,
                "max_partners": max(partnership_stats.values()) if partnership_stats else 0,
                "avg_partners": round(sum(partnership_stats.values()) / len(partnership_stats), 1) if partnership_stats else 0
            },
            "sitout_distribution": {
                "min_sitouts": min(sitout_stats.values()) if sitout_stats else 0,
                "max_sitouts": max(sitout_stats.values()) if sitout_stats else 0,
                "avg_sitouts": round(sum(sitout_stats.values()) / len(sitout_stats), 1) if sitout_stats else 0
            }
        })
        
    except Exception as e:
        print(f"ERROR in tournament_stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_round_assignments', methods=['POST'])
def update_round_assignments():
    """Regenerate current round with different parameters like skips or swaps."""
    try:
        data = request.json
        current = session.get('current_round', 0)
        tournament = session.get('tournament', {})
        config = session.get('config', {})
        
        if not tournament:
            return jsonify({"error": "No tournament data"}), 400
        
        # Get parameters
        skip_players = data.get('skipPlayers', [])
        player_swaps = data.get('playerSwitches', []) # [{oldPlayer, newPlayer}]
        
        # Get config
        courts = config.get('courts', 2)
        avoid_mm_vs_ff = config.get('avoidMMvsFF', True)
        use_rating_balance = config.get('useRatingBalance', True)
        only_mixed_doubles = config.get('onlyMixedDoubles', False)
        rating_factor = config.get('ratingFactor', 3)
        players_list = tournament['players']
        
        # This is complex logic. For now, we'll just regenerate the round with new sit-outs.
        # A full swap implementation would require more sophisticated logic.
        
        # For now, we just regenerate the round with the new skip list
        matches, sit_outs = tournament_gen.generate_round_matches(
            players_list, courts, current, avoid_mm_vs_ff, use_rating_balance, rating_factor,
            manual_sit_outs=skip_players, only_mixed_doubles=only_mixed_doubles
        )
        
        if matches is None:
            return jsonify({"error": sit_outs}), 400
        
        # Update tournament schedule for the current round
        tournament['schedule'][current] = {
            "round": current + 1,
            "matches": matches,
            "sit_outs": sit_outs
        }
        
        session['tournament'] = tournament
        session.modified = True
        
        return jsonify({
            "success": True,
            "new_schedule": tournament['schedule'][current],
            "message": f"Round {current + 1} assignments updated."
        })
        
    except Exception as e:
        print(f"ERROR in update_round_assignments: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

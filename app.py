# Additional API endpoints to add to your app.py

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
        tournament = session.get('tournament', {})
        scores = session.get('scores', {})
        
        if not tournament or 'players' not in tournament:
            return jsonify({"error": "No tournament data found"}), 400
        
        # Calculate results
        results = tournament_gen.calculate_dinkr_performance_analysis(
            tournament['players'], scores, tournament['schedule']
        )
        
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
            "filename": f"dinkr_tournament_results_{tournament.get('tournament_id', 'unknown')}.csv"
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
                "completion_percentage": round((completed_rounds / total_rounds) * 100, 1)
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

@app.route('/api/regenerate_round', methods=['POST'])
def regenerate_round():
    """Regenerate current round with different parameters"""
    try:
        data = request.json
        current = session.get('current_round', 0)
        tournament = session.get('tournament', {})
        config = session.get('config', {})
        
        if not tournament:
            return jsonify({"error": "No tournament data"}), 400
        
        # Get parameters
        skip_players = data.get('skipPlayers', [])
        avoid_mm_vs_ff = data.get('avoidMMvsFF', config.get('avoidMMvsFF', True))
        use_rating_balance = data.get('useRatingBalance', config.get('useRatingBalance', True))
        rating_factor = data.get('ratingFactor', config.get('ratingFactor', 3))
        
        # Regenerate current round
        players_list = tournament['players']
        courts = config.get('courts', 2)
        
        available_players = [p for p in players_list if p.get('name') not in skip_players]
        
        matches, sit_outs = tournament_gen.generate_round_matches(
            available_players, courts, current, avoid_mm_vs_ff, use_rating_balance, rating_factor,
            manual_sit_outs=skip_players
        )
        
        if matches is None:
            return jsonify({"error": sit_outs}), 400
        
        # Update tournament
        all_sit_outs = list(set(list(sit_outs) + skip_players))
        
        tournament['schedule'][current] = {
            "round": current + 1,
            "matches": matches,
            "sit_outs": all_sit_outs
        }
        
        session['tournament'] = tournament
        session.modified = True
        
        return jsonify({
            "success": True,
            "message": f"Round {current + 1} regenerated successfully"
        })
        
    except Exception as e:
        print(f"ERROR in regenerate_round: {str(e)}")
        return jsonify({"error": str(e)}), 500
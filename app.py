from flask import Flask, render_template, request, jsonify, session
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Debug logging
logger.info("Flask app starting...")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")

@app.route('/api/test')
def health_check():
    """Health check endpoint for Railway"""
    logger.info("Health check endpoint called")
    try:
        response = {
            "status": "healthy", 
            "message": "DiNKR Tournament System is running",
            "port": os.environ.get('PORT', 'not set'),
            "env": os.environ.get('RAILWAY_ENVIRONMENT', 'not set')
        }
        logger.info(f"Health check response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    """Main page - simplified for debugging"""
    logger.info("Index page requested")
    try:
        # Check if templates folder exists
        templates_exist = os.path.exists(os.path.join(app.root_path, 'templates'))
        index_exists = os.path.exists(os.path.join(app.root_path, 'templates', 'index.html'))
        
        logger.info(f"Templates folder exists: {templates_exist}")
        logger.info(f"Index.html exists: {index_exists}")
        
        if templates_exist and index_exists:
            return render_template('index.html')
        else:
            # Fallback HTML response
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>DiNKR Tournament</title>
                <style>body { font-family: Arial; padding: 20px; background: #325682; color: white; text-align: center; }</style>
            </head>
            <body>
                <h1>🏓 DiNKR Tournament System</h1>
                <p>Template files missing. App is running but templates need to be uploaded.</p>
                <p>Debug info:</p>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>Templates folder exists: ''' + str(templates_exist) + '''</li>
                    <li>Index.html exists: ''' + str(index_exists) + '''</li>
                    <li>App root: ''' + str(app.root_path) + '''</li>
                    <li>Working dir: ''' + str(os.getcwd()) + '''</li>
                </ul>
            </body>
            </html>
            '''
    except Exception as e:
        logger.error(f"Index route failed: {str(e)}")
        return f"Error loading page: {str(e)}", 500

@app.route('/debug')
def debug():
    """Debug endpoint"""
    logger.info("Debug endpoint called")
    return jsonify({
        "status": "running",
        "port": os.environ.get('PORT'),
        "railway_env": os.environ.get('RAILWAY_ENVIRONMENT'),
        "templates_exist": os.path.exists(os.path.join(app.root_path, 'templates')),
        "index_exists": os.path.exists(os.path.join(app.root_path, 'templates', 'index.html')),
        "working_dir": os.getcwd(),
        "app_root": app.root_path,
        "python_path": os.environ.get('PYTHONPATH')
    })

@app.route('/api/generate_tournament', methods=['POST'])
def generate_tournament():
    """Simplified tournament generation for testing"""
    logger.info("Tournament generation requested")
    try:
        data = request.get_json() or {}
        courts = data.get('courts', 2)
        
        response = {
            "success": True,
            "message": f"Tournament with {courts} courts would be generated here",
            "schedule": [],
            "players": []
        }
        
        logger.info("Tournament generation successful")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Tournament generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    
    # Start the app
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,
        threaded=True
    )

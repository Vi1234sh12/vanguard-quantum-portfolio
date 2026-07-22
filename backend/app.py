import sys
import os
import threading
import uuid

# Add the parent directory to sys.path so Flask can find the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models import db, OptimizationRun
from config import Config
from src.api_runner import run_optimization_pipeline  # Import the Quantum Runner

app = Flask(__name__)
app.config.from_object(Config)
CORS(app) # Allow Cross-Origin requests from our Frontend later

db.init_app(app)

# Initialize Database Tables
with app.app_context():
    db.create_all()

# In-memory task tracker for asynchronous UI updates
tasks = {}

# ==========================================
# CRUD API ENDPOINTS
# ==========================================

# 1. CREATE: Save a new optimization run manually
@app.route('/api/portfolios', methods=['POST'])
def create_portfolio():
    data = request.get_json()
    
    new_run = OptimizationRun(
        profile_name=data.get('profile_name', 'Custom'),
        status=data.get('status', 'PENDING'),
        config=data.get('config', {}),
        weights=data.get('weights', {}),
        metrics=data.get('metrics', {})
    )
    db.session.add(new_run)
    db.session.commit()
    
    return jsonify({"message": "Portfolio saved successfully", "data": new_run.to_dict()}), 201

# 2. READ: Get all saved optimization runs
@app.route('/api/portfolios', methods=['GET'])
def get_portfolios():
    runs = OptimizationRun.query.order_by(OptimizationRun.created_at.desc()).all()
    return jsonify({"data": [run.to_dict() for run in runs]}), 200

# 3. READ: Get a specific optimization run by ID
@app.route('/api/portfolios/<int:id>', methods=['GET'])
def get_portfolio(id):
    run = OptimizationRun.query.get_or_404(id)
    return jsonify({"data": run.to_dict()}), 200

# 4. UPDATE: Modify an existing run (e.g., rename profile or update status)
@app.route('/api/portfolios/<int:id>', methods=['PUT'])
def update_portfolio(id):
    run = OptimizationRun.query.get_or_404(id)
    data = request.get_json()
    
    if 'profile_name' in data:
        run.profile_name = data['profile_name']
    if 'status' in data:
        run.status = data['status']
    if 'weights' in data:
        run.weights = data['weights']
    if 'metrics' in data:
        run.metrics = data['metrics']
        
    db.session.commit()
    return jsonify({"message": "Portfolio updated successfully", "data": run.to_dict()}), 200

# 5. DELETE: Remove an optimization run
@app.route('/api/portfolios/<int:id>', methods=['DELETE'])
def delete_portfolio(id):
    run = OptimizationRun.query.get_or_404(id)
    db.session.delete(run)
    db.session.commit()
    return jsonify({"message": "Portfolio deleted successfully"}), 200

# ==========================================
# QUANTUM OPTIMIZATION ENDPOINT (ASYNCHRONOUS)
# ==========================================
@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    req_data = request.get_json()
    config = {
        "PROFILE_NAME": req_data.get('profile_name', 'API Custom'),
        "LAMBDA_RISK": float(req_data.get('lambda', 5.0)),
        "MAX_TURNOVER": float(req_data.get('max_turnover', 0.15)),
        "EQUITY_STRESS_CAP": float(req_data.get('equity_stress_cap', 0.40)),
        "MIN_RETURN": float(req_data.get('min_return', 0.07)),
        "P_return": 10.0, "bits": 6, "max_weight": 0.25,
        "MAX_SECTOR": 0.40, "P_budget": 25.0, "P_sector": 25.0, "P_turnover": 15.0
    }

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "PENDING", "progress": "Initializing...", "run_id": None}

    def background_task(app, task_id, config):
        with app.app_context():
            try:
                def update_status(msg):
                    tasks[task_id]["progress"] = msg
                
                print(f"⚙️ Starting background optimization for profile: {config['PROFILE_NAME']}...")
                opt_results = run_optimization_pipeline(config, status_callback=update_status)
                
                new_run = OptimizationRun(
                    profile_name=config['PROFILE_NAME'], status='COMPLETED',
                    config=opt_results['config'], weights=opt_results['weights'],
                    metrics=opt_results['metrics'], constraints=opt_results.get('constraints'),
                    ai_memo=opt_results.get('ai_memo'), frontier=opt_results.get('frontier'),
                    circuit_diagram=opt_results.get('circuit_diagram'),
                    expert_analysis=opt_results.get('expert_analysis'),
                    qubo_snippet=opt_results.get('qubo_snippet')
                )
                db.session.add(new_run)
                db.session.commit()
                
                tasks[task_id]["status"] = "COMPLETED"
                tasks[task_id]["run_id"] = new_run.id
                print(f"✅ Task {task_id} completed. Run ID: {new_run.id}")
            except Exception as e:
                print(f"❌ Optimization Error: {e}")
                tasks[task_id]["status"] = "FAILED"
                tasks[task_id]["progress"] = str(e)

    thread = threading.Thread(target=background_task, args=(app, task_id, config))
    thread.start()
    
    return jsonify({"message": "Optimization started", "task_id": task_id}), 202

# ==========================================
# TASK STATUS ENDPOINT (FOR PROGRESS BAR)
# ==========================================
@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task), 200

# ==========================================
# FRONTEND DASHBOARD ROUTES
# ==========================================

@app.route('/')
def dashboard():
    runs = OptimizationRun.query.order_by(OptimizationRun.created_at.desc()).all()
    return render_template('dashboard.html', runs=runs)

@app.route('/portfolio/<int:id>')
def portfolio_details(id):
    run = OptimizationRun.query.get_or_404(id)
    return render_template('portfolio_details.html', run=run)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class OptimizationRun(db.Model):
    __tablename__ = 'optimization_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    profile_name = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='COMPLETED')
    config = db.Column(db.JSON, nullable=True)
    weights = db.Column(db.JSON, nullable=True)
    metrics = db.Column(db.JSON, nullable=True)
    ai_memo = db.Column(db.Text, nullable=True)         
    constraints = db.Column(db.JSON, nullable=True)     
    frontier = db.Column(db.JSON, nullable=True)        
    circuit_diagram = db.Column(db.Text, nullable=True) 
    expert_analysis = db.Column(db.JSON, nullable=True) 
    qubo_snippet = db.Column(db.JSON, nullable=True)    # <-- NEW COLUMN
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'profile_name': self.profile_name,
            'status': self.status,
            'config': self.config,
            'weights': self.weights,
            'metrics': self.metrics,
            'ai_memo': self.ai_memo,
            'constraints': self.constraints,
            'frontier': self.frontier,
            'circuit_diagram': self.circuit_diagram,
            'expert_analysis': self.expert_analysis,
            'qubo_snippet': self.qubo_snippet,          # <-- ADDED HERE
            'created_at': self.created_at.isoformat()
        }
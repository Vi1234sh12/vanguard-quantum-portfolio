import os
import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    # PostgreSQL connection string
    SQLALCHEMY_DATABASE_URI = 'postgresql://quant_admin:quantum123@localhost:5432/quantum_portfolio'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY")
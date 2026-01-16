from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./logs.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_text = Column(String)
    prediction = Column(String)
    model_name = Column(String)
    inference_time_ms = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_prediction(input_text, prediction, model_name, inference_time):
    db = SessionLocal()
    try:
        log_entry = PredictionLog(
            input_text=input_text,
            prediction=prediction,
            model_name=model_name,
            inference_time_ms=inference_time
        )
        db.add(log_entry)
        db.commit()
    finally:
        db.close()
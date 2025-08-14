"""
FastAPI application for Betika Virtual Games Prediction API

Provides REST endpoints for predictions, model management, and analytics.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uvicorn

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..utils.config import Config
from ..utils.database import DatabaseManager
from ..data.collector import BetikaDataCollector
from ..models.predictor import GamePredictor


# Pydantic models for API
class GameData(BaseModel):
    """Game data input model."""
    game_type: str = Field(..., description="Type of virtual game")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_odds: float = Field(..., gt=0, description="Home team odds")
    draw_odds: Optional[float] = Field(None, gt=0, description="Draw odds")
    away_odds: float = Field(..., gt=0, description="Away team odds")
    league: Optional[str] = Field(None, description="League name")
    game_time: Optional[str] = Field(None, description="Game time")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: str = Field(..., description="Predicted outcome")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each outcome")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelPerformance(BaseModel):
    """Model performance response."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float


class TrainingRequest(BaseModel):
    """Model training request."""
    game_type: str = Field(..., description="Type of virtual game")
    retrain: bool = Field(False, description="Force retrain even if models exist")


# Global variables
config = None
predictor = None
collector = None
db_manager = None


def get_config():
    """Get application configuration."""
    global config
    if config is None:
        config = Config()
    return config


def get_predictor():
    """Get game predictor instance."""
    global predictor
    if predictor is None:
        predictor = GamePredictor(get_config())
    return predictor


def get_collector():
    """Get data collector instance."""
    global collector
    if collector is None:
        collector = BetikaDataCollector(get_config())
    return collector


def get_db_manager():
    """Get database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(get_config())
    return db_manager


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app_config = get_config()
    
    app = FastAPI(
        title="Betika Virtual Games Prediction API",
        description="AI-powered predictions for Betika virtual games",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.api['cors_origins'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Betika Virtual Games Prediction API")
    
    # Validate configuration
    config = get_config()
    if not config.validate():
        raise RuntimeError("Invalid configuration")
    
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger = logging.getLogger(__name__)
    logger.info("Shutting down API")
    
    # Clean up resources
    global collector, db_manager
    if collector:
        collector.close()
    if db_manager:
        db_manager.close()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Betika Virtual Games Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_manager = get_db_manager()
        recent_games = db_manager.get_recent_games("virtual_football", limit=1)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "last_data_update": recent_games.iloc[0]['timestamp'] if not recent_games.empty else None
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_game(
    game_data: GameData,
    model_name: str = Query("ensemble", description="Model to use for prediction")
):
    """Predict outcome for a single game."""
    try:
        predictor = get_predictor()
        
        # Convert Pydantic model to dict
        game_dict = game_data.dict()
        
        # Make prediction
        result = predictor.predict_game(game_dict, model_name)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error predicting game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{game_type}", response_model=List[Dict[str, Any]])
async def get_upcoming_predictions(
    game_type: str,
    model_name: str = Query("ensemble", description="Model to use for predictions"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of predictions")
):
    """Get predictions for upcoming games."""
    try:
        predictor = get_predictor()
        
        # Get predictions
        predictions = predictor.predict_upcoming_games(game_type, model_name)
        
        # Limit results
        limited_predictions = predictions[:limit]
        
        return limited_predictions
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=Dict[str, Any])
async def train_models(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train prediction models."""
    try:
        # Add training task to background
        background_tasks.add_task(
            _train_models_background,
            training_request.game_type,
            training_request.retrain
        )
        
        return {
            "message": f"Training started for {training_request.game_type}",
            "status": "training",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_models_background(game_type: str, retrain: bool):
    """Background task for model training."""
    try:
        predictor = get_predictor()
        model_scores = predictor.train_models(game_type, retrain)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Training completed for {game_type}: {model_scores}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in background training: {e}")


@app.get("/models/{model_name}/performance", response_model=Dict[str, Any])
async def get_model_performance(
    model_name: str,
    game_type: str = Query(..., description="Game type"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get model performance metrics."""
    try:
        predictor = get_predictor()
        
        performance = predictor.evaluate_model_performance(model_name, game_type, days)
        
        return performance
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/features", response_model=Dict[str, float])
async def get_feature_importance(
    model_name: str,
    game_type: str = Query(..., description="Game type")
):
    """Get feature importance for a model."""
    try:
        predictor = get_predictor()
        
        feature_importance = predictor.get_feature_importance(model_name, game_type)
        
        # Sort by importance
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_features
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/stats", response_model=Dict[str, Any])
async def get_data_statistics():
    """Get data collection statistics."""
    try:
        db_manager = get_db_manager()
        
        stats = {}
        game_types = ["virtual_football", "virtual_basketball", "virtual_tennis"]
        
        for game_type in game_types:
            recent_data = db_manager.get_recent_games(game_type, limit=1000)
            
            if not recent_data.empty:
                stats[game_type] = {
                    "total_games": len(recent_data),
                    "games_with_results": len(recent_data[recent_data['result'].notna()]),
                    "latest_game": recent_data.iloc[0]['timestamp'] if not recent_data.empty else None,
                    "oldest_game": recent_data.iloc[-1]['timestamp'] if not recent_data.empty else None
                }
            else:
                stats[game_type] = {
                    "total_games": 0,
                    "games_with_results": 0,
                    "latest_game": None,
                    "oldest_game": None
                }
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting data statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/collect", response_model=Dict[str, Any])
async def trigger_data_collection(background_tasks: BackgroundTasks):
    """Trigger manual data collection."""
    try:
        # Add collection task to background
        background_tasks.add_task(_collect_data_background)
        
        return {
            "message": "Data collection started",
            "status": "collecting",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _collect_data_background():
    """Background task for data collection."""
    try:
        collector = get_collector()
        data = await collector.collect_all_games()
        
        total_games = sum(len(games) for games in data.values())
        
        logger = logging.getLogger(__name__)
        logger.info(f"Data collection completed: {total_games} games collected")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in background data collection: {e}")


@app.get("/config", response_model=Dict[str, Any])
async def get_configuration():
    """Get current configuration (excluding sensitive data)."""
    try:
        config = get_config()
        
        # Return safe configuration data
        safe_config = {
            "data_collection": {
                "virtual_games": config.data_collection['virtual_games'],
                "collection_interval": config.data_collection['collection_interval']
            },
            "models": {
                "algorithms": config.models['algorithms'],
                "features": config.models['features']
            },
            "prediction": config.prediction,
            "api": {
                "host": config.api['host'],
                "port": config.api['port']
            }
        }
        
        return safe_config
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.api['host'],
        port=config.api['port'],
        reload=config.api['debug'],
        log_level="info"
    )
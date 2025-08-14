"""
Database Manager for Betika Virtual Games Data

Handles SQLite database operations for storing game data, odds, and predictions.
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from .config import Config


class DatabaseManager:
    """Manages SQLite database operations for virtual games data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.database['path']
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Games table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS games (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        game_type TEXT NOT NULL,
                        match_id TEXT,
                        home_team TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        league TEXT,
                        game_time TEXT,
                        home_odds REAL,
                        draw_odds REAL,
                        away_odds REAL,
                        result TEXT,
                        home_score INTEGER,
                        away_score INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Odds history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS odds_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        home_odds REAL,
                        draw_odds REAL,
                        away_odds REAL,
                        FOREIGN KEY (game_id) REFERENCES games (id)
                    )
                ''')
                
                # Predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id INTEGER,
                        model_name TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        features TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        actual_result TEXT,
                        correct BOOLEAN,
                        FOREIGN KEY (game_id) REFERENCES games (id)
                    )
                ''')
                
                # Model performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        date DATE NOT NULL,
                        total_predictions INTEGER DEFAULT 0,
                        correct_predictions INTEGER DEFAULT 0,
                        accuracy REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        profit_loss REAL DEFAULT 0.0
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_timestamp ON games (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_type ON games (game_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_name)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def save_games_data(self, games_data: List[Dict[str, Any]], game_type: str):
        """Save collected games data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for game in games_data:
                    # Check if game already exists
                    cursor.execute('''
                        SELECT id FROM games 
                        WHERE match_id = ? AND game_type = ?
                    ''', (game.get('match_id', ''), game_type))
                    
                    if cursor.fetchone():
                        continue  # Skip if already exists
                    
                    # Insert new game
                    teams = game.get('teams', {})
                    odds = game.get('odds', {})
                    
                    cursor.execute('''
                        INSERT INTO games (
                            timestamp, game_type, match_id, home_team, away_team,
                            league, game_time, home_odds, draw_odds, away_odds
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        game.get('timestamp', datetime.now()),
                        game_type,
                        game.get('match_id', ''),
                        teams.get('home', ''),
                        teams.get('away', ''),
                        game.get('league', ''),
                        game.get('game_time', ''),
                        odds.get('home_win', 0.0),
                        odds.get('draw', 0.0),
                        odds.get('away_win', 0.0)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving games data: {e}")
            raise
    
    def save_prediction(self, game_id: int, model_name: str, prediction: str, 
                       confidence: float, features: Dict[str, Any]):
        """Save model prediction to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO predictions (
                        game_id, model_name, prediction, confidence, features
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    game_id,
                    model_name,
                    prediction,
                    confidence,
                    json.dumps(features)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            raise
    
    def update_game_result(self, game_id: int, result: str, 
                          home_score: int = None, away_score: int = None):
        """Update game with actual result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE games 
                    SET result = ?, home_score = ?, away_score = ?
                    WHERE id = ?
                ''', (result, home_score, away_score, game_id))
                
                # Update prediction accuracy
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_result = ?, correct = (prediction = ?)
                    WHERE game_id = ?
                ''', (result, result, game_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating game result: {e}")
            raise
    
    def get_historical_data(self, game_type: str, days: int = 30) -> pd.DataFrame:
        """Retrieve historical games data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM games 
                    WHERE game_type = ? 
                    AND timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                
                return pd.read_sql_query(query, conn, params=(game_type,))
                
        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def get_recent_games(self, game_type: str, limit: int = 100) -> pd.DataFrame:
        """Get most recent games of specified type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM games 
                    WHERE game_type = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                
                return pd.read_sql_query(query, conn, params=(game_type, limit))
                
        except Exception as e:
            self.logger.error(f"Error retrieving recent games: {e}")
            return pd.DataFrame()
    
    def get_model_performance(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Get model performance statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                        AVG(confidence) as avg_confidence
                    FROM predictions 
                    WHERE model_name = ? 
                    AND timestamp >= datetime('now', '-{} days')
                '''.format(days), (model_name,))
                
                stats = cursor.fetchone()
                
                if stats and stats[0] > 0:
                    total, correct, avg_conf = stats
                    accuracy = correct / total if total > 0 else 0.0
                    
                    return {
                        'total_predictions': total,
                        'correct_predictions': correct,
                        'accuracy': accuracy,
                        'avg_confidence': avg_conf or 0.0
                    }
                else:
                    return {
                        'total_predictions': 0,
                        'correct_predictions': 0,
                        'accuracy': 0.0,
                        'avg_confidence': 0.0
                    }
                    
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}
    
    def get_predictions_for_evaluation(self, model_name: str = None) -> pd.DataFrame:
        """Get predictions that need evaluation (have actual results)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT p.*, g.result as actual_result
                    FROM predictions p
                    JOIN games g ON p.game_id = g.id
                    WHERE g.result IS NOT NULL
                '''
                
                params = []
                if model_name:
                    query += ' AND p.model_name = ?'
                    params.append(model_name)
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error getting predictions for evaluation: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days: int = 90):
        """Remove data older than specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove old games and related data
                cursor.execute('''
                    DELETE FROM games 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days))
                
                # Remove orphaned predictions
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE game_id NOT IN (SELECT id FROM games)
                ''')
                
                # Remove orphaned odds history
                cursor.execute('''
                    DELETE FROM odds_history 
                    WHERE game_id NOT IN (SELECT id FROM games)
                ''')
                
                conn.commit()
                self.logger.info(f"Cleaned up data older than {days} days")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def backup_database(self, backup_path: str):
        """Create database backup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    conn.backup(backup_conn)
            
            self.logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
    
    def close(self):
        """Close database connections."""
        # SQLite connections are automatically closed when context manager exits
        pass
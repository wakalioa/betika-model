"""
Feature Engineering for Betika Virtual Games Prediction

Creates meaningful features from raw game data for machine learning models.
Includes team statistics, historical performance, time patterns, and odds analysis.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler

from ..utils.config import Config


class FeatureEngineer:
    """Feature engineering for virtual games prediction."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features from games data."""
        try:
            if games_df.empty:
                return pd.DataFrame()
            
            # Copy dataframe to avoid modifying original
            df = games_df.copy()
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            features_list = []
            
            # Process each game
            for idx, game in df.iterrows():
                game_features = self._create_single_game_features_internal(game, df, idx)
                features_list.append(game_features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            self.logger.info(f"Created {len(features_df.columns)} features for {len(features_df)} games")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def create_single_game_features(self, game_data: Dict[str, Any]) -> pd.DataFrame:
        """Create features for a single game."""
        try:
            # Convert to DataFrame row format
            game_series = pd.Series(game_data)
            
            # Create basic features
            features = self._create_single_game_features_internal(game_series)
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating single game features: {e}")
            return pd.DataFrame()
    
    def _create_single_game_features_internal(self, game: pd.Series, 
                                             full_df: Optional[pd.DataFrame] = None, 
                                             current_idx: Optional[int] = None) -> Dict[str, float]:
        """Internal method to create features for a single game."""
        features = {}
        
        # Basic game features
        features.update(self._create_basic_features(game))
        
        # Odds features
        features.update(self._create_odds_features(game))
        
        # Time-based features
        features.update(self._create_time_features(game))
        
        # Team-based features (if historical data available)
        if full_df is not None and current_idx is not None:
            features.update(self._create_team_features(game, full_df, current_idx))
            features.update(self._create_historical_features(game, full_df, current_idx))
        
        return features
    
    def _create_basic_features(self, game: pd.Series) -> Dict[str, float]:
        """Create basic features from game data."""
        features = {}
        
        # Odds features
        home_odds = game.get('home_odds', 0.0)
        draw_odds = game.get('draw_odds', 0.0)
        away_odds = game.get('away_odds', 0.0)
        
        if home_odds > 0 and away_odds > 0:
            # Implied probabilities
            total_odds_inv = (1/home_odds if home_odds > 0 else 0) + \
                           (1/draw_odds if draw_odds > 0 else 0) + \
                           (1/away_odds if away_odds > 0 else 0)
            
            if total_odds_inv > 0:
                features['home_prob'] = (1/home_odds) / total_odds_inv if home_odds > 0 else 0
                features['draw_prob'] = (1/draw_odds) / total_odds_inv if draw_odds > 0 else 0
                features['away_prob'] = (1/away_odds) / total_odds_inv if away_odds > 0 else 0
            else:
                features['home_prob'] = 0.33
                features['draw_prob'] = 0.33
                features['away_prob'] = 0.33
            
            # Odds ratios
            features['home_away_odds_ratio'] = home_odds / away_odds if away_odds > 0 else 1.0
            features['total_odds_sum'] = home_odds + draw_odds + away_odds
            features['odds_variance'] = np.var([home_odds, draw_odds, away_odds])
            
            # Favorite indicator
            min_odds = min(home_odds, draw_odds, away_odds)
            features['is_home_favorite'] = 1.0 if home_odds == min_odds else 0.0
            features['is_draw_favorite'] = 1.0 if draw_odds == min_odds else 0.0
            features['is_away_favorite'] = 1.0 if away_odds == min_odds else 0.0
            
            # Odds confidence (inverse of variance)
            features['odds_confidence'] = 1.0 / (1.0 + features['odds_variance'])
        
        # Team name features (basic encoding)
        home_team = str(game.get('home_team', '')).lower()
        away_team = str(game.get('away_team', '')).lower()
        
        # Simple team strength indicators (based on common strong team names)
        strong_keywords = ['united', 'real', 'barcelona', 'arsenal', 'liverpool', 'milan', 'bayern']
        features['home_team_strength'] = sum(1 for keyword in strong_keywords if keyword in home_team)
        features['away_team_strength'] = sum(1 for keyword in strong_keywords if keyword in away_team)
        
        # League features
        league = str(game.get('league', '')).lower()
        features['is_premier_league'] = 1.0 if 'premier' in league else 0.0
        features['is_championship'] = 1.0 if 'championship' in league else 0.0
        features['is_serie_a'] = 1.0 if 'serie' in league else 0.0
        features['is_bundesliga'] = 1.0 if 'bundesliga' in league else 0.0
        
        return features
    
    def _create_odds_features(self, game: pd.Series) -> Dict[str, float]:
        """Create advanced odds-based features."""
        features = {}
        
        home_odds = game.get('home_odds', 0.0)
        draw_odds = game.get('draw_odds', 0.0)
        away_odds = game.get('away_odds', 0.0)
        
        if all(odds > 0 for odds in [home_odds, draw_odds, away_odds]):
            # Market efficiency indicators
            features['total_implied_prob'] = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            features['bookmaker_margin'] = features['total_implied_prob'] - 1.0
            
            # Odds entropy (measure of uncertainty)
            probs = [1/home_odds, 1/draw_odds, 1/away_odds]
            total_prob = sum(probs)
            normalized_probs = [p/total_prob for p in probs]
            features['odds_entropy'] = -sum(p * np.log2(p) for p in normalized_probs if p > 0)
            
            # Odds momentum (difference from expected even odds)
            even_odds = 3.0  # For 3-outcome game
            features['home_odds_momentum'] = (home_odds - even_odds) / even_odds
            features['away_odds_momentum'] = (away_odds - even_odds) / even_odds
            features['draw_odds_momentum'] = (draw_odds - even_odds) / even_odds
            
            # Value betting indicators
            fair_prob = 1/3  # Assuming equal probability for simplicity
            features['home_value'] = (fair_prob * home_odds) - 1 if home_odds > 0 else 0
            features['away_value'] = (fair_prob * away_odds) - 1 if away_odds > 0 else 0
            features['draw_value'] = (fair_prob * draw_odds) - 1 if draw_odds > 0 else 0
            
        return features
    
    def _create_time_features(self, game: pd.Series) -> Dict[str, float]:
        """Create time-based features."""
        features = {}
        
        # Extract timestamp
        timestamp = game.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)
            
            # Day of week (virtual games might have patterns)
            features['day_of_week'] = timestamp.dayofweek
            features['is_weekend'] = 1.0 if timestamp.dayofweek >= 5 else 0.0
            features['is_monday'] = 1.0 if timestamp.dayofweek == 0 else 0.0
            features['is_friday'] = 1.0 if timestamp.dayofweek == 4 else 0.0
            
            # Hour of day
            features['hour'] = timestamp.hour
            features['is_morning'] = 1.0 if 6 <= timestamp.hour < 12 else 0.0
            features['is_afternoon'] = 1.0 if 12 <= timestamp.hour < 18 else 0.0
            features['is_evening'] = 1.0 if 18 <= timestamp.hour < 24 else 0.0
            features['is_night'] = 1.0 if 0 <= timestamp.hour < 6 else 0.0
            
            # Month features (seasonal patterns)
            features['month'] = timestamp.month
            features['is_summer'] = 1.0 if timestamp.month in [6, 7, 8] else 0.0
            features['is_winter'] = 1.0 if timestamp.month in [12, 1, 2] else 0.0
            
            # Time since epoch (trend features)
            epoch_hours = (timestamp - pd.Timestamp('2020-01-01')).total_seconds() / 3600
            features['time_trend'] = epoch_hours / 1000  # Normalized
        
        # Game time features (if available)
        game_time = game.get('game_time', '')
        if game_time:
            # Extract patterns from game time string
            if ':' in str(game_time):
                try:
                    parts = str(game_time).split(':')
                    if len(parts) >= 2:
                        features['game_minute'] = int(parts[0])
                        features['game_second'] = int(parts[1])
                        features['total_game_seconds'] = features['game_minute'] * 60 + features['game_second']
                except:
                    pass
        
        return features
    
    def _create_team_features(self, game: pd.Series, full_df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
        """Create team-based historical features."""
        features = {}
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        # Look at historical data before current game
        historical_data = full_df.iloc[:current_idx] if current_idx > 0 else pd.DataFrame()
        
        if not historical_data.empty:
            # Home team statistics
            home_games = historical_data[
                (historical_data['home_team'] == home_team) | 
                (historical_data['away_team'] == home_team)
            ]
            
            if not home_games.empty:
                # Recent performance (last 5 games)
                recent_home = home_games.tail(5)
                features['home_recent_games'] = len(recent_home)
                
                # Win rate calculation
                home_wins = 0
                home_losses = 0
                for _, h_game in recent_home.iterrows():
                    result = h_game.get('result', '')
                    if h_game.get('home_team') == home_team:
                        if result == 'home_win':
                            home_wins += 1
                        elif result == 'away_win':
                            home_losses += 1
                    else:  # Away game for home team
                        if result == 'away_win':
                            home_wins += 1
                        elif result == 'home_win':
                            home_losses += 1
                
                features['home_win_rate'] = home_wins / len(recent_home) if len(recent_home) > 0 else 0.5
                features['home_loss_rate'] = home_losses / len(recent_home) if len(recent_home) > 0 else 0.5
            
            # Away team statistics
            away_games = historical_data[
                (historical_data['home_team'] == away_team) | 
                (historical_data['away_team'] == away_team)
            ]
            
            if not away_games.empty:
                recent_away = away_games.tail(5)
                features['away_recent_games'] = len(recent_away)
                
                away_wins = 0
                away_losses = 0
                for _, a_game in recent_away.iterrows():
                    result = a_game.get('result', '')
                    if a_game.get('home_team') == away_team:
                        if result == 'home_win':
                            away_wins += 1
                        elif result == 'away_win':
                            away_losses += 1
                    else:  # Away game for away team
                        if result == 'away_win':
                            away_wins += 1
                        elif result == 'home_win':
                            away_losses += 1
                
                features['away_win_rate'] = away_wins / len(recent_away) if len(recent_away) > 0 else 0.5
                features['away_loss_rate'] = away_losses / len(recent_away) if len(recent_away) > 0 else 0.5
            
            # Head-to-head record
            h2h_games = historical_data[
                ((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
                ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))
            ]
            
            if not h2h_games.empty:
                features['h2h_games_count'] = len(h2h_games)
                
                h2h_home_wins = 0
                for _, h2h_game in h2h_games.iterrows():
                    if h2h_game.get('home_team') == home_team and h2h_game.get('result') == 'home_win':
                        h2h_home_wins += 1
                    elif h2h_game.get('away_team') == home_team and h2h_game.get('result') == 'away_win':
                        h2h_home_wins += 1
                
                features['h2h_home_advantage'] = h2h_home_wins / len(h2h_games) if len(h2h_games) > 0 else 0.5
            else:
                features['h2h_games_count'] = 0
                features['h2h_home_advantage'] = 0.5
        
        return features
    
    def _create_historical_features(self, game: pd.Series, full_df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
        """Create features based on historical patterns."""
        features = {}
        
        # Look at recent games trends
        if current_idx >= 10:
            recent_games = full_df.iloc[current_idx-10:current_idx]
            
            # Recent results distribution
            if not recent_games.empty:
                results = recent_games['result'].value_counts(normalize=True)
                features['recent_home_win_pct'] = results.get('home_win', 0.33)
                features['recent_away_win_pct'] = results.get('away_win', 0.33)
                features['recent_draw_pct'] = results.get('draw', 0.33)
                
                # Recent odds trends
                features['recent_avg_home_odds'] = recent_games['home_odds'].mean()
                features['recent_avg_away_odds'] = recent_games['away_odds'].mean()
                features['recent_avg_draw_odds'] = recent_games['draw_odds'].mean()
                
                # Odds volatility
                features['recent_home_odds_std'] = recent_games['home_odds'].std()
                features['recent_away_odds_std'] = recent_games['away_odds'].std()
        
        # Same league historical patterns
        league = game.get('league', '')
        if league and current_idx > 0:
            league_games = full_df.iloc[:current_idx][full_df.iloc[:current_idx]['league'] == league]
            
            if not league_games.empty:
                league_results = league_games['result'].value_counts(normalize=True)
                features['league_home_win_pct'] = league_results.get('home_win', 0.33)
                features['league_away_win_pct'] = league_results.get('away_win', 0.33)
                features['league_draw_pct'] = league_results.get('draw', 0.33)
                
                # League average odds
                features['league_avg_home_odds'] = league_games['home_odds'].mean()
                features['league_avg_away_odds'] = league_games['away_odds'].mean()
                features['league_avg_draw_odds'] = league_games['draw_odds'].mean()
        
        return features
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if features_df.empty:
            return features_df
        
        # Fill missing values with appropriate defaults
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        # Fill with median for most features
        for col in numeric_columns:
            if features_df[col].isna().any():
                if 'rate' in col or 'pct' in col or 'prob' in col:
                    # Probability-like features default to neutral
                    features_df[col].fillna(0.5, inplace=True)
                elif 'odds' in col:
                    # Odds features default to even odds
                    features_df[col].fillna(2.0, inplace=True)
                elif 'count' in col:
                    # Count features default to 0
                    features_df[col].fillna(0, inplace=True)
                else:
                    # Other features use median or 0
                    median_val = features_df[col].median()
                    features_df[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
        
        return features_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # This would be called after creating features to get the column names
        # For now, return the common feature names
        basic_features = [
            'home_prob', 'draw_prob', 'away_prob',
            'home_away_odds_ratio', 'total_odds_sum', 'odds_variance',
            'is_home_favorite', 'is_draw_favorite', 'is_away_favorite',
            'odds_confidence', 'home_team_strength', 'away_team_strength'
        ]
        
        odds_features = [
            'total_implied_prob', 'bookmaker_margin', 'odds_entropy',
            'home_odds_momentum', 'away_odds_momentum', 'draw_odds_momentum',
            'home_value', 'away_value', 'draw_value'
        ]
        
        time_features = [
            'day_of_week', 'is_weekend', 'hour', 'is_morning', 'is_afternoon',
            'is_evening', 'month', 'is_summer', 'time_trend'
        ]
        
        team_features = [
            'home_recent_games', 'home_win_rate', 'home_loss_rate',
            'away_recent_games', 'away_win_rate', 'away_loss_rate',
            'h2h_games_count', 'h2h_home_advantage'
        ]
        
        historical_features = [
            'recent_home_win_pct', 'recent_away_win_pct', 'recent_draw_pct',
            'recent_avg_home_odds', 'recent_avg_away_odds', 'recent_avg_draw_odds',
            'league_home_win_pct', 'league_away_win_pct', 'league_draw_pct'
        ]
        
        return basic_features + odds_features + time_features + team_features + historical_features
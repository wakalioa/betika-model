"""
Betika Virtual Games Prediction Models

Implements multiple machine learning algorithms for predicting virtual game outcomes.
Includes ensemble methods, feature engineering, and model evaluation.
"""

import logging
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

from ..utils.config import Config
from ..utils.database import DatabaseManager
from .feature_engineering import FeatureEngineer


class GamePredictor:
    """Main prediction engine for virtual games."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_performance = {}
        
        # Setup algorithms
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models."""
        algorithms = self.config.models['algorithms']
        
        if 'random_forest' in algorithms:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        if 'xgboost' in algorithms:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        if 'lightgbm' in algorithms:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        if 'neural_network' in algorithms:
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
        
        if 'svm' in algorithms:
            self.models['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
        self.logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_training_data(self, game_type: str, days: int = 90) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and targets."""
        # Get historical data
        historical_data = self.db_manager.get_historical_data(game_type, days)
        
        if historical_data.empty:
            raise ValueError(f"No historical data found for {game_type}")
        
        # Filter games with results
        data_with_results = historical_data[historical_data['result'].notna()].copy()
        
        if len(data_with_results) < self.config.prediction['min_data_points']:
            raise ValueError(f"Insufficient data: {len(data_with_results)} games, need at least {self.config.prediction['min_data_points']}")
        
        # Engineer features
        features_df = self.feature_engineer.create_features(data_with_results)
        
        # Prepare target variable
        target = data_with_results['result'].copy()
        
        self.logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, target
    
    def train_models(self, game_type: str, retrain: bool = False) -> Dict[str, float]:
        """Train all configured models."""
        try:
            # Check if models are already trained and not retraining
            if not retrain and self._models_exist(game_type):
                self.logger.info(f"Models for {game_type} already trained. Use retrain=True to retrain.")
                return self._load_model_performance(game_type)
            
            # Prepare training data
            X, y = self.prepare_training_data(game_type)
            
            # Encode target labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders[game_type] = le
            
            # Split data
            training_config = self.config.models['training']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=training_config['test_size'],
                random_state=42,
                stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[game_type] = scaler
            
            model_scores = {}
            
            # Train each model
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name} for {game_type}...")
                
                try:
                    # Train model
                    if model_name in ['neural_network', 'svm']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate model
                    accuracy = accuracy_score(y_test, y_pred)
                    model_scores[model_name] = accuracy
                    
                    # Cross-validation
                    if model_name in ['neural_network', 'svm']:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    
                    self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                    # Save model
                    self._save_model(model, model_name, game_type)
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                    model_scores[model_name] = 0.0
            
            # Create ensemble model
            ensemble_score = self._create_ensemble_model(game_type, X_train, y_train, X_test, y_test)
            if ensemble_score:
                model_scores['ensemble'] = ensemble_score
            
            # Save performance metrics
            self._save_model_performance(game_type, model_scores)
            
            self.logger.info(f"Training completed for {game_type}. Best model: {max(model_scores, key=model_scores.get)} ({max(model_scores.values()):.4f})")
            
            return model_scores
            
        except Exception as e:
            self.logger.error(f"Error training models for {game_type}: {e}")
            raise
    
    def _create_ensemble_model(self, game_type: str, X_train: pd.DataFrame, y_train: np.ndarray, 
                              X_test: pd.DataFrame, y_test: np.ndarray) -> Optional[float]:
        """Create ensemble model from trained models."""
        try:
            estimators = []
            
            # Add available models to ensemble
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    estimators.append((model_name, model))
            
            if len(estimators) < 2:
                self.logger.warning("Not enough models for ensemble")
                return None
            
            # Create voting classifier
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            # Use scaled data for models that need it
            X_train_to_use = X_train
            X_test_to_use = X_test
            
            if any(name in ['neural_network', 'svm'] for name, _ in estimators):
                scaler = self.scalers[game_type]
                X_train_to_use = scaler.transform(X_train)
                X_test_to_use = scaler.transform(X_test)
            
            ensemble.fit(X_train_to_use, y_train)
            y_pred = ensemble.predict(X_test_to_use)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save ensemble model
            self._save_model(ensemble, 'ensemble', game_type)
            
            self.logger.info(f"Ensemble model accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble model: {e}")
            return None
    
    def predict_game(self, game_data: Dict[str, Any], model_name: str = 'ensemble') -> Dict[str, Any]:
        """Predict outcome for a single game."""
        try:
            game_type = game_data.get('game_type', 'virtual_football')
            
            # Load model
            model = self._load_model(model_name, game_type)
            if model is None:
                raise ValueError(f"Model {model_name} not found for {game_type}")
            
            # Create features
            features_df = self.feature_engineer.create_single_game_features(game_data)
            
            # Scale features if needed
            if model_name in ['neural_network', 'svm', 'ensemble']:
                scaler = self.scalers.get(game_type)
                if scaler:
                    features_scaled = scaler.transform(features_df)
                    features_to_use = features_scaled
                else:
                    features_to_use = features_df
            else:
                features_to_use = features_df
            
            # Make prediction
            prediction = model.predict(features_to_use)[0]
            prediction_proba = model.predict_proba(features_to_use)[0]
            
            # Decode prediction
            le = self.label_encoders.get(game_type)
            if le:
                prediction_label = le.inverse_transform([prediction])[0]
            else:
                prediction_label = str(prediction)
            
            # Get confidence (max probability)
            confidence = float(np.max(prediction_proba))
            
            # Get class probabilities
            class_probabilities = {}
            if le:
                for i, prob in enumerate(prediction_proba):
                    class_probabilities[le.classes_[i]] = float(prob)
            
            result = {
                'prediction': prediction_label,
                'confidence': confidence,
                'model_used': model_name,
                'class_probabilities': class_probabilities,
                'features_used': features_df.columns.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prediction for {game_type}: {prediction_label} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting game: {e}")
            raise
    
    def predict_upcoming_games(self, game_type: str, model_name: str = 'ensemble') -> List[Dict[str, Any]]:
        """Predict outcomes for upcoming games."""
        try:
            # Get upcoming games (games without results)
            recent_games = self.db_manager.get_recent_games(game_type, limit=50)
            upcoming_games = recent_games[recent_games['result'].isna()]
            
            if upcoming_games.empty:
                self.logger.info(f"No upcoming games found for {game_type}")
                return []
            
            predictions = []
            confidence_threshold = self.config.prediction['confidence_threshold']
            
            for _, game_row in upcoming_games.iterrows():
                try:
                    game_data = game_row.to_dict()
                    prediction_result = self.predict_game(game_data, model_name)
                    
                    # Only include high-confidence predictions
                    if prediction_result['confidence'] >= confidence_threshold:
                        prediction_result['game_id'] = game_row['id']
                        prediction_result['home_team'] = game_row['home_team']
                        prediction_result['away_team'] = game_row['away_team']
                        prediction_result['game_time'] = game_row['game_time']
                        prediction_result['league'] = game_row['league']
                        
                        predictions.append(prediction_result)
                        
                        # Save prediction to database
                        self.db_manager.save_prediction(
                            game_row['id'],
                            model_name,
                            prediction_result['prediction'],
                            prediction_result['confidence'],
                            {'features': prediction_result['features_used']}
                        )
                    
                except Exception as e:
                    self.logger.warning(f"Error predicting game {game_row.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Generated {len(predictions)} high-confidence predictions for {game_type}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting upcoming games: {e}")
            return []
    
    def evaluate_model_performance(self, model_name: str, game_type: str, days: int = 30) -> Dict[str, Any]:
        """Evaluate model performance on recent predictions."""
        try:
            performance_data = self.db_manager.get_model_performance(model_name, days)
            
            # Get detailed predictions for analysis
            predictions_df = self.db_manager.get_predictions_for_evaluation(model_name)
            
            if predictions_df.empty:
                return performance_data
            
            # Calculate additional metrics
            recent_predictions = predictions_df[
                predictions_df['timestamp'] >= (datetime.now() - timedelta(days=days))
            ]
            
            if not recent_predictions.empty:
                # Accuracy by confidence level
                confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                accuracy_by_confidence = {}
                
                for i in range(len(confidence_bins) - 1):
                    bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]
                    bin_data = recent_predictions[
                        (recent_predictions['confidence'] >= bin_min) & 
                        (recent_predictions['confidence'] < bin_max)
                    ]
                    
                    if not bin_data.empty:
                        accuracy = bin_data['correct'].mean()
                        accuracy_by_confidence[f"{bin_min}-{bin_max}"] = {
                            'accuracy': float(accuracy),
                            'count': len(bin_data)
                        }
                
                performance_data['accuracy_by_confidence'] = accuracy_by_confidence
                
                # Recent trend (last 7 days vs previous 7 days)
                last_7_days = recent_predictions[
                    recent_predictions['timestamp'] >= (datetime.now() - timedelta(days=7))
                ]
                prev_7_days = recent_predictions[
                    (recent_predictions['timestamp'] >= (datetime.now() - timedelta(days=14))) &
                    (recent_predictions['timestamp'] < (datetime.now() - timedelta(days=7)))
                ]
                
                performance_data['trend'] = {
                    'last_7_days_accuracy': float(last_7_days['correct'].mean()) if not last_7_days.empty else 0.0,
                    'prev_7_days_accuracy': float(prev_7_days['correct'].mean()) if not prev_7_days.empty else 0.0,
                    'last_7_days_count': len(last_7_days),
                    'prev_7_days_count': len(prev_7_days)
                }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def _save_model(self, model, model_name: str, game_type: str):
        """Save trained model to disk."""
        try:
            model_dir = f"models/{game_type}"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = f"{model_dir}/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler if exists
            if game_type in self.scalers:
                scaler_path = f"{model_dir}/scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[game_type], f)
            
            # Save label encoder if exists
            if game_type in self.label_encoders:
                le_path = f"{model_dir}/label_encoder.pkl"
                with open(le_path, 'wb') as f:
                    pickle.dump(self.label_encoders[game_type], f)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def _load_model(self, model_name: str, game_type: str):
        """Load trained model from disk."""
        try:
            model_path = f"models/{game_type}/{model_name}.pkl"
            if not os.path.exists(model_path):
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler if exists
            scaler_path = f"models/{game_type}/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[game_type] = pickle.load(f)
            
            # Load label encoder if exists
            le_path = f"models/{game_type}/label_encoder.pkl"
            if os.path.exists(le_path):
                with open(le_path, 'rb') as f:
                    self.label_encoders[game_type] = pickle.load(f)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def _models_exist(self, game_type: str) -> bool:
        """Check if models exist for game type."""
        model_dir = f"models/{game_type}"
        if not os.path.exists(model_dir):
            return False
        
        # Check if at least one model exists
        for model_name in self.models.keys():
            if os.path.exists(f"{model_dir}/{model_name}.pkl"):
                return True
        
        return False
    
    def _save_model_performance(self, game_type: str, model_scores: Dict[str, float]):
        """Save model performance metrics."""
        try:
            performance_path = f"models/{game_type}/performance.json"
            os.makedirs(os.path.dirname(performance_path), exist_ok=True)
            
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'scores': model_scores,
                'best_model': max(model_scores, key=model_scores.get),
                'best_score': max(model_scores.values())
            }
            
            with open(performance_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving model performance: {e}")
    
    def _load_model_performance(self, game_type: str) -> Dict[str, float]:
        """Load model performance metrics."""
        try:
            performance_path = f"models/{game_type}/performance.json"
            if not os.path.exists(performance_path):
                return {}
            
            with open(performance_path, 'r') as f:
                performance_data = json.load(f)
            
            return performance_data.get('scores', {})
            
        except Exception as e:
            self.logger.error(f"Error loading model performance: {e}")
            return {}
    
    def get_feature_importance(self, model_name: str, game_type: str) -> Dict[str, float]:
        """Get feature importance for tree-based models."""
        try:
            model = self._load_model(model_name, game_type)
            if model is None:
                return {}
            
            # Get feature importance for supported models
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                X, _ = self.prepare_training_data(game_type, days=30)
                feature_names = X.columns.tolist()
                importances = model.feature_importances_
                
                return dict(zip(feature_names, importances.tolist()))
            
            elif hasattr(model, 'coef_'):
                # For linear models
                X, _ = self.prepare_training_data(game_type, days=30)
                feature_names = X.columns.tolist()
                coefficients = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                
                return dict(zip(feature_names, coefficients.tolist()))
            
            else:
                self.logger.warning(f"Feature importance not available for {model_name}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
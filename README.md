# Betika Virtual Games Prediction Model

A comprehensive machine learning system for predicting virtual sports game outcomes on the Betika platform. This system includes data collection, feature engineering, multiple ML algorithms, and a REST API for serving predictions.

## Features

- **Multi-Game Support**: Virtual football, basketball, tennis, and horse racing
- **Multiple ML Algorithms**: Random Forest, XGBoost, LightGBM, Neural Networks, SVM, and Ensemble methods
- **Real-time Data Collection**: Automated scraping from Betika's virtual games platform
- **Feature Engineering**: Advanced statistical and temporal features for better predictions
- **REST API**: FastAPI-based service for easy integration
- **Performance Tracking**: Model evaluation and accuracy monitoring
- **CLI Interface**: Command-line tools for training, prediction, and management

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd betika-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Initialize the system:**
```bash
python main.py init
```

### Basic Usage

1. **Collect data:**
```bash
# One-time collection
python main.py collect --once

# Continuous collection
python main.py collect --continuous
```

2. **Train models:**
```bash
python main.py train virtual_football
```

3. **Generate predictions:**
```bash
python main.py predict virtual_football --limit 5
```

4. **Start API server:**
```bash
python main.py serve
```

## System Architecture

```
betika-model/
├── src/
│   ├── data/
│   │   └── collector.py          # Data collection from Betika
│   ├── models/
│   │   ├── predictor.py          # ML model ensemble
│   │   └── feature_engineering.py # Feature creation
│   ├── utils/
│   │   ├── config.py             # Configuration management
│   │   └── database.py           # SQLite database operations
│   └── api/
│       └── main.py               # FastAPI REST endpoints
├── config/
│   └── config.yaml               # Main configuration file
├── data/                         # SQLite database storage
├── models/                       # Trained model files
├── logs/                         # Application logs
└── main.py                       # CLI entry point
```

## Configuration

The system is configured via `config/config.yaml`:

```yaml
# Data Collection Settings
data_collection:
  base_url: "https://www.betika.com"
  virtual_games:
    - "virtual-football"
    - "virtual-basketball"
  collection_interval: 300  # seconds

# Model Settings
models:
  algorithms:
    - "random_forest"
    - "xgboost" 
    - "neural_network"
  confidence_threshold: 0.7

# API Settings
api:
  host: "0.0.0.0"
  port: 8000
```

## CLI Commands

### System Management
```bash
# Initialize system
python main.py init

# Check system status
python main.py status

# Clean up old data
python main.py cleanup --days 90
```

### Data Collection
```bash
# Collect data once
python main.py collect --once

# Start continuous collection
python main.py collect --continuous
```

### Model Training
```bash
# Train all models for virtual football
python main.py train virtual_football

# Force retrain existing models
python main.py train virtual_football --retrain

# Train specific algorithms only
python main.py train virtual_football -a random_forest -a xgboost
```

### Predictions
```bash
# Generate predictions
python main.py predict virtual_football --limit 10

# Use specific model
python main.py predict virtual_football --model xgboost

# Save predictions to file
python main.py predict virtual_football --save predictions.json
```

### Model Evaluation
```bash
# Evaluate model performance
python main.py evaluate ensemble virtual_football --days 30
```

### API Server
```bash
# Start API server
python main.py serve

# Custom host and port
python main.py serve --host 127.0.0.1 --port 8080

# Development mode with auto-reload
python main.py serve --reload
```

## API Endpoints

### Predictions
- `POST /predict` - Predict single game outcome
- `GET /predictions/{game_type}` - Get upcoming game predictions

### Model Management
- `POST /train` - Train models (background task)
- `GET /models/{model_name}/performance` - Model performance metrics
- `GET /models/{model_name}/features` - Feature importance

### Data & System
- `GET /health` - Health check
- `GET /data/stats` - Data collection statistics
- `POST /data/collect` - Trigger data collection
- `GET /config` - System configuration

### Example API Usage

**Predict a single game:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "game_type": "virtual_football",
    "home_team": "Arsenal",
    "away_team": "Chelsea", 
    "home_odds": 2.1,
    "draw_odds": 3.2,
    "away_odds": 3.8,
    "league": "Premier League"
  }'
```

**Get upcoming predictions:**
```bash
curl "http://localhost:8000/predictions/virtual_football?limit=5"
```

## Machine Learning Models

### Algorithms Used
1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting framework
3. **LightGBM** - Fast gradient boosting
4. **Neural Network** - Multi-layer perceptron
5. **SVM** - Support vector machine
6. **Ensemble** - Voting classifier combining all models

### Features Generated
- **Odds Analysis**: Implied probabilities, market efficiency, value betting indicators
- **Team Statistics**: Historical performance, win rates, head-to-head records
- **Temporal Patterns**: Time of day, day of week, seasonal trends
- **League Analysis**: League-specific patterns and statistics
- **Recent Form**: Short-term performance trends

### Model Training Process
1. **Data Preparation**: Historical games with known outcomes
2. **Feature Engineering**: Create meaningful features from raw data
3. **Model Training**: Train multiple algorithms with cross-validation
4. **Ensemble Creation**: Combine models using voting classifier
5. **Performance Evaluation**: Track accuracy and confidence metrics

## Performance Monitoring

The system tracks model performance through:
- **Accuracy Metrics**: Overall and confidence-level accuracy
- **Prediction Tracking**: All predictions stored with outcomes
- **Trend Analysis**: Recent performance trends
- **Feature Importance**: Understanding key predictive factors

## Development

### Adding New Game Types
1. Update `config.yaml` to include new game type
2. Extend data collector for new game type parsing
3. Train models for the new game type

### Adding New Algorithms
1. Import algorithm in `src/models/predictor.py`
2. Add to `_initialize_models()` method
3. Update configuration with new algorithm name

### Custom Features
1. Extend `FeatureEngineer` class in `src/models/feature_engineering.py`
2. Add new feature creation methods
3. Update feature handling in predictor

## Dependencies

- **Python 3.8+**
- **Machine Learning**: scikit-learn, xgboost, lightgbm, tensorflow
- **Data Processing**: pandas, numpy
- **Web Framework**: FastAPI, uvicorn
- **Data Collection**: requests, beautifulsoup4, selenium
- **Database**: SQLite (via sqlite3)
- **Configuration**: PyYAML, python-dotenv

## Environment Variables

Override configuration with environment variables:
- `BETIKA_BASE_URL` - Base URL for data collection
- `BETIKA_COLLECTION_INTERVAL` - Collection interval in seconds
- `BETIKA_DB_PATH` - Database file path
- `BETIKA_API_HOST` - API server host
- `BETIKA_API_PORT` - API server port
- `BETIKA_LOG_LEVEL` - Logging level

## Troubleshooting

### Common Issues

**"No data available for training"**
- Run data collection first: `python main.py collect --once`
- Ensure sufficient historical data (100+ games with results)

**"Configuration file not found"**
- Ensure `config/config.yaml` exists
- Use `--config` option to specify custom path

**"ChromeDriver not found"**
- Install ChromeDriver for Selenium
- Ensure it's in PATH or use headless mode

**API server fails to start**
- Check if port is already in use
- Verify configuration file is valid

### Logs
Check application logs in the `logs/` directory for detailed error information.

## License

This project is for educational and research purposes. Ensure compliance with Betika's terms of service when collecting data.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Open an issue with detailed error information

---

**Note**: This system is designed for educational purposes. Always comply with the website's terms of service and robots.txt when collecting data.

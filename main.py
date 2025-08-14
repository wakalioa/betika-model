#!/usr/bin/env python3
"""
Betika Virtual Games Prediction Model - Main Entry Point

Command-line interface for managing data collection, model training, predictions, and API server.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import click
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.database import DatabaseManager
from src.data.collector import BetikaDataCollector
from src.models.predictor import GamePredictor


@click.group()
@click.option('--config', '-c', default=None, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Betika Virtual Games Prediction Model CLI."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        ctx.obj['config'] = Config(config)
        click.echo(f"✅ Configuration loaded successfully")
    except Exception as e:
        click.echo(f"❌ Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the prediction model system."""
    click.echo("🚀 Initializing Betika Virtual Games Prediction Model...")
    
    config = ctx.obj['config']
    
    try:
        # Initialize database
        db_manager = DatabaseManager(config)
        click.echo("✅ Database initialized")
        
        # Validate configuration
        if config.validate():
            click.echo("✅ Configuration validated")
        else:
            click.echo("❌ Configuration validation failed", err=True)
            return
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        click.echo("✅ System initialized successfully!")
        click.echo("\nNext steps:")
        click.echo("1. Run 'python main.py collect' to start data collection")
        click.echo("2. Run 'python main.py train virtual_football' to train models")
        click.echo("3. Run 'python main.py serve' to start the API server")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--continuous', '-c', is_flag=True, help='Run continuous data collection')
@click.option('--once', is_flag=True, help='Run data collection once')
@click.pass_context
def collect(ctx, continuous, once):
    """Collect data from Betika virtual games."""
    click.echo("📊 Starting data collection...")
    
    config = ctx.obj['config']
    
    async def run_collection():
        collector = BetikaDataCollector(config)
        
        try:
            if continuous:
                click.echo("🔄 Starting continuous data collection...")
                click.echo("Press Ctrl+C to stop")
                await collector.start_continuous_collection()
            else:
                click.echo("📥 Collecting data once...")
                data = await collector.collect_all_games()
                
                total_games = sum(len(games) for games in data.values())
                click.echo(f"✅ Collection completed: {total_games} games collected")
                
                # Show breakdown by game type
                for game_type, games in data.items():
                    if games:
                        click.echo(f"  {game_type}: {len(games)} games")
        
        except KeyboardInterrupt:
            click.echo("\n⏹️  Data collection stopped by user")
        except Exception as e:
            click.echo(f"❌ Data collection failed: {e}", err=True)
        finally:
            collector.close()
    
    # Default to once if no option specified
    if not continuous and not once:
        once = True
    
    asyncio.run(run_collection())


@cli.command()
@click.argument('game_type')
@click.option('--retrain', is_flag=True, help='Force retrain even if models exist')
@click.option('--algorithm', '-a', multiple=True, help='Specific algorithms to train')
@click.pass_context
def train(ctx, game_type, retrain, algorithm):
    """Train prediction models for specified game type."""
    click.echo(f"🧠 Training models for {game_type}...")
    
    config = ctx.obj['config']
    
    try:
        predictor = GamePredictor(config)
        
        # Filter algorithms if specified
        if algorithm:
            original_algorithms = config.models['algorithms']
            config.models['algorithms'] = [alg for alg in algorithm if alg in original_algorithms]
            click.echo(f"Training specific algorithms: {config.models['algorithms']}")
        
        # Train models
        model_scores = predictor.train_models(game_type, retrain)
        
        if model_scores:
            click.echo("✅ Training completed!")
            click.echo("\nModel Performance:")
            for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
                click.echo(f"  {model_name}: {score:.4f}")
            
            best_model = max(model_scores, key=model_scores.get)
            click.echo(f"\n🏆 Best model: {best_model} ({model_scores[best_model]:.4f})")
        else:
            click.echo("❌ Training failed - no models trained", err=True)
    
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('game_type')
@click.option('--model', '-m', default='ensemble', help='Model to use for predictions')
@click.option('--limit', '-l', default=10, help='Maximum number of predictions')
@click.option('--save', '-s', help='Save predictions to file')
@click.pass_context
def predict(ctx, game_type, model, limit, save):
    """Generate predictions for upcoming games."""
    click.echo(f"🔮 Generating predictions for {game_type}...")
    
    config = ctx.obj['config']
    
    try:
        predictor = GamePredictor(config)
        
        # Get predictions
        predictions = predictor.predict_upcoming_games(game_type, model)
        
        if not predictions:
            click.echo(f"ℹ️  No upcoming games found for {game_type}")
            return
        
        # Limit results
        limited_predictions = predictions[:limit]
        
        click.echo(f"✅ Generated {len(limited_predictions)} predictions:")
        click.echo()
        
        for i, pred in enumerate(limited_predictions, 1):
            click.echo(f"{i}. {pred['home_team']} vs {pred['away_team']}")
            click.echo(f"   Prediction: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
            click.echo(f"   League: {pred.get('league', 'Unknown')}")
            click.echo(f"   Time: {pred.get('game_time', 'Unknown')}")
            click.echo()
        
        # Save to file if requested
        if save:
            with open(save, 'w') as f:
                json.dump(limited_predictions, f, indent=2, default=str)
            click.echo(f"💾 Predictions saved to {save}")
    
    except Exception as e:
        click.echo(f"❌ Prediction failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default=None, help='Host to bind the server to')
@click.option('--port', default=None, type=int, help='Port to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the API server."""
    click.echo("🌐 Starting API server...")
    
    config = ctx.obj['config']
    
    # Override config with command line options
    server_host = host or config.api['host']
    server_port = port or config.api['port']
    
    try:
        import uvicorn
        from src.api.main import app
        
        click.echo(f"🚀 Server starting on http://{server_host}:{server_port}")
        click.echo(f"📖 API docs available at http://{server_host}:{server_port}/docs")
        click.echo("Press Ctrl+C to stop")
        
        uvicorn.run(
            "src.api.main:app",
            host=server_host,
            port=server_port,
            reload=reload,
            log_level="info"
        )
    
    except KeyboardInterrupt:
        click.echo("\n⏹️  Server stopped by user")
    except Exception as e:
        click.echo(f"❌ Server failed to start: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_name')
@click.argument('game_type')
@click.option('--days', default=30, help='Number of days to analyze')
@click.pass_context
def evaluate(ctx, model_name, game_type, days):
    """Evaluate model performance."""
    click.echo(f"📈 Evaluating {model_name} model for {game_type}...")
    
    config = ctx.obj['config']
    
    try:
        predictor = GamePredictor(config)
        
        performance = predictor.evaluate_model_performance(model_name, game_type, days)
        
        if performance:
            click.echo("✅ Performance Analysis:")
            click.echo(f"  Total Predictions: {performance.get('total_predictions', 0)}")
            click.echo(f"  Correct Predictions: {performance.get('correct_predictions', 0)}")
            click.echo(f"  Accuracy: {performance.get('accuracy', 0):.4f}")
            click.echo(f"  Average Confidence: {performance.get('avg_confidence', 0):.4f}")
            
            # Show accuracy by confidence if available
            if 'accuracy_by_confidence' in performance:
                click.echo("\nAccuracy by Confidence Level:")
                for conf_range, stats in performance['accuracy_by_confidence'].items():
                    click.echo(f"  {conf_range}: {stats['accuracy']:.4f} ({stats['count']} predictions)")
            
            # Show trend if available
            if 'trend' in performance:
                trend = performance['trend']
                click.echo(f"\nRecent Trend:")
                click.echo(f"  Last 7 days: {trend['last_7_days_accuracy']:.4f} ({trend['last_7_days_count']} predictions)")
                click.echo(f"  Previous 7 days: {trend['prev_7_days_accuracy']:.4f} ({trend['prev_7_days_count']} predictions)")
        else:
            click.echo("ℹ️  No performance data available")
    
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics."""
    click.echo("📊 System Status")
    click.echo("=" * 50)
    
    config = ctx.obj['config']
    
    try:
        db_manager = DatabaseManager(config)
        
        # Show data statistics
        game_types = ['virtual_football', 'virtual_basketball', 'virtual_tennis']
        
        for game_type in game_types:
            click.echo(f"\n{game_type.replace('_', ' ').title()}:")
            
            recent_games = db_manager.get_recent_games(game_type, limit=1000)
            
            if not recent_games.empty:
                total_games = len(recent_games)
                games_with_results = len(recent_games[recent_games['result'].notna()])
                latest_game = recent_games.iloc[0]['timestamp']
                
                click.echo(f"  Total games: {total_games}")
                click.echo(f"  Games with results: {games_with_results}")
                click.echo(f"  Latest game: {latest_game}")
                
                # Check if models exist
                model_dir = Path(f"models/{game_type}")
                if model_dir.exists():
                    model_files = list(model_dir.glob("*.pkl"))
                    click.echo(f"  Trained models: {len(model_files)}")
                else:
                    click.echo(f"  Trained models: 0")
            else:
                click.echo(f"  No data available")
        
        # Show configuration summary
        click.echo(f"\nConfiguration:")
        click.echo(f"  Collection interval: {config.data_collection['collection_interval']}s")
        click.echo(f"  Algorithms: {', '.join(config.models['algorithms'])}")
        click.echo(f"  Confidence threshold: {config.prediction['confidence_threshold']}")
        click.echo(f"  API port: {config.api['port']}")
    
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}", err=True)


@cli.command()
@click.option('--days', default=90, help='Days of data to keep')
@click.pass_context
def cleanup(ctx, days):
    """Clean up old data and logs."""
    click.echo(f"🧹 Cleaning up data older than {days} days...")
    
    config = ctx.obj['config']
    
    try:
        db_manager = DatabaseManager(config)
        db_manager.cleanup_old_data(days)
        
        click.echo("✅ Database cleanup completed")
        
        # TODO: Add log file cleanup
        
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}", err=True)


if __name__ == '__main__':
    cli()
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
import sys
from stock_analysis.ai.model_manager import ModelManager
from stock_analysis.ai.technical_analysis.movement_classification import train_classifier_single_stock, train_classifier_tickers
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from stock_analysis.ai.helpers import get_index_data

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app and ModelManager
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model_manager = ModelManager()

# Configuration


class Config:
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('FLASK_PORT', 5000))
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')

# Root endpoint


@app.route('/', methods=['GET'])
def index():
    logger.info('Accessing root endpoint')
    try:
        # Get the API status
        api_status = "Operational"
        logger.info('Rendering index template')
        return render_template('index.html', status=api_status)
    except Exception as e:
        logger.error(f'Error in root endpoint: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Health check endpoint


@app.route('/health', methods=['GET'])
def health_check():
    logger.info('Health check requested')
    return jsonify({
        'status': 'healthy',
        'message': 'Stock Analysis API is running'
    })

# Error handlers


@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f'404 error: {request.url}')
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


# API version prefix
API_PREFIX = '/api/v1'

# Example of a versioned endpoint


@app.route(f'{API_PREFIX}/status', methods=['GET'])
def get_status():
    logger.info('Status endpoint accessed')
    return jsonify({
        'status': 'operational',
        'version': '1.0.0',
        'endpoints_available': [
            '/health',
            f'{API_PREFIX}/status'
        ]
    })

# Model endpoints


@app.route(f'{API_PREFIX}/models', methods=['GET'])
def list_models():
    """List all available models"""
    models = model_manager.list_models()
    return jsonify({
        'models': models
    })


@app.route(f'{API_PREFIX}/predict/<model_name>/<model_type>', methods=['POST'])
def predict(model_name, model_type):
    """Make predictions using a specific model"""
    try:
        # Get cache bypass flag from request
        bypass_cache = request.args.get('bypass_cache', '').lower() == 'true'

        # Load the model
        model = model_manager.load_model(
            model_name, model_type, bypass_cache=bypass_cache)
        if model is None:
            return jsonify({
                'error': 'Model not found'
            }), 404

        # Get input data from request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'No features provided'
            }), 400

        # Make prediction
        prediction = model.predict([data['features']])[0]

        # Get model metadata
        metadata = model_manager.get_model_metadata(model_name, model_type)

        return jsonify({
            'prediction': prediction,
            'model_info': {
                'name': model_name,
                'type': model_type,
                'metadata': metadata,
                'used_cache': not bypass_cache
            }
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route(f'{API_PREFIX}/models/cache', methods=['DELETE'])
def clear_model_cache():
    """Clear the model cache"""
    try:
        model_manager.clear_cache()
        return jsonify({
            'message': 'Model cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'error': 'Failed to clear cache',
            'message': str(e)
        }), 500

# Movement Classification endpoints


@app.route(f'{API_PREFIX}/train/<ticker>', methods=['POST'])
def train_movement_classifier(ticker):
    """Train movement classifier for a specific ticker"""
    try:
        # Get parameters from request
        params = request.get_json()
        logger.info(f"Training model for {ticker} with parameters: {params}")

        threshold = float(params.get('threshold', 0.005))
        use_extra_features = params.get('use_extra_features', False)
        period = params.get('period', '5y')

        # Get stock data
        logger.info(f"Fetching data for {ticker} over period {period}")
        data = get_index_data(ticker, period)
        if data is None or len(data) == 0:
            logger.error(f"No data received for {ticker}")
            return jsonify({
                'error': 'Data fetch failed',
                'message': f'Could not fetch data for {ticker}'
            }), 500

        logger.info(f"Received {len(data)} data points for {ticker}")

        # Prepare data for training
        logger.info("Preparing data for training")
        from stock_analysis.ai.technical_analysis.movement_classification import prepare_classification_data
        prepared_data = prepare_classification_data(
            data,
            predict_weekly=False,
            threshold=threshold,
            use_extra_features=use_extra_features
        )
        prepared_weekly_data = prepare_classification_data(
            data,
            predict_weekly=True,
            threshold=threshold,
            use_extra_features=use_extra_features
        )

        # Train models
        logger.info("Initializing classifiers")
        daily_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)
        weekly_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42)

        # Train daily model
        logger.info("Training daily model")
        try:
            daily_model, daily_scaler, daily_data = train_classifier_single_stock(
                prepared_data,
                predict_weekly=False,
                threshold=threshold,
                use_extra_features=use_extra_features,
                classifier=daily_classifier,
                plot=False
            )
        except Exception as e:
            logger.error(f"Error training daily model: {str(e)}")
            raise

        # Train weekly model
        logger.info("Training weekly model")
        try:
            weekly_model, weekly_scaler, weekly_data = train_classifier_single_stock(
                prepared_weekly_data,
                predict_weekly=True,
                threshold=threshold,
                use_extra_features=use_extra_features,
                classifier=weekly_classifier,
                plot=False
            )
        except Exception as e:
            logger.error(f"Error training weekly model: {str(e)}")
            raise

        # Save models with metadata
        logger.info("Saving models")
        model_name = f"{ticker}_movement"
        metadata = {
            'ticker': ticker,
            'threshold': threshold,
            'use_extra_features': use_extra_features,
            'period': period,
            'training_date': pd.Timestamp.now().isoformat()
        }

        try:
            # Save daily model
            model_manager.save_model(
                model=daily_model,
                model_name=model_name,
                model_type='daily',
                metadata=metadata
            )

            # Save daily scaler
            model_manager.save_model(
                model=daily_scaler,
                model_name=f"{model_name}_scaler",
                model_type='daily',
                metadata=metadata
            )

            # Save weekly model
            model_manager.save_model(
                model=weekly_model,
                model_name=model_name,
                model_type='weekly',
                metadata=metadata
            )

            # Save weekly scaler
            model_manager.save_model(
                model=weekly_scaler,
                model_name=f"{model_name}_scaler",
                model_type='weekly',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

        logger.info(f"Successfully trained and saved models for {ticker}")
        return jsonify({
            'message': f'Successfully trained models for {ticker}',
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error training models for {ticker}: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500


@app.route(f'{API_PREFIX}/predict/movement/<ticker>', methods=['GET'])
def predict_movement(ticker):
    """Get movement predictions for a ticker"""
    try:
        # Get parameters
        bypass_cache = request.args.get('bypass_cache', '').lower() == 'true'
        days = int(request.args.get('days', 5))

        # Load models and scalers
        model_name = f"{ticker}_movement"

        daily_model = model_manager.load_model(
            model_name, 'daily', bypass_cache=bypass_cache)
        daily_scaler = model_manager.load_model(
            f"{model_name}_scaler", 'daily', bypass_cache=bypass_cache)
        weekly_model = model_manager.load_model(
            model_name, 'weekly', bypass_cache=bypass_cache)
        weekly_scaler = model_manager.load_model(
            f"{model_name}_scaler", 'weekly', bypass_cache=bypass_cache)

        if not all([daily_model, daily_scaler, weekly_model, weekly_scaler]):
            return jsonify({
                'error': 'Models not found',
                'message': f'Please train models for {ticker} first'
            }), 404

        # Get metadata
        metadata = model_manager.get_model_metadata(model_name, 'daily')
        threshold = metadata.get('threshold', 0.005)
        use_extra_features = metadata.get('use_extra_features', False)

        # Get recent data
        # Get enough data for feature calculation
        data = get_index_data(ticker, '1y')

        # Prepare data and get predictions
        from stock_analysis.ai.technical_analysis.movement_classification import get_recent_predictions
        predictions = get_recent_predictions(
            data,
            data,  # daily_data
            daily_model,
            daily_scaler,
            data,  # weekly_data
            weekly_model,
            weekly_scaler,
            days=days,
            threshold=threshold,
            use_extra_features=use_extra_features
        )

        return jsonify({
            'ticker': ticker,
            'predictions': predictions.to_dict(orient='records'),
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error getting predictions for {ticker}: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route(f'{API_PREFIX}/models/movement', methods=['GET'])
def list_movement_models():
    """List all available movement classifier models"""
    try:
        models = model_manager.list_models()
        # Filter to only movement classifier models
        movement_models = [
            model for model in models
            if not model['name'].endswith('_scaler')  # Exclude scalers
            and '_movement' in model['name']
        ]

        return jsonify({
            'models': movement_models
        })

    except Exception as e:
        logger.error(f"Error listing movement models: {str(e)}")
        return jsonify({
            'error': 'Failed to list models',
            'message': str(e)
        }), 500


@app.route('/movement-analysis')
def movement_analysis():
    """Serve the movement analysis interface"""
    return render_template('movement_analysis.html')


if __name__ == '__main__':
    app.config.from_object(Config)
    logger.info(f"Starting Flask server on {Config.HOST}:{Config.PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )

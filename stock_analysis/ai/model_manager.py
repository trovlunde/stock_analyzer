import os
import joblib
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import hashlib
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class CachedModel:
    def __init__(self, model, timestamp: float):
        self.model = model
        self.timestamp = timestamp


class ModelManager:
    def __init__(self, models_dir="/app/models", cache_timeout_minutes: int = 60):
        self.models_dir = models_dir
        self.cache_timeout_minutes = cache_timeout_minutes
        self.model_cache: Dict[str, CachedModel] = {}
        os.makedirs(models_dir, exist_ok=True)

    def _get_model_path(self, model_name, model_type):
        """Generate a path for the model file"""
        return os.path.join(self.models_dir, f"{model_name}_{model_type}.joblib")

    def _get_model_metadata_path(self, model_name, model_type):
        """Generate a path for the model metadata file"""
        return os.path.join(self.models_dir, f"{model_name}_{model_type}_metadata.joblib")

    def _get_cache_key(self, model_name: str, model_type: str) -> str:
        """Generate a unique cache key for a model"""
        return f"{model_name}_{model_type}"

    def _is_cache_valid(self, cache_entry: CachedModel) -> bool:
        """Check if a cached model is still valid"""
        if not cache_entry:
            return False

        cache_age = time.time() - cache_entry.timestamp
        return cache_age < (self.cache_timeout_minutes * 60)

    def save_model(self, model, model_name, model_type, metadata: Optional[Dict[str, Any]] = None):
        """Save a model and its metadata to disk"""
        try:
            model_path = self._get_model_path(model_name, model_type)
            metadata_path = self._get_model_metadata_path(
                model_name, model_type)

            # Save the model
            joblib.dump(model, model_path)

            # Save metadata
            metadata = metadata or {}
            metadata.update({
                'saved_at': datetime.now().isoformat(),
                'model_name': model_name,
                'model_type': model_type,
                'cache_timeout_minutes': self.cache_timeout_minutes
            })
            joblib.dump(metadata, metadata_path)

            # Update cache
            cache_key = self._get_cache_key(model_name, model_type)
            self.model_cache[cache_key] = CachedModel(model, time.time())

            logger.info(f"Saved model {model_name} of type {model_type}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False

    def load_model(self, model_name: str, model_type: str, bypass_cache: bool = False) -> Optional[Any]:
        """
        Load a model from disk with caching

        Args:
            model_name: Name of the model
            model_type: Type of the model
            bypass_cache: If True, will load directly from disk ignoring cache
        """
        try:
            cache_key = self._get_cache_key(model_name, model_type)

            # Check cache first (unless bypassing)
            if not bypass_cache:
                cached = self.model_cache.get(cache_key)
                if cached and self._is_cache_valid(cached):
                    logger.info(
                        f"Using cached model {model_name} of type {model_type}")
                    return cached.model

            # Load from disk
            model_path = self._get_model_path(model_name, model_type)
            if not os.path.exists(model_path):
                logger.warning(
                    f"Model {model_name} of type {model_type} not found")
                return None

            model = joblib.load(model_path)

            # Update cache (unless bypassing)
            if not bypass_cache:
                self.model_cache[cache_key] = CachedModel(model, time.time())

            logger.info(
                f"Loaded model {model_name} of type {model_type} from disk")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None

    def get_model_metadata(self, model_name, model_type):
        """Get metadata for a model"""
        try:
            metadata_path = self._get_model_metadata_path(
                model_name, model_type)
            if not os.path.exists(metadata_path):
                return None
            return joblib.load(metadata_path)
        except Exception as e:
            logger.error(f"Error loading metadata for {model_name}: {str(e)}")
            return None

    def list_models(self):
        """List all available models"""
        try:
            models = []
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.joblib') and not filename.endswith('_metadata.joblib'):
                    model_info = filename.replace('.joblib', '').split('_')
                    if len(model_info) >= 2:
                        model_name = '_'.join(model_info[:-1])
                        model_type = model_info[-1]
                        metadata = self.get_model_metadata(
                            model_name, model_type)

                        # Check cache status
                        cache_key = self._get_cache_key(model_name, model_type)
                        cached_model = self.model_cache.get(cache_key)
                        cache_status = "not_cached"
                        if cached_model:
                            cache_status = "valid" if self._is_cache_valid(
                                cached_model) else "expired"

                        models.append({
                            'name': model_name,
                            'type': model_type,
                            'metadata': metadata,
                            'cache_status': cache_status
                        })
            return models
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def delete_model(self, model_name, model_type):
        """Delete a model and its metadata"""
        try:
            model_path = self._get_model_path(model_name, model_type)
            metadata_path = self._get_model_metadata_path(
                model_name, model_type)

            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            # Remove from cache
            cache_key = self._get_cache_key(model_name, model_type)
            self.model_cache.pop(cache_key, None)

            logger.info(f"Deleted model {model_name} of type {model_type}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False

    def clear_cache(self):
        """Clear all cached models"""
        self.model_cache.clear()
        logger.info("Model cache cleared")

    @staticmethod
    def generate_model_name(features, parameters):
        """Generate a unique model name based on features and parameters"""
        # Create a string representation of features and parameters
        model_config = f"{sorted(features)}_{sorted(parameters.items())}"
        # Generate a hash of the configuration
        return hashlib.md5(model_config.encode()).hexdigest()[:12]

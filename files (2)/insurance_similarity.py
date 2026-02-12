"""
Insurance Policy Similarity Matcher - Production Script

This module provides a production-ready implementation for finding similar
insurance policies to assist underwriters in risk analysis.

Usage:
    from insurance_similarity import PolicySimilaritySystem
    
    system = PolicySimilaritySystem()
    system.train(historical_data)
    
    similar_policies = system.find_similar(new_policy, top_n=3)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import shap
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsurancePolicyPreprocessor:
    """Handles all data preprocessing for insurance policies."""
    
    def __init__(self):
        self.numerical_scaler = RobustScaler()
        self.categorical_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw data."""
        df_eng = df.copy()
        
        # Financial ratios
        df_eng['tiv_per_location'] = (
            df_eng['policy_tiv'] / (df_eng['highest_location_tiv'] + 1)
        )
        df_eng['revenue_per_employee'] = (
            df_eng['Revenue'] / (df_eng['EMP_TOT'] + 1)
        )
        df_eng['sales_to_revenue_ratio'] = (
            df_eng['SLES_VOL'] / (df_eng['Revenue'] + 1)
        )
        
        # Company age
        current_year = datetime.now().year
        df_eng['company_age'] = current_year - df_eng['YR_STRT']
        
        # Coordinate features
        df_eng['lat_consistency'] = np.abs(df_eng['LAT_NEW'] - df_eng['LATIT'])
        df_eng['long_consistency'] = np.abs(df_eng['LONG_NEW'] - df_eng['LONGIT'])
        
        # Log transforms for skewed features
        df_eng['log_policy_tiv'] = np.log1p(df_eng['policy_tiv'])
        df_eng['log_revenue'] = np.log1p(df_eng['Revenue'])
        df_eng['log_employees'] = np.log1p(df_eng['EMP_TOT'])
        
        return df_eng
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str],
        fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables."""
        df_encoded = df.copy()
        encoded_features = []
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            # Frequency encoding
            if fit:
                freq_encoding = df[col].value_counts(normalize=True).to_dict()
                self.categorical_encoders[f'{col}_freq'] = freq_encoding
            
            freq_encoding = self.categorical_encoders.get(f'{col}_freq', {})
            df_encoded[f'{col}_freq'] = df[col].map(freq_encoding).fillna(0)
            encoded_features.append(f'{col}_freq')
            
            # Label encoding for low cardinality
            if df[col].nunique() < 50:
                if fit:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    self.categorical_encoders[f'{col}_label'] = le
                
                le = self.categorical_encoders.get(f'{col}_label')
                if le is not None:
                    df_encoded[f'{col}_label'] = le.transform(df[col].astype(str))
                    encoded_features.append(f'{col}_label')
        
        return df_encoded, encoded_features
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fit and transform the data."""
        return self._transform(df, numerical_cols, categorical_cols, fit=True)
    
    def transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self._transform(df, numerical_cols, categorical_cols, fit=False)
    
    def _transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        fit: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Internal transform method."""
        # Engineer features
        df_processed = self.engineer_features(df)
        
        # Encode categorical
        df_processed, cat_encoded_features = self.encode_categorical(
            df_processed, categorical_cols, fit=fit
        )
        
        # Prepare numerical features
        num_cols_available = [col for col in numerical_cols if col in df_processed.columns]
        engineered_num_cols = [
            'tiv_per_location', 'revenue_per_employee', 'sales_to_revenue_ratio',
            'company_age', 'lat_consistency', 'long_consistency',
            'log_policy_tiv', 'log_revenue', 'log_employees'
        ]
        
        all_num_cols = num_cols_available + engineered_num_cols
        
        # Handle missing values
        df_processed[all_num_cols] = df_processed[all_num_cols].fillna(
            df_processed[all_num_cols].median()
        )
        df_processed[cat_encoded_features] = df_processed[cat_encoded_features].fillna(0)
        
        # Scale numerical features
        if fit:
            X_numerical = self.numerical_scaler.fit_transform(df_processed[all_num_cols])
            self.is_fitted = True
        else:
            X_numerical = self.numerical_scaler.transform(df_processed[all_num_cols])
        
        X_categorical = df_processed[cat_encoded_features].values
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_categorical])
        
        # Store feature names
        if fit:
            self.feature_names = all_num_cols + cat_encoded_features
        
        return X_combined, df_processed


class PolicySimilaritySystem:
    """
    Complete system for finding similar insurance policies.
    
    This class handles:
    - Data preprocessing
    - Dimensionality reduction
    - Clustering
    - Similarity search
    - Explainability
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_pca_components: int = 50,
        n_umap_components: int = 10,
        n_neighbors: int = 3,
        random_state: int = 42
    ):
        """
        Initialize the similarity system.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters. If None, will be determined automatically.
        n_pca_components : int
            Number of PCA components to retain
        n_umap_components : int
            Number of UMAP components for clustering
        n_neighbors : int
            Number of similar policies to find
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.n_umap_components = n_umap_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        # Components
        self.preprocessor = InsurancePolicyPreprocessor()
        self.pca = None
        self.umap_reducer = None
        self.kmeans = None
        self.nn_model = None
        self.rf_explainer = None
        self.shap_explainer = None
        
        # Data
        self.X_original = None
        self.X_transformed = None
        self.df_processed = None
        self.cluster_labels = None
        
        self.is_trained = False
        
    def _determine_optimal_clusters(self, X: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette score."""
        logger.info("Determining optimal number of clusters...")
        
        K_range = range(3, min(15, len(X) // 10))
        best_score = -1
        best_k = 5
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info(f"Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def train(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> Dict[str, float]:
        """
        Train the similarity system on historical data.
        
        Parameters:
        -----------
        df : DataFrame
            Historical insurance policies
        numerical_features : list
            List of numerical feature names
        categorical_features : list
            List of categorical feature names
            
        Returns:
        --------
        dict : Training metrics
        """
        logger.info(f"Training on {len(df)} policies...")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, df_processed = self.preprocessor.fit_transform(
            df, numerical_features, categorical_features
        )
        self.X_original = X
        self.df_processed = df_processed
        
        # Dimensionality reduction - PCA
        logger.info(f"Applying PCA (n_components={self.n_pca_components})...")
        self.pca = PCA(n_components=min(self.n_pca_components, X.shape[1]), 
                       random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        
        variance_retained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA variance retained: {variance_retained:.2%}")
        
        # Dimensionality reduction - UMAP
        logger.info(f"Applying UMAP (n_components={self.n_umap_components})...")
        self.umap_reducer = umap.UMAP(
            n_components=self.n_umap_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=self.random_state
        )
        X_umap = self.umap_reducer.fit_transform(X_pca)
        self.X_transformed = X_umap
        
        # Determine optimal clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._determine_optimal_clusters(X_umap)
        
        # Clustering
        logger.info(f"Performing K-Means clustering (k={self.n_clusters})...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20
        )
        self.cluster_labels = self.kmeans.fit_predict(X_umap)
        self.df_processed['cluster'] = self.cluster_labels
        
        # Evaluate clustering
        silhouette = silhouette_score(X_umap, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(X_umap, self.cluster_labels)
        
        logger.info(f"Silhouette score: {silhouette:.3f}")
        logger.info(f"Davies-Bouldin score: {davies_bouldin:.3f}")
        
        # Build nearest neighbor index
        logger.info("Building nearest neighbor index...")
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric='euclidean',
            algorithm='auto'
        )
        self.nn_model.fit(X_umap)
        
        # Train explainer model
        logger.info("Training explainability model...")
        self.rf_explainer = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_explainer.fit(X, self.cluster_labels)
        rf_accuracy = self.rf_explainer.score(X, self.cluster_labels)
        
        logger.info(f"Explainer model accuracy: {rf_accuracy:.2%}")
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        self.shap_explainer = shap.TreeExplainer(self.rf_explainer)
        
        self.is_trained = True
        
        metrics = {
            'n_policies': len(df),
            'n_features': X.shape[1],
            'n_clusters': self.n_clusters,
            'pca_variance_retained': variance_retained,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'explainer_accuracy': rf_accuracy
        }
        
        logger.info("Training complete!")
        return metrics
    
    def find_similar(
        self,
        new_policy: Union[pd.DataFrame, Dict],
        numerical_features: List[str],
        categorical_features: List[str],
        top_n: Optional[int] = None,
        explain: bool = True
    ) -> Dict:
        """
        Find similar policies for a new policy submission.
        
        Parameters:
        -----------
        new_policy : DataFrame or dict
            New policy data
        numerical_features : list
            List of numerical feature names
        categorical_features : list
            List of categorical feature names
        top_n : int, optional
            Number of similar policies to return
        explain : bool
            Whether to include SHAP explanations
            
        Returns:
        --------
        dict : Similar policies and explanations
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        if top_n is None:
            top_n = self.n_neighbors
        
        # Convert to DataFrame if dict
        if isinstance(new_policy, dict):
            new_policy_df = pd.DataFrame([new_policy])
        else:
            new_policy_df = new_policy.copy()
        
        # Preprocess
        X_new, _ = self.preprocessor.transform(
            new_policy_df,
            numerical_features,
            categorical_features
        )
        
        # Transform
        X_new_pca = self.pca.transform(X_new)
        X_new_umap = self.umap_reducer.transform(X_new_pca)
        
        # Find similar policies
        distances, indices = self.nn_model.kneighbors(X_new_umap)
        
        # Get similar policies (exclude first if it's the query itself)
        similar_policies = self.df_processed.iloc[indices[0][:top_n+1]].copy()
        similar_policies['similarity_distance'] = distances[0][:top_n+1]
        similar_policies['similarity_score'] = 1 / (1 + distances[0][:top_n+1])
        
        # Predict cluster
        predicted_cluster = self.kmeans.predict(X_new_umap)[0]
        
        # Prepare result
        result = {
            'query_policy': new_policy_df.iloc[0].to_dict(),
            'predicted_cluster': int(predicted_cluster),
            'similar_policies': []
        }
        
        # Format similar policies
        for idx, (_, policy) in enumerate(similar_policies.iterrows(), 1):
            if idx > top_n:
                break
                
            policy_info = {
                'rank': idx,
                'similarity_score': float(policy['similarity_score']),
                'similarity_distance': float(policy['similarity_distance']),
                'policy_tiv': float(policy['policy_tiv']),
                'revenue': float(policy['Revenue']),
                'employees': int(policy['EMP_TOT']),
                'cluster': int(policy['cluster'])
            }
            
            # Add categorical info if available
            for col in ['Product', 'Sub Product', 'Policy Industry Description',
                       'Portfolio Segmentation', 'Short Tail / Long Tail']:
                if col in policy:
                    policy_info[col.lower().replace(' ', '_')] = str(policy[col])
            
            result['similar_policies'].append(policy_info)
        
        # Add SHAP explanation
        if explain and self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(X_new)
            
            # Get feature contributions for predicted cluster
            feature_contributions = pd.DataFrame({
                'feature': self.preprocessor.feature_names,
                'shap_value': shap_values[predicted_cluster][0],
                'abs_shap_value': np.abs(shap_values[predicted_cluster][0])
            }).sort_values('abs_shap_value', ascending=False).head(10)
            
            result['explanation'] = {
                'top_features': feature_contributions[['feature', 'shap_value']].to_dict('records'),
                'base_value': float(self.shap_explainer.expected_value[predicted_cluster])
            }
        
        return result
    
    def get_cluster_profile(self, cluster_id: int) -> Dict:
        """Get profile statistics for a specific cluster."""
        if not self.is_trained:
            raise ValueError("System not trained.")
        
        cluster_data = self.df_processed[self.df_processed['cluster'] == cluster_id]
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'size_percentage': len(cluster_data) / len(self.df_processed) * 100,
            'avg_policy_tiv': float(cluster_data['policy_tiv'].mean()),
            'median_policy_tiv': float(cluster_data['policy_tiv'].median()),
            'avg_revenue': float(cluster_data['Revenue'].mean()),
            'avg_employees': float(cluster_data['EMP_TOT'].mean()),
        }
        
        # Add categorical summaries
        for col in ['Product', 'Policy Industry Description']:
            if col in cluster_data.columns:
                top_value = cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else None
                profile[f'top_{col.lower().replace(" ", "_")}'] = str(top_value)
        
        return profile
    
    def save(self, filepath: str):
        """Save the trained system to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained system.")
        
        artifacts = {
            'preprocessor': self.preprocessor,
            'pca': self.pca,
            'umap_reducer': self.umap_reducer,
            'kmeans': self.kmeans,
            'nn_model': self.nn_model,
            'rf_explainer': self.rf_explainer,
            'shap_explainer': self.shap_explainer,
            'n_clusters': self.n_clusters,
            'n_pca_components': self.n_pca_components,
            'n_umap_components': self.n_umap_components,
            'n_neighbors': self.n_neighbors,
            'random_state': self.random_state
        }
        
        joblib.dump(artifacts, filepath)
        logger.info(f"System saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PolicySimilaritySystem':
        """Load a trained system from disk."""
        artifacts = joblib.load(filepath)
        
        system = cls(
            n_clusters=artifacts['n_clusters'],
            n_pca_components=artifacts['n_pca_components'],
            n_umap_components=artifacts['n_umap_components'],
            n_neighbors=artifacts['n_neighbors'],
            random_state=artifacts['random_state']
        )
        
        system.preprocessor = artifacts['preprocessor']
        system.pca = artifacts['pca']
        system.umap_reducer = artifacts['umap_reducer']
        system.kmeans = artifacts['kmeans']
        system.nn_model = artifacts['nn_model']
        system.rf_explainer = artifacts['rf_explainer']
        system.shap_explainer = artifacts['shap_explainer']
        system.is_trained = True
        
        logger.info(f"System loaded from {filepath}")
        return system


# Example usage
if __name__ == "__main__":
    # This is a demonstration with synthetic data
    
    # Define features
    numerical_features = [
        'DUNS_NUMBER_1', 'policy_tiv', 'Revenue', 'highest_location_tiv',
        'POSTAL_CD', 'LAT_NEW', 'LATIT', 'LONGIT', 'LONG_NEW', 'SIC_1',
        'EMP_TOT', 'SLES_VOL', 'YR_STRT', 'STAT_IND', 'SUBS_IND', 'outliers'
    ]
    
    categorical_features = [
        'Product', 'Sub Product', 'Portfolio Segmentation',
        'Programme Type', 'Short Tail / Long Tail', 'Policy Industry Description'
    ]
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'DUNS_NUMBER_1': np.random.randint(100000000, 999999999, n_samples),
        'policy_tiv': np.random.lognormal(15, 2, n_samples),
        'Revenue': np.random.lognormal(16, 1.5, n_samples),
        'highest_location_tiv': np.random.lognormal(14, 2, n_samples),
        'POSTAL_CD': np.random.randint(10000, 99999, n_samples),
        'LAT_NEW': np.random.uniform(25, 48, n_samples),
        'LATIT': np.random.uniform(25, 48, n_samples),
        'LONG_NEW': np.random.uniform(-120, -70, n_samples),
        'LONGIT': np.random.uniform(-120, -70, n_samples),
        'SIC_1': np.random.choice([1731, 3571, 5411, 7372, 8062], n_samples),
        'EMP_TOT': np.random.lognormal(5, 2, n_samples).astype(int),
        'SLES_VOL': np.random.lognormal(16, 1.5, n_samples),
        'YR_STRT': np.random.randint(1950, 2020, n_samples),
        'STAT_IND': np.random.choice([0, 1], n_samples),
        'SUBS_IND': np.random.choice([0, 1], n_samples),
        'outliers': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'Product': np.random.choice(['Property', 'Casualty', 'Professional Liability'], n_samples),
        'Sub Product': np.random.choice(['General Liability', 'E&O', 'D&O', 'Property Damage'], n_samples),
        'Portfolio Segmentation': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], n_samples),
        'Programme Type': np.random.choice(['Standard', 'Package', 'Umbrella'], n_samples),
        'Short Tail / Long Tail': np.random.choice(['Short Tail', 'Long Tail'], n_samples),
        'Policy Industry Description': np.random.choice(
            ['Manufacturing', 'Technology', 'Healthcare', 'Retail', 'Construction'], n_samples
        )
    }
    
    df = pd.DataFrame(data)
    
    # Initialize and train system
    system = PolicySimilaritySystem(n_clusters=5)
    metrics = system.train(df, numerical_features, categorical_features)
    
    print("\n" + "="*80)
    print("TRAINING METRICS")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Test with a new policy
    new_policy = df.sample(1).to_dict('records')[0]
    result = system.find_similar(new_policy, numerical_features, categorical_features, top_n=3)
    
    print("\n" + "="*80)
    print("SIMILARITY SEARCH RESULT")
    print("="*80)
    print(f"\nQuery Policy TIV: ${result['query_policy']['policy_tiv']:,.0f}")
    print(f"Predicted Cluster: {result['predicted_cluster']}")
    
    print(f"\nTop 3 Similar Policies:")
    for policy in result['similar_policies']:
        print(f"\n  Rank {policy['rank']} - Similarity Score: {policy['similarity_score']:.3f}")
        print(f"    TIV: ${policy['policy_tiv']:,.0f}")
        print(f"    Revenue: ${policy['revenue']:,.0f}")
        print(f"    Cluster: {policy['cluster']}")
    
    # Save system
    system.save('policy_similarity_system.pkl')
    
    print("\n" + "="*80)
    print("System saved successfully!")
    print("="*80)

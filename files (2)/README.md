# Insurance Policy Similarity Matching System

## 📋 Overview

This system helps underwriters identify similar historical or renewal insurance policies when evaluating new business submissions. By finding the top 3 most similar policies, underwriters can make more informed risk assessments based on historical data.

## 🎯 Business Problem

When underwriters receive new insurance policy applications, they need to:
- Assess risk accurately
- Price policies competitively
- Reference similar past policies for guidance
- Make consistent decisions across submissions

**Solution**: An AI-powered similarity matching system that finds the most relevant historical policies based on comprehensive feature analysis.

## 🏗️ Architecture

### Components

1. **Data Preprocessing**
   - Feature engineering (financial ratios, company metrics)
   - Categorical encoding (frequency + label encoding)
   - Robust scaling for outlier handling

2. **Dimensionality Reduction**
   - PCA for variance retention
   - UMAP for non-linear structure preservation

3. **Clustering**
   - K-Means for policy segmentation
   - Automatic optimal cluster determination
   - Multiple algorithm evaluation

4. **Similarity Search**
   - K-Nearest Neighbors for fast retrieval
   - Distance-based similarity scoring
   - Top-N policy matching

5. **Explainability**
   - SHAP values for feature importance
   - Random Forest surrogate model
   - Individual prediction explanations

### Workflow

```
New Policy → Preprocessing → Dimensionality Reduction → Similarity Search → Top 3 Matches
                                                              ↓
                                                        Explainability
                                                              ↓
                                                    Feature Contributions
```

## 📊 Features Used

### Numerical Features (16)
- `DUNS_NUMBER_1`: Business identifier
- `policy_tiv`: Total insured value
- `Revenue`: Company revenue
- `highest_location_tiv`: Highest location risk value
- `POSTAL_CD`: Postal/ZIP code
- `LAT_NEW`, `LATIT`: Latitude coordinates
- `LONG_NEW`, `LONGIT`: Longitude coordinates
- `SIC_1`: Standard Industrial Classification
- `EMP_TOT`: Total employees
- `SLES_VOL`: Sales volume
- `YR_STRT`: Year company started
- `STAT_IND`: Status indicator
- `SUBS_IND`: Subsidiary indicator
- `outliers`: Outlier flag

### Categorical Features (32)
Including but not limited to:
- Product types (Property, Casualty, Professional Liability)
- Industry classifications (NAIC codes and descriptions)
- Policy characteristics (Portfolio Segmentation, Programme Type)
- Business details (Producer, Location, Company information)

### Engineered Features
- `tiv_per_location`: Risk concentration metric
- `revenue_per_employee`: Efficiency indicator
- `sales_to_revenue_ratio`: Business model indicator
- `company_age`: Years in operation
- `lat_consistency`, `long_consistency`: Data quality metrics
- `log_policy_tiv`, `log_revenue`, `log_employees`: Log-transformed features

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
umap-learn>=0.5.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
category-encoders>=2.3.0
jupyter>=1.0.0
joblib>=1.0.0
```

### Quick Start - Jupyter Notebook

```bash
jupyter notebook insurance_policy_similarity.ipynb
```

The notebook includes:
- Complete data analysis
- Model training and evaluation
- Visualization of results
- SHAP explanations
- Example use cases

### Quick Start - Python Script

```python
from insurance_similarity import PolicySimilaritySystem
import pandas as pd

# Load your data
df = pd.read_csv('historical_policies.csv')

# Define your features
numerical_features = ['policy_tiv', 'Revenue', 'EMP_TOT', ...]
categorical_features = ['Product', 'Sub Product', ...]

# Initialize and train
system = PolicySimilaritySystem(n_clusters=5)
metrics = system.train(df, numerical_features, categorical_features)

# Find similar policies for a new submission
new_policy = {...}  # Your new policy data
result = system.find_similar(
    new_policy, 
    numerical_features, 
    categorical_features,
    top_n=3,
    explain=True
)

# Access results
print(f"Predicted Cluster: {result['predicted_cluster']}")
for policy in result['similar_policies']:
    print(f"Rank {policy['rank']}: Similarity {policy['similarity_score']:.3f}")
    print(f"  TIV: ${policy['policy_tiv']:,.0f}")

# View explanations
for feature in result['explanation']['top_features']:
    print(f"{feature['feature']}: {feature['shap_value']:.4f}")

# Save trained system
system.save('trained_model.pkl')

# Load later
loaded_system = PolicySimilaritySystem.load('trained_model.pkl')
```

## 📈 Evaluation Metrics

### Clustering Quality
- **Silhouette Score**: Measures cluster separation (higher is better)
  - Range: -1 to 1
  - Typical good score: > 0.5
  
- **Davies-Bouldin Index**: Measures cluster compactness (lower is better)
  - Range: 0 to ∞
  - Typical good score: < 1.0

- **Calinski-Harabasz Score**: Variance ratio (higher is better)

### Similarity Matching Quality
- **Cluster Consistency**: % of similar policies from same cluster
- **Product Consistency**: % of similar policies with same product type
- **Industry Consistency**: % of similar policies from same industry
- **Business Metrics**: Median TIV, Revenue, Employee differences

### Example Results
```
Clustering Metrics:
  Silhouette Score: 0.487
  Davies-Bouldin Score: 0.823
  
Similarity Matching:
  Cluster Consistency: 78.3%
  Product Consistency: 65.4%
  Industry Consistency: 58.2%
  Median TIV Difference: 15.2%
```

## 🔍 Model Explainability

### SHAP Values
The system uses SHAP (SHapley Additive exPlanations) to explain:

1. **Global Feature Importance**: Which features matter most overall
2. **Individual Predictions**: Why specific policies were matched
3. **Feature Contributions**: How each feature influenced the similarity

### Interpretation

```python
# Get explanation for a prediction
result = system.find_similar(new_policy, ..., explain=True)

for feature in result['explanation']['top_features']:
    name = feature['feature']
    value = feature['shap_value']
    
    if value > 0:
        print(f"{name} pushes toward this cluster")
    else:
        print(f"{name} pushes away from this cluster")
```

## 📁 File Structure

```
.
├── insurance_policy_similarity.ipynb  # Complete interactive notebook
├── insurance_similarity.py            # Production-ready Python module
├── README.md                          # This file
├── requirements.txt                   # Package dependencies
└── data/
    └── sample_data.csv               # Example data format
```

## 🛠️ Customization

### Adjusting Cluster Count

```python
# Auto-determine optimal clusters
system = PolicySimilaritySystem()

# Or specify manually
system = PolicySimilaritySystem(n_clusters=8)
```

### Changing Similarity Metric

```python
# In the NearestNeighbors initialization (insurance_similarity.py)
self.nn_model = NearestNeighbors(
    n_neighbors=self.n_neighbors + 1,
    metric='cosine',  # Change to 'cosine', 'manhattan', etc.
    algorithm='auto'
)
```

### Adding Custom Features

```python
# In InsurancePolicyPreprocessor.engineer_features()
df_eng['custom_ratio'] = df_eng['feature_a'] / df_eng['feature_b']
df_eng['custom_score'] = (df_eng['x'] * df_eng['y']) ** 0.5
```

## 📊 Cluster Profiling

```python
# Get profile for a specific cluster
profile = system.get_cluster_profile(cluster_id=0)

print(f"Cluster Size: {profile['size']} ({profile['size_percentage']:.1f}%)")
print(f"Average TIV: ${profile['avg_policy_tiv']:,.0f}")
print(f"Top Product: {profile['top_product']}")
print(f"Top Industry: {profile['top_policy_industry_description']}")
```

## 🎓 Best Practices

### Data Preparation
1. **Ensure data quality**: Remove duplicates, handle missing values
2. **Feature consistency**: Ensure all features are available for new policies
3. **Categorical encoding**: Maintain consistent category mappings
4. **Outlier handling**: The system uses RobustScaler but extreme outliers should be investigated

### Model Training
1. **Training data size**: Minimum 500 policies recommended, 1000+ ideal
2. **Retraining frequency**: Quarterly or when significant market changes occur
3. **Validation**: Use holdout set to validate similarity quality
4. **A/B testing**: Test with underwriters before full deployment

### Production Deployment
1. **API Integration**: Wrap in REST API (Flask/FastAPI)
2. **Caching**: Cache preprocessed features for frequent lookups
3. **Monitoring**: Track similarity scores and user feedback
4. **Fallback**: Have manual process for edge cases

## 🔄 Model Updates

### When to Retrain
- Quarterly schedule
- New product lines introduced
- Significant market changes
- Data drift detected (similarity scores decline)

### Retraining Process
```python
# Load new data
df_new = pd.read_csv('updated_policies.csv')

# Retrain
system = PolicySimilaritySystem()
metrics = system.train(df_new, numerical_features, categorical_features)

# Validate performance
# ... validation code ...

# Save new model
system.save('model_v2.pkl')
```

## 🐛 Troubleshooting

### Issue: Poor Clustering Quality (Low Silhouette Score)

**Solutions:**
- Increase n_umap_components (e.g., from 10 to 20)
- Try different clustering algorithms
- Remove irrelevant features
- Check for data quality issues

### Issue: Inconsistent Similar Policies

**Solutions:**
- Increase n_pca_components to retain more variance
- Adjust UMAP parameters (n_neighbors, min_dist)
- Ensure categorical encoding is working correctly
- Check for missing values

### Issue: Slow Performance

**Solutions:**
- Use approximate nearest neighbors (e.g., Annoy, FAISS)
- Reduce n_pca_components
- Batch process multiple queries
- Use GPU acceleration for UMAP

## 📞 Support & Contributing

### Getting Help
- Check the Jupyter notebook for detailed examples
- Review SHAP visualizations for model behavior
- Examine cluster profiles for unexpected patterns

### Contributing
Improvements welcome in areas:
- Additional feature engineering techniques
- Alternative clustering algorithms
- Performance optimizations
- Additional evaluation metrics

## 📝 License

MIT License - Free to use and modify for commercial purposes.

## 🎯 Roadmap

### Planned Features
- [ ] Real-time similarity API
- [ ] Interactive dashboard for underwriters
- [ ] Automated feature selection
- [ ] Multi-modal similarity (text + numerical)
- [ ] Incremental learning support
- [ ] Integration with policy management systems

## 📚 References

### Algorithms
- **K-Means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection"
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions"

### Libraries
- scikit-learn: https://scikit-learn.org/
- UMAP: https://umap-learn.readthedocs.io/
- SHAP: https://shap.readthedocs.io/

---

**Version**: 1.0.0  
**Last Updated**: February 2025  
**Author**: AI-Powered Insurance Analytics Team

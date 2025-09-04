import os
import streamlit as st

# Fix for Hugging Face Spaces permission issues
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'


import io
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import time
import base64
import warnings
import xgboost as xgb  
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, MACCSkeys, AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from PIL import Image

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


# Add XGBoost import with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional SHAP import with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP is not installed. Some advanced visualization features will be disabled.")
    st.info("To install SHAP: pip install shap")

# -----------------------------
# Configuration and Constants
# -----------------------------
SELECT_K = None
CV_FOLDS = 5
CHUNK_SIZE = 1000
MAX_DISPLAY_COMPOUNDS = 12
MAX_FEATURES_DISPLAY = 15

# Initialize session state variables
if 'trained_state' not in st.session_state:
    st.session_state.trained_state = {
        'model': None,
        'training_data': None,
        'model_name': None,
        'results': None,
        'feature_names': None,
        'training_time': None,
        'pipeline': None,
        'y_train': None
    }

if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'current_tab': 'Home',
        'file_uploaded': False,
        'data_preprocessed': False
    }

# -----------------------------
# Enhanced Pipeline with Memory and Performance Optimizations
# -----------------------------
class RememberingPipeline(Pipeline):
    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        
    def fit(self, X, y=None, **fit_params):
        start_time = time.time()
        super().fit(X, y, **fit_params)
        self.fit_time_ = time.time() - start_time
        self.X_train_ = X
        self.y_train_ = y
        return self

# -----------------------------
# OPTIMIZED Feature extractor with caching and parallel processing
# -----------------------------
class QSARFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, morgan_radius=2, morgan_nbits=1024, use_maccs=True, 
                 use_morgan=True, use_physchem=True, use_atom_pair=False, 
                 use_topological=False):
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.use_maccs = use_maccs
        self.use_morgan = use_morgan
        self.use_physchem = use_physchem
        self.use_atom_pair = use_atom_pair
        self.use_topological = use_topological
        self.feature_names_ = None
        self._molecule_cache = {}

    def _mol_from_smiles(self, s):
        if s in self._molecule_cache:
            return self._molecule_cache[s]
        try:
            mol = Chem.MolFromSmiles(s)
            if mol:
                # Add hydrogens for better descriptor calculation
                mol = Chem.AddHs(mol)
            self._molecule_cache[s] = mol
            return mol
        except Exception:
            self._molecule_cache[s] = None
            return None

    def _maccs(self, mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)

    def _morgan(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_nbits)
        return np.array(fp, dtype=np.float32)

    def _physchem(self, mol):
        descriptors = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol)
        ]
        return np.array(descriptors, dtype=np.float32)

    def _atom_pair(self, mol):
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
        return np.array(fp, dtype=np.float32)

    def _topological(self, mol):
        fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=512)
        return np.array(fp, dtype=np.float32)

    def fit(self, X, y=None):
        names = []
        if self.use_maccs:
            names += [f"MACCS_{i}" for i in range(167)]
        if self.use_morgan:
            names += [f"Morgan_{i}" for i in range(self.morgan_nbits)]
        if self.use_physchem:
            names += ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings", 
                     "AromaticRings", "AliphaticRings", "FractionCSP3", "HeavyAtomCount",
                     "ValenceElectrons", "RadicalElectrons"]
        if self.use_atom_pair:
            names += [f"AtomPair_{i}" for i in range(512)]
        if self.use_topological:
            names += [f"Topological_{i}" for i in range(512)]
            
        self.feature_names_ = names
        return self

    def transform(self, X):
        # Process in chunks to avoid memory issues
        chunks = [X[i:i+CHUNK_SIZE] for i in range(0, len(X), CHUNK_SIZE)]
        all_feats = []
        valids = []
        
        for chunk in chunks:
            chunk_feats, chunk_valids = [], []
            for s in chunk:
                try:
                    mol = self._mol_from_smiles(s)
                    if mol is None:
                        chunk_feats.append(None)
                        chunk_valids.append(False)
                        continue
                        
                    vecs = []
                    if self.use_maccs:
                        vecs.append(self._maccs(mol))
                    if self.use_morgan:
                        vecs.append(self._morgan(mol))
                    if self.use_physchem:
                        vecs.append(self._physchem(mol))
                    if self.use_atom_pair:
                        vecs.append(self._atom_pair(mol))
                    if self.use_topological:
                        vecs.append(self._topological(mol))
                        
                    chunk_feats.append(np.concatenate(vecs).astype(np.float32))
                    chunk_valids.append(True)
                    
                except Exception as e:
                    # Log the error but continue processing other molecules
                    st.warning(f"Failed to featurize molecule {s}: {str(e)}")
                    chunk_feats.append(None)
                    chunk_valids.append(False)
                    
            all_feats.extend(chunk_feats)
            valids.extend(chunk_valids)
            
        self.last_valid_mask_ = np.array(valids, dtype=bool)
        valid_rows = [f for f in all_feats if f is not None]
        
        if len(valid_rows) == 0:
            return np.empty((0, len(self.feature_names_)), dtype=np.float32)
            
        result = np.vstack(valid_rows)
        # Handle potential NaN values
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

# Enhanced Utility Functions
# -----------------------------
def reset_training_state():
    """Completely reset the training state"""
    st.session_state.trained_state = {
        'model': None,
        'training_data': None,
        'model_name': None,
        'results': None,
        'feature_names': None,
        'training_time': None,
        'pipeline': None,
        'y_train': None
    }
    # Also clear any prediction results
    if 'prediction_results' in st.session_state:
        del st.session_state.prediction_results
    st.success("Training state reset! You can now train a new model.")


def prepare_training_frame(df_raw):
    df = df_raw.copy()
    
    # More robust column detection
    possible_smiles_cols = ["smiles", "smile", "canonical_smiles", "structure", "mol", "molecule"]
    possible_ic50_cols = ["ic50", "ic50_nm", "ic50_m", "ic50_um", "activity", "potency", "ki", "ec50", "pic50"]
    
    smiles_col = None
    ic50_col = None
    
    # Case-insensitive column matching
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in possible_smiles_cols and smiles_col is None:
            smiles_col = col
        if col_lower in possible_ic50_cols and ic50_col is None:
            ic50_col = col
    
    if smiles_col is None:
        # Try to find any column that might contain SMILES by pattern
        for col in df.columns:
            if df[col].dtype == 'object' and any('c' in str(val).lower() for val in df[col].head(5)):
                if sum([1 for val in df[col].head(10) if isinstance(val, str) and 
                       (('c' in val and '=' in val) or ('c' in val and '(' in val))]) > 3:
                    smiles_col = col
                    break
    
    if smiles_col is None:
        raise ValueError("Could not find a SMILES column. Please ensure your CSV contains a column with molecular structures.")
    
    if ic50_col is None:
        # Try to find numeric columns that might contain activity values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ic50_col = numeric_cols[0]
            st.warning(f"Using '{ic50_col}' as activity column. Please verify this is correct.")
        else:
            raise ValueError("Could not find an IC50 column. Please ensure your CSV contains a column with activity values.")
    
    df = df[[smiles_col, ic50_col]].rename(columns={smiles_col: "SMILES", ic50_col: "IC50"})
    
    # More robust IC50 value extraction
    def extract_numeric_value(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        
        # Handle common formats: "10 nM", "IC50 = 1.2μM", ">100", "<0.1"
        val_str = str(val).lower().strip()
        
        # Remove common prefixes
        for prefix in ["ic50", "ic50=", "ic50 =", "ki", "ki=", "ec50", "ec50="]:
            if val_str.startswith(prefix):
                val_str = val_str[len(prefix):].strip()
        
        # Handle inequality signs
        inequality_multiplier = 1.0
        if val_str.startswith('>'):
            val_str = val_str[1:].strip()
            inequality_multiplier = 1.1  # Slightly above the value
        elif val_str.startswith('<'):
            val_str = val_str[1:].strip()
            inequality_multiplier = 0.9  # Slightly below the value
        
        # Extract numeric part
        numeric_part = ''.join([c for c in val_str if c.isdigit() or c in ['.', '-', '+']])
        if not numeric_part:
            return np.nan
            
        try:
            value = float(numeric_part) * inequality_multiplier
        except:
            return np.nan
            
        # Handle units
        if 'nm' in val_str:
            return value  # Already in nM
        elif 'μm' in val_str or 'um' in val_str:
            return value * 1000  # Convert μM to nM
        elif 'mm' in val_str:
            return value * 1000000  # Convert mM to nM
        elif 'm' in val_str and 'mm' not in val_str:
            return value * 1000000000  # Convert M to nM
        elif 'pm' in val_str:
            return value / 1000  # Convert pM to nM
            
        return value
    
    df["IC50_numeric"] = df["IC50"].apply(extract_numeric_value)
    
    # Remove extreme outliers (outside 5 standard deviations)
    mean_ic50 = df["IC50_numeric"].mean()
    std_ic50 = df["IC50_numeric"].std()
    if not np.isnan(std_ic50) and std_ic50 > 0:
        df = df[(df["IC50_numeric"] > mean_ic50 - 5*std_ic50) & 
                (df["IC50_numeric"] < mean_ic50 + 5*std_ic50)]
    
    df = df.dropna(subset=["SMILES", "IC50_numeric"])
    
    if len(df) == 0:
        raise ValueError("No valid data found after cleaning. Please check your SMILES strings are valid and IC50 values contain numbers.")
    
    # Convert to log scale (pIC50)
    df["pIC50"] = -np.log10(df["IC50_numeric"] * 1e-9)  # Convert nM to M and then to pIC50
    
    return df[["SMILES", "pIC50", "IC50_numeric"]].reset_index(drop=True)


class SafeKNN(RegressorMixin, BaseEstimator):
    """KNN regressor that handles small datasets safely"""
    def __init__(self, n_neighbors=5, **kwargs):
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.knn = None
        self.fallback_model = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # If we don't have enough samples for KNN, use a fallback model
        if n_samples <= self.n_neighbors:
            print(f"Warning: Only {n_samples} samples available. Using LinearRegression instead of KNN (requires at least {self.n_neighbors + 1} samples).")
            self.fallback_model = LinearRegression()
            self.fallback_model.fit(X, y)
            self.knn = None
        else:
            self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, **self.kwargs)
            self.knn.fit(X, y)
            self.fallback_model = None
            
        return self
    
    def predict(self, X):
        if self.fallback_model is not None:
            return self.fallback_model.predict(X)
        else:
            return self.knn.predict(X)
    
    def __getattr__(self, name):
        # Delegate other methods to the underlying KNN model if available
        if self.knn is not None and hasattr(self.knn, name):
            return getattr(self.knn, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def build_pipeline(model_name="RandomForest", random_state=42, use_feature_selection=False, 
                  k_features=50, use_polynomial=False, poly_degree=2, X_train=None):
    # Enhanced model configurations with hyperparameter options
    
    # Calculate appropriate n_neighbors based on available data
    n_neighbors = 5  # Default value
    if X_train is not None and hasattr(X_train, 'shape'):
        n_samples = X_train.shape[0]
        n_neighbors = min(5, max(1, n_samples - 1))  # Ensure valid number of neighbors
    
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=2, 
            n_jobs=-1, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.05, 
            subsample=0.8, random_state=random_state
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, max_depth=20, n_jobs=-1, 
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=7, learning_rate=0.05, 
            random_state=random_state, early_stopping=True
        ),
        "LinearRegression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, max_iter=1000, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000, random_state=random_state),
        "SVR(RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
        "LinearSVR": LinearSVR(C=1.0, random_state=random_state, max_iter=1000),
        "KNN": KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=random_state),
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        )
    
    model = models.get(model_name, RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state))

    steps = [("feats", QSARFeaturizer())]
    
    # ADD POLYNOMIAL FEATURES STEP
    if use_polynomial:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    
    steps.append(("scale", RobustScaler()))
    
    # Optional feature selection
    if use_feature_selection and k_features > 0:
        steps.append(("select", SelectKBest(score_func=mutual_info_regression, k=min(k_features, 100))))
    
    steps.append(("model", model))
    return RememberingPipeline(steps)

def calculate_similarity(smiles_list, training_smiles):
    # Precompute training fingerprints once (limit to 1000 for performance)
    train_sample = training_smiles
    if len(training_smiles) > 1000:
        sample_size = min(1000, len(training_smiles))
        train_sample = np.random.choice(training_smiles, sample_size, replace=False)
    
    train_fps = []
    for train_smi in train_sample:
        train_mol = Chem.MolFromSmiles(train_smi)
        if train_mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, 1024))
    
    similarities = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            similarities.append(0.0)
            continue
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        
        if train_fps:
            # Use bulk calculation for better performance
            bulk_similarities = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_similarity = max(bulk_similarities) if bulk_similarities else 0.0
        else:
            max_similarity = 0.0
        
        similarities.append(max_similarity)
    
    return similarities

def calculate_additional_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Pearson correlation
    r, p_value = pearsonr(y_true, y_pred)
    
    # Calculate explained variance
    explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true)) if np.var(y_true) > 0 else 0
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "pearson_r": r,
        "p_value": p_value,
        "explained_variance": explained_variance,
        "mape": mape
    }

def optimize_hyperparameters(X, y, model_name="RandomForest"):
    """Simple hyperparameter optimization for better performance"""
    if model_name == "RandomForest":
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    elif model_name == "GradientBoosting":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    else:
        return None
    
    # Create a simple pipeline for tuning
    base_pipe = Pipeline([
        ("feats", QSARFeaturizer()),
        ("scale", RobustScaler())
    ])
    
    X_transformed = base_pipe.fit_transform(X)
    
    model = build_pipeline(model_name).named_steps['model']
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_transformed, y)
    
    return grid_search.best_params_


# -----------------------------
# Y-Randomization Functions
# -----------------------------

def y_randomization_test(X, y, pipeline, n_permutations=100, random_state=42):
    """
    Perform Y-randomization test to validate model significance
    
    Parameters:
    X: feature matrix (SMILES strings)
    y: target values
    pipeline: trained pipeline object
    n_permutations: number of random permutations to test
    random_state: random seed for reproducibility
    
    Returns:
    Dictionary with original performance and permutation test results
    """
    # Store original performance
    original_r2 = pipeline.score(X, y)
    
    # Perform permutation test
    permutation_scores = []
    permutation_r2_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_permutations):
        status_text.text(f"Running permutation {i+1}/{n_permutations}")
        progress_bar.progress((i + 1) / n_permutations)
        
        # Shuffle target variable
        y_permuted = np.random.RandomState(random_state + i).permutation(y)
        
        try:
            # Create a new pipeline instance to avoid contamination
            model_name = pipeline.named_steps['model'].__class__.__name__
            perm_pipe = build_pipeline(model_name, random_state=random_state + i)
            
            # Train on permuted data
            perm_pipe.fit(X, y_permuted)
            
            # Score on permuted data
            perm_score = perm_pipe.score(X, y_permuted)
            permutation_scores.append(perm_score)
            
            # Also calculate R² for consistency
            y_pred = perm_pipe.predict(X)
            perm_r2 = r2_score(y_permuted, y_pred)
            permutation_r2_scores.append(perm_r2)
            
        except Exception as e:
            st.warning(f"Permutation {i+1} failed: {str(e)}")
            continue
    
    # Calculate p-value (proportion of permutations with better or equal performance)
    p_value = np.mean(np.array(permutation_scores) >= original_r2)
    
    status_text.text("Y-randomization test completed!")
    progress_bar.progress(100)
    
    return {
        'original_r2': original_r2,
        'permutation_scores': permutation_scores,
        'permutation_r2_scores': permutation_r2_scores,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.05  # Standard significance threshold
    }

def display_y_randomization_results(y_rand_results):
    """Display Y-randomization test results in a user-friendly format"""
    st.subheader("Y-Randomization Test Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original R²", f"{y_rand_results['original_r2']:.3f}")
    
    with col2:
        mean_perm_r2 = np.mean(y_rand_results['permutation_r2_scores'])
        st.metric("Mean Permutation R²", f"{mean_perm_r2:.3f}")
    
    with col3:
        st.metric("P-value", f"{y_rand_results['p_value']:.4f}")
        if y_rand_results['p_value'] < 0.05:
            st.success("Model is significant (p < 0.05)")
        else:
            st.error("Model may not be significant (p ≥ 0.05)")
    
    # Create histogram of permutation scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_rand_results['permutation_r2_scores'], bins=20, alpha=0.7, 
            label='Permutation Scores')
    ax.axvline(y_rand_results['original_r2'], color='red', linestyle='--', 
               linewidth=2, label=f'Original R² ({y_rand_results["original_r2"]:.3f})')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Y-Randomization Test Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Interpretation
    st.markdown("### Interpretation")
    if y_rand_results['p_value'] < 0.05:
        st.success("""
         **Good news!** Your model appears to be learning real structure-activity relationships.
        - The original model performance is significantly better than random chance
        - This suggests the model is capturing meaningful patterns in your data
        - The model is likely to generalize well to new, similar compounds
        """)
    else:
        st.error("""
         **Caution!** Your model may not be learning meaningful patterns.
        - The model performance is not significantly better than random chance
        - This could indicate:
          - The dataset is too small
          - The features don't capture relevant molecular properties
          - The activity values contain too much noise
          - The model is overfitting
        - Consider collecting more data or improving feature engineering
        """)

def train_and_evaluate(df, model_name="RandomForest", 
                      use_feature_selection=False, k_features=50, 
                      use_polynomial=False, poly_degree=2, 
                      tune_hyperparams=False,
                      test_size=0.2, cv_folds=5):
     
    # Split dataset into train/test
    X = df["SMILES"].values
    y = df["pIC50"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) 
    

     # ADD HYPERPARAMETER TUNING
    best_params = None
    if tune_hyperparams and len(df) > 50:  # Only tune if we have enough data
        with st.spinner("Optimizing hyperparameters..."):
            X_temp, y_temp = df["SMILES"].values, df["pIC50"].values
            best_params = optimize_hyperparameters(X_temp, y_temp, model_name)
            if best_params:
                st.success(f"Best parameters found: {best_params}")

                
    if len(df) > 1000:
        st.warning(f"Large dataset detected ({len(df)} compounds). Using optimized settings.")
    
    X, y = df["SMILES"].values, df["pIC50"].values
    
    # For very large datasets, use smaller test set
    test_size = 0.15 if len(df) > 2000 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=pd.qcut(y, 5) if len(df) > 100 else None
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    status_text.text("Building pipeline...")
    pipe = build_pipeline(model_name, use_feature_selection=use_feature_selection, k_features=k_features)
    progress_bar.progress(10)
    
    try:
        status_text.text("Training model...")
        pipe.fit(X_train, y_train)
        progress_bar.progress(50)
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None
        
    status_text.text("Making predictions...")
    y_pred = pipe.predict(X_test)
    
    # Calculate all metrics
    metrics = calculate_additional_metrics(y_test, y_pred)
    mse, r2 = metrics["mse"], metrics["r2"]
    
    # Display metrics during training
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{r2:.3f}")
        col2.metric("RMSE", f"{metrics['rmse']:.3f}")
        col3.metric("MAE", f"{metrics['mae']:.3f}")
    
    progress_bar.progress(70)
    
    cv_summary = (np.nan, np.nan)
    cv_preds_df = None
    
    # Use simpler validation for large datasets
    if len(df) <= 2000:
        try:
            status_text.text("Running cross-validation...")
            cv = KFold(n_splits=min(5, CV_FOLDS), shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
            cv_summary = (float(np.mean(cv_scores)), float(np.std(cv_scores)))
            
            cv_pred = cross_val_predict(
                build_pipeline(model_name, use_feature_selection=use_feature_selection, k_features=k_features), 
                X_train, y_train, cv=cv, n_jobs=1
            )
            cv_preds_df = pd.DataFrame(
                {"SMILES": X_train, "Actual_pIC50": y_train, "Pred_pIC50": cv_pred}
            )
            
        except Exception as e:
            st.warning(f"Cross-validation skipped: {str(e)}")
    else:
        st.info("Cross-validation skipped for very large dataset.")
    
    progress_bar.progress(90)
    
    # Get feature names
    feature_names = pipe.named_steps["feats"].feature_names_
    if use_feature_selection and "select" in pipe.named_steps:
        selected_mask = pipe.named_steps["select"].get_support()
        feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    results = {
        "pipeline": pipe,
        "metrics": metrics,
        "mse": float(mse),
        "r2": float(r2),
        "cv_mean_r2": cv_summary[0],
        "cv_std_r2": cv_summary[1],
        "test_preds": pd.DataFrame({"SMILES": X_test, "Actual_pIC50": y_test, "Pred_pIC50": y_pred}),
        "cv_preds": cv_preds_df,
        "X_train": X_train,
        "y_train": y_train,
        "model_name": model_name,
        "feature_names": feature_names,
        "training_time": pipe.fit_time_
    }
    
    # Store the trained model in session state - CONSISTENTLY
    st.session_state.trained_state = {
        'model': pipe,
        'training_data': X_train,
        'model_name': model_name,
        'results': results,
        'feature_names': feature_names,
        'training_time': pipe.fit_time_,
        'pipeline': pipe,
        'y_train': y_train  # Added this line
    }
    
    progress_bar.progress(100)
    status_text.text("Training completed!")
    
    
    
    # Y-Randomization validation (for datasets that aren't too large)
    if len(df) <= 1000:  # Only run for reasonably sized datasets
        st.markdown("---")
        st.subheader("Y-Randomization Validation")
        
        if st.checkbox("Run Y-randomization test (recommended for validation)", value=True):
            with st.spinner("Running Y-randomization test (this may take a while)..."):
                y_rand_results = y_randomization_test(
                    X_train, y_train, pipe, 
                    n_permutations=50,
                    random_state=42
                )
                
                # Store Y-randomization results
                results["y_randomization"] = y_rand_results
                
                # Display results
                display_y_randomization_results(y_rand_results)
    else:
        st.info("Y-randomization skipped for large datasets (>1000 compounds).")
    
    return results

def predict_with_pipeline(pipe, smiles_list):
    # Predict in chunks for large datasets
    if len(smiles_list) > 1000:
        chunks = [smiles_list[i:i+CHUNK_SIZE] for i in range(0, len(smiles_list), CHUNK_SIZE)]
        predictions = []
        for chunk in chunks:
            predictions.extend(pipe.predict(chunk))
        return predictions
    else:
        return pipe.predict(smiles_list)

def mols_from_smiles(smiles_list):
    mols = []
    for s in smiles_list:
        try:
            mols.append(Chem.MolFromSmiles(s))
        except Exception:
            mols.append(None)
    return mols

def show_mols_grid(smiles_list, legends=None, mols_per_row=4, sub_img_size=(200, 200)):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [mol for mol in mols if mol is not None]
    valid_legends = [legends[i] for i, mol in enumerate(mols) if mol is not None] if legends else None
    
    if not valid_mols:
        st.warning("No valid molecules to display.")
        return
        
    img = Draw.MolsToGridImage(
        valid_mols, 
        molsPerRow=mols_per_row, 
        subImgSize=sub_img_size, 
        legends=valid_legends, 
        useSVG=False
    )
    st.image(img)

def mol_to_image(mol, size=(300, 300)):
    """Convert RDKit mol to image"""
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

def list_all_rdkit_descriptors():
    return Descriptors._descList

def calc_selected_descriptors(smiles_list, selected_names):
    name_to_func = dict(Descriptors._descList)
    rows = []
    valid_smiles = []
    
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
            
        vals = []
        for name in selected_names:
            func = name_to_func.get(name)
            try:
                              vals.append(func(mol))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
        valid_smiles.append(s)
    
    df = pd.DataFrame(rows, columns=selected_names)
    df.insert(0, "SMILES", valid_smiles)
    return df

def model_supports_interpretation(model):
    """Check if model supports interpretation features"""
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        return True
    return False

def interpret_results(results, df_train):
    """
    Provide detailed interpretation of model results and fit assessment
    """
    st.subheader(" Model Performance Interpretation")
    
    # Check if results are available
    if results is None or 'r2' not in results:
        st.warning("Interpretation not available: No valid results")
        return
    
    # Get metrics
    r2 = results["r2"]
    mse = results["mse"]
    cv_r2 = results.get('cv_mean_r2', np.nan)
    
    # Create interpretation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
        if r2 >= 0.8:
            st.success("Excellent fit")
            st.info("The model explains ≥80% of variance in your data")
        elif r2 >= 0.6:
            st.warning("Good fit")
            st.info("The model explains 60-79% of variance")
        elif r2 >= 0.4:
            st.warning("Moderate fit")
            st.info("The model explains 40-59% of variance")
        else:
            st.error("Poor fit")
            st.info("The model explains <40% of variance. Consider feature engineering or trying different algorithms.")
    
    with col2:
        st.metric("MSE", f"{mse:.3f}")
        # Contextualize MSE based on target variable range
        y_range = df_train["pIC50"].max() - df_train["pIC50"].min()
        relative_error = np.sqrt(mse) / y_range if y_range > 0 else 0
        if relative_error < 0.1:
            st.success("Low error")
            st.info(f"RMSE is {relative_error*100:.1f}% of target range")
        elif relative_error < 0.2:
            st.warning("Moderate error")
            st.info(f"RMSE is {relative_error*100:.1f}% of target range")
        else:
            st.error("High error")
            st.info(f"RMSE is {relative_error*100:.1f}% of target range")
    
    with col3:
        if not np.isnan(cv_r2):
            st.metric("CV R²", f"{cv_r2:.3f}")
            cv_gap = abs(r2 - cv_r2)
            if cv_gap < 0.1:
                st.success("Stable model")
                st.info("Small gap between test and CV performance suggests good generalization")
            elif cv_gap < 0.2:
                st.warning("Moderate overfitting")
                st.info("Model may be slightly overfit to training data")
            else:
                st.error("Significant overfitting")
                st.info("Large performance drop in CV suggests overfitting")
        else:
            st.info("CV not performed (large dataset)")
    
    # Additional interpretation
    st.markdown("---")
    st.subheader(" Detailed Assessment")
    
    # Data quality assessment
    n_samples = len(df_train)
    st.write(f"*Dataset size*: {n_samples} compounds")
    if n_samples < 50:
        st.warning("⚠ Small dataset: Models trained on <50 compounds may not generalize well")
    elif n_samples < 100:
        st.info(" Moderate dataset: Consider collecting more data for better models")
    else:
        st.success(" Good dataset size: Sufficient for robust modeling")
    
    # Activity range assessment
    activity_range = df_train["pIC50"].max() - df_train["pIC50"].min()
    st.write(f"*Activity range*: {activity_range:.2f} pIC50 units")
    if activity_range < 1:
        st.warning("⚠ Limited activity range: Model may struggle to find patterns")
    elif activity_range < 2:
        st.info(" Moderate activity range: Reasonable for QSAR modeling")
    else:
        st.success(" Good activity range: Sufficient spread for effective modeling")
    
    # Model applicability
    st.markdown("---")
    st.subheader(" Model Applicability")
    
    if r2 >= 0.6 and not np.isnan(cv_r2) and abs(r2 - cv_r2) < 0.15:
        st.success(" This model is likely suitable for prediction")
        st.markdown("""
        - Good explanatory power (R² ≥ 0.6)
        - Reasonable generalization (small test-CV gap)
        - Can be used for similar compounds with caution
        """)
    elif r2 >= 0.4:
        st.warning(" This model has limited predictive power")
        st.markdown("""
        - Moderate explanatory power (R² = 0.4-0.6)
        - Use primarily for trend identification, not precise prediction
        - Predictions should be considered approximate
        """)
    else:
        st.error(" This model is not suitable for prediction")
        st.markdown("""
        - Low explanatory power (R² < 0.4)
        - Use only for educational purposes
        - Do not rely on predictions from this model
        """)
    
    # Recommendations for improvement
    st.markdown("---")
    st.subheader("💡 Recommendations for Improvement")
    
    if r2 < 0.6:
        st.markdown("""
        To improve model performance:
        1. *Add more data* - More compounds usually improve models
        2. *Check data quality* - Ensure accurate IC50 measurements and valid SMILES
        3. *Try different algorithms* - Some models work better with certain data types
        """)
    
    if not np.isnan(cv_r2) and abs(r2 - cv_r2) > 0.15:
        st.markdown("""
        To address overfitting:
        1. *Simplify the model* - Reduce complexity (fewer trees, shallower depth)
        2. *Use feature selection* - Focus on most important descriptors
        3. *Regularization* - Try Ridge, Lasso, or ElasticNet models
        4. *Increase training data* - More data reduces overfitting
        """)
    
    # Y-randomization interpretation if available
    if "y_randomization" in results:
        st.markdown("---")
        st.subheader("Y-Randomization Assessment")
        
        y_rand = results["y_randomization"]
        if y_rand["significant"]:
            st.success("✅ Y-randomization test passed: Model is statistically significant")
            st.info("The model performs significantly better than random chance, indicating it's learning real patterns.")
        else:
            st.error("⚠️ Y-randomization test failed: Model may not be significant")
            st.warning("The model's performance is not significantly better than random permutations. This suggests:")
            st.markdown("""
            - The model might be overfitting to noise
            - The features may not capture relevant molecular properties
            - Consider collecting more data or improving feature selection
            """)

def create_example_datasets():
    examples = {
        "Small Drug-like Molecules": pd.DataFrame({
            "SMILES": [
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                "C1=CC(=C(C=C1O)O)CO",  # Dopamine
                "CCCCC1=CN=CC=C1"  # Nicotine
            ],
            "IC50_nM": [100.0, 50.0, 75.0, 25.0, 150.0]
        }),
        "Kinase Inhibitors": pd.DataFrame({
            "SMILES": [
                "CC1=CC=C(C=C1)NC(=O)C2=CC(=NN2C3=CC=CC=C3)C4=CC=CC=C4",  # Imatinib-like
                "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Theophylline
                "COc1cc2c(cc1OC)CCN(C(=O)Cc3ccccc3)C2",  # Staurosporine-like
                "Cc1noc(C)n1-c1ccc(Cl)cc1",  # A kinase inhibitor
                "CN(C)CCCN1Cc2ccccc2Sc3ccccc13"  # Another kinase inhibitor
            ],
            "IC50_nM": [10.0, 250.0, 5.0, 30.0, 100.0]
        })
    }
    return examples

def create_feature_importance_plot(model, feature_names, max_features=MAX_FEATURES_DISPLAY):
    """Create feature importance visualization"""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:max_features]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(indices)), importances[indices][::-1], align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importances')
            plt.tight_layout()
            return fig
            
        elif hasattr(model, "coef_"):
            coefficients = model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
                
            indices = np.argsort(np.abs(coefficients))[::-1][:max_features]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['red' if coef < 0 else 'blue' for coef in coefficients[indices]]
            ax.barh(range(len(indices)), coefficients[indices][::-1], align='center', color=colors[::-1])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Top Feature Coefficients (Red = Negative, Blue = Positive)')
            plt.tight_layout()
            return fig
        
        return None
    except Exception as e:
        st.warning(f"Feature importance visualization failed: {str(e)}")
        return None

def create_shap_analysis(pipe, X_train, X_test, feature_names, max_display=10):
    """Create SHAP analysis for model interpretation"""
    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed. Please install with: pip install shap")
        return None
        
    try:
        # Get the model and preprocessing steps
        model = pipe.named_steps['model']
        preprocessor = Pipeline(pipe.steps[:-1])
        
        # Transform the data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Sample for large datasets
        if X_train_transformed.shape[0] > 1000:
            sample_idx = np.random.choice(X_train_transformed.shape[0], 1000, replace=False)
            X_train_sampled = X_train_transformed[sample_idx]
        else:
            X_train_sampled = X_train_transformed
            
        if X_test_transformed.shape[0] > 100:
            sample_idx_test = np.random.choice(X_test_transformed.shape[0], 100, replace=False)
            X_test_sampled = X_test_transformed[sample_idx_test]
        else:
            X_test_sampled = X_test_transformed
        
        # Create explainer
        if hasattr(model, 'predict'):
            explainer = shap.Explainer(model, X_train_sampled, feature_names=feature_names)
            shap_values = explainer(X_test_sampled)
            
            # Summary plot
            fig_summary = plt.figure()
            shap.summary_plot(shap_values, X_test_sampled, feature_names=feature_names, max_display=max_display, show=False)
            plt.tight_layout()
            
            return fig_summary
            
    except Exception as e:
        st.warning(f"SHAP analysis skipped: {str(e)}")
        return None

def create_pca_plot(X_train, y_train, feature_names):
    """Create PCA visualization of the feature space"""
    try:
        # Sample for large datasets
        if X_train.shape[0] > 1000:
            sample_idx = np.random.choice(X_train.shape[0], 1000, replace=False)
            X_sampled = X_train[sample_idx]
            y_sampled = y_train[sample_idx]
        else:
            X_sampled = X_train
            y_sampled = y_train
            
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sampled)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sampled, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA of Molecular Feature Space')
        plt.colorbar(scatter, label='pIC50')
        plt.tight_layout()
        
        return fig, pca.explained_variance_ratio_
        
    except Exception as e:
        st.warning(f"PCA visualization skipped: {str(e)}")
        return None, None

# -----------------------------
# Enhanced Streamlit UI with Better Layout and User Experience
# -----------------------------
# Fixed BioQuantify Landing Page

st.set_page_config(
    page_title="Bioquantify - Advanced QSAR Modeling", 
    layout="wide", 
    page_icon="my_logo.png",
    initial_sidebar_state="expanded"
)
# Professional Bioactive Theme - Human Centered & Sexy Professional
st.markdown("""
<style>
    /* Deep Bioactive Theme */
    .stApp {
        background: linear-gradient(135deg, #0c1e3e 0%, #1a365d 50%, #2c5aa0 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Premium Dark Header */
    .stApp header {
        background: rgba(12, 30, 62, 0.95) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #4f97d1;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Luxury Glass Sidebar */
    .css-1d391kg {
        background: rgba(12, 30, 62, 0.85) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid #4f97d1;
        color: #ffffff;
    }
    
    /* Premium Glass Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 2px solid rgba(79, 151, 209, 0.3);
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(12, 30, 62, 0.7) !important;
        color: #a8d1ff !important;
        border: 1px solid rgba(79, 151, 209, 0.2);
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2c5aa0 0%, #4f97d1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #4f97d1;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(44, 90, 160, 0.4);
    }
    
    /* Premium Glass Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2c5aa0 0%, #4f97d1 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 14px;
        text-transform: none;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 25px rgba(44, 90, 160, 0.4);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1a365d 0%, #3a7ca5 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(44, 90, 160, 0.6);
    }
    
    /* Premium Text Styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #e0f0ff !important;
        line-height: 1.6;
    }
    
    /* Glass Cards */
    .feature-card {
        background: rgba(12, 30, 62, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(79, 151, 209, 0.3);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border-color: #4f97d1;
    }
    
    .feature-card h4 {
        color: #4f97d1 !important;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(79, 151, 209, 0.3);
        padding-bottom: 10px;
    }
    
    .feature-card ul {
        color: #e0f0ff;
        line-height: 1.8;
    }
    
    .feature-card li {
        margin: 10px 0;
        padding-left: 12px;
        border-left: 3px solid #4f97d1;
    }
    
    /* Molecular Guide Card */
    .guide-card {
        background: rgba(12, 30, 62, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(79, 151, 209, 0.4);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
    }
    
    .guide-card ol {
        color: #e0f0ff;
        line-height: 1.8;
    }
    
    .guide-card li {
        margin: 15px 0;
        padding: 12px 16px;
        background: rgba(79, 151, 209, 0.1);
        border-radius: 8px;
        border-left: 4px solid #4f97d1;
    }
    
    .guide-card strong {
        color: #4f97d1;
        font-weight: 600;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: rgba(12, 30, 62, 0.7) !important;
        color: #ffffff !important;
        border: 2px solid rgba(79, 151, 209, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #4f97d1;
        box-shadow: 0 0 0 2px rgba(79, 151, 209, 0.2);
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(12, 30, 62, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(79, 151, 209, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Subtle Animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    .feature-card, .guide-card {
        animation: float 8s ease-in-out infinite;
    }
    
    .feature-card { animation-delay: 0s; }
    .guide-card { animation-delay: 2s; }
</style>
""", unsafe_allow_html=True)

# Navigation tabs with clean styling
tabs = st.tabs(["Home", "Train Model", "Predict", "Analysis", "Learn", "Settings"])

# -----------------------------
# Home Tab - Professional Edition
# -----------------------------
with tabs[0]:
    st.markdown('<h1 style="text-align: center; color: #4f97d1; font-weight: 700; margin-bottom: 1rem;">BIOQUANTIFY</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
    <h3 style="color: #a8d1ff; font-weight: 300;">Advanced QSAR Modeling for Drug Discovery Research</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Features overview
    st.markdown('<h2 style="color: #4f97d1;padding-bottom: 10px;">Core Capabilities</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>Molecular Featurization</h4>
        <ul>
        <li>MACCS Structural Keys</li>
        <li>Morgan Fingerprints</li>
        <li>Physicochemical Properties</li>
        <li>3D Molecular Descriptors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>Machine Learning</h4>
        <ul>
        <li>Multiple Algorithms</li>
        <li>Hyperparameter Optimization</li>
        <li>Automated Feature Selection</li>
        <li>Cross-Validation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h4>Visual Analytics</h4>
        <ul>
        <li>Model Interpretation</li>
        <li>Feature Importance</li>
        <li>Applicability Domain</li>
        <li>Real-time Results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown('<h2 style="color: #4f97d1; padding-bottom: 10px;">Getting Started</h2>', unsafe_allow_html=True)
    
    # Professional Step-by-step guide
    st.markdown("""
    <div class="guide-card">
    <ol>
    <li><strong>Prepare your data</strong>: CSV with SMILES and bioactivity values</li>
    <li><strong>Train models</strong>: Access machine learning algorithms</li>
    <li><strong>Evaluate performance</strong>: Comprehensive validation metrics</li>
    <li><strong>Make predictions</strong>: Real-time compound screening</li>
    <li><strong>Analyze results</strong>: Detailed molecular insights</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean Call-to-Action
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Modeling", use_container_width=True, type="primary"):
            st.session_state.app_state['current_tab'] = "Train Model"
    with col2:
        if st.button("Quick Predict", use_container_width=True, type="primary"):
            st.session_state.app_state['current_tab'] = "Predict"
   
   
    # Example datasets
    st.markdown("")
    st.subheader("Example Datasets")
    
    examples = create_example_datasets()
    for name, df in examples.items():
        csv = df.to_csv(index=False).encode('utf-8')
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
            st.dataframe(df.head(3), use_container_width=True)
        with col2:
            st.download_button(
                label="Download",
                data=csv,
                file_name=f"{name.replace(' ', '_')}_example.csv",
                mime="text/csv",
                key=f"dl_{name}"
            )
    
   
# -----------------------------
# Train Model Tab
# -----------------------------
with tabs[1]:
    st.subheader("Train QSAR Model")
    
    # Add reset button at the top
    if st.session_state.trained_state.get('model') is not None:
        if st.button("Train New Model (Start Fresh)", type="secondary", use_container_width=True, key="reset_train_btn"):
            reset_training_state()
            st.rerun()
    
    # Data input section
    data_source = st.radio(
        "Select Data Source",
        ["Upload CSV File", "Paste Data"],
        horizontal=True,
        key="data_source_radio" 
    )
    
    df_raw = None
    
    if data_source == "Upload CSV File":
        train_file = st.file_uploader("Upload CSV", type=["csv"], key="train_file")
        if train_file:
            df_raw = pd.read_csv(train_file)
    
    
    else:  # Paste Data
        paste_data = st.text_area("Paste CSV data (with header row)", height=150, key="paste_data_area")
        if paste_data:
            try:
                from io import StringIO
                df_raw = pd.read_csv(StringIO(paste_data))
            except Exception as e:
                st.error(f"Could not parse pasted data: {e}")
    
    if df_raw is not None:
        st.write("Data Preview:")
        st.dataframe(df_raw.head(), use_container_width=True)
        
        # Data preprocessing options
        with st.expander("Data Preprocessing Options"):
            col1, col2 = st.columns(2)
            with col1:
                remove_outliers = st.checkbox("Remove outliers", value=True, key="remove_outliers_cb")
                log_transform = st.checkbox("Log-transform activity values", value=True, key="log_transform_cb")
            with col2:
                activity_unit = st.selectbox(
                    "Activity unit",
                    ["nM", "μM", "mM", "M", "pIC50", "pKi"],
                    index=0,
                    key="activity_unit_select"
                )
        
        # Model training options
        with st.expander("Model Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox(
                    "Select Algorithm",
                    [
                        "RandomForest", "GradientBoosting", "ExtraTrees", "HistGradientBoosting",
                        "LinearRegression", "Ridge", "Lasso", "ElasticNet",
                        "SVR(RBF)", "LinearSVR", "KNN", "DecisionTree", "AdaBoost"
                    ],
                    index=0,
                    key="model_choice_select"
                )
                
                # Algorithm-specific options
                if model_choice in ["RandomForest", "ExtraTrees", "GradientBoosting"]:
                    n_estimators = st.slider("Number of estimators", 10, 500, 100, 10, key="n_estimators_slider")
                
                if model_choice in ["RandomForest", "ExtraTrees", "DecisionTree"]:
                    max_depth = st.slider("Max depth", 3, 30, 15, 1, key="max_depth_slider")
            
            with col2:
                test_size = st.slider("Test set size (%)", 10, 40, 25, 5, key="test_size_slider")
                cv_folds = st.slider("Cross-validation folds", 2, 10, 5, 1, key="cv_folds_slider")
        
        # Feature options
        with st.expander("Feature Engineering Options"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Fingerprint Types")
                use_maccs = st.checkbox("MACCS Keys", value=True, key="use_maccs_cb")
                use_morgan = st.checkbox("Morgan Fingerprints", value=True, key="use_morgan_cb")
                use_physchem = st.checkbox("Physicochemical", value=True, key="use_physchem_cb")
            with col2:
                st.write("Additional Features")
                use_atom_pair = st.checkbox("Atom Pair", value=False, key="use_atom_pair_cb")
                use_topological = st.checkbox("Topological Torsion", value=False, key="use_topological_cb")
                
                if use_morgan:
                    morgan_bits = st.slider("Morgan fingerprint bits", 128, 2048, 1024, 128, key="morgan_bits_slider")
                    morgan_radius = st.slider("Morgan radius", 1, 4, 2, 1, key="morgan_radius_slider")
            
        # Advanced Feature Engineering
        with st.expander("Advanced Feature Engineering (Recommended for better performance)"):
            col1, col2 = st.columns(2)
            with col1:
                use_polynomial = st.checkbox("Add polynomial features", value=True,
                                           help="Captures non-linear relationships between features", key="use_polynomial_cb")
                if use_polynomial:
                    poly_degree = st.slider("Polynomial degree", 2, 3, 2, 1,
                                          help="Degree 2 for quadratic, Degree 3 for cubic relationships", key="poly_degree_slider")
            
            with col2:
                use_feature_selection = st.checkbox("Enable feature selection", value=True, key="use_feature_selection_cb")
                if use_feature_selection:
                    k_features = st.slider("Number of features to keep", 10, 200, 50, 5, key="k_features_slider")
                    feature_selection_method = st.selectbox(
                        "Feature selection method",
                        ["mutual_info", "f_regression"],
                        index=0,
                        key="feature_selection_method_select"
                    )

        # Hyperparameter Optimization
        hyperparam_expander = st.expander("Hyperparameter Optimization (Slower but better models)")
        with hyperparam_expander:
            tune_hyperparams = st.checkbox("Optimize hyperparameters", value=False,
                                         help="This will take longer but may significantly improve performance", key="tune_hyperparams_cb")
            
            # Check if the model choice supports early stopping
            early_stopping_models = ["GradientBoosting", "HistGradientBoosting"]
            if model_choice in early_stopping_models:
                early_stopping = st.checkbox("Use early stopping", value=True,
                                           help="Prevents overfitting by stopping training when validation score stops improving", key="early_stopping_cb")
                if early_stopping:
                    validation_fraction = st.slider("Validation fraction for early stopping", 0.1, 0.3, 0.2, 0.05, key="validation_fraction_slider")

        # Train button
        if st.button("Train Model", type="primary", use_container_width=True, key="train_model_main_btn"):
            try:
                # Prepare data
                with st.spinner("Preprocessing data..."):
                    df_train = prepare_training_frame(df_raw)
                
                if df_train is not None and len(df_train) > 0:
                    st.success(f"Data prepared: {len(df_train)} compounds")
                    st.dataframe(df_train.head(), use_container_width=True)
                    
                    # Convert test_size from percentage to decimal
                    test_size_decimal = test_size / 100
                    
                    # Train model
                    with st.spinner("Training model..."):
                        results = train_and_evaluate(
                            df_train, 
                            model_name=model_choice,
                            use_feature_selection=use_feature_selection,
                            k_features=k_features if use_feature_selection else 50,
                            test_size=test_size_decimal,
                            cv_folds=cv_folds
                        )
                    
                    if results is not None:
                        st.success("Model training completed successfully!")
                        
                        # Display results
                        st.subheader("Training Results")
                        
                        # Metrics cards
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R²", f"{results.get('r2', 0):.3f}")
                        with col2:
                            st.metric("RMSE", f"{results.get('metrics', {}).get('rmse', 0):.3f}")
                        with col3:
                            st.metric("MAE", f"{results.get('metrics', {}).get('mae', 0):.3f}")
                        with col4:
                            cv_r2 = results.get('cv_mean_r2', float('nan'))
                            if not np.isnan(cv_r2):
                                st.metric("CV R²", f"{cv_r2:.3f}")
                            else:
                                st.metric("CV R²", "N/A")
                        
                        # Detailed results in tabs
                        results_tabs = st.tabs(["Performance", "Visualizations", "Interpretation"])
                        
                        with results_tabs[0]:
                            if 'test_preds' in results and results['test_preds'] is not None:
                                st.dataframe(results['test_preds'].head(10), use_container_width=True)
                            else:
                                st.info("No test predictions available")
                            
                            if results.get('cv_preds') is not None:
                                with st.expander("Cross-Validation Results"):
                                    st.dataframe(results['cv_preds'].head(10), use_container_width=True)
                        
                        with results_tabs[1]:
                            # Actual vs Predicted plot
                            if 'test_preds' in results and results['test_preds'] is not None:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.scatter(results['test_preds']['Actual_pIC50'], results['test_preds']['Pred_pIC50'], alpha=0.6)
                                min_val = min(results['test_preds']['Actual_pIC50'].min(), results['test_preds']['Pred_pIC50'].min())
                                max_val = max(results['test_preds']['Actual_pIC50'].max(), results['test_preds']['Pred_pIC50'].max())
                                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                                ax.set_xlabel('Actual pIC50')
                                ax.set_ylabel('Predicted pIC50')
                                ax.set_title('Actual vs Predicted Values')
                                st.pyplot(fig)
                                
                                # Residual plot
                                fig2, ax2 = plt.subplots(figsize=(8, 6))
                                residuals = results['test_preds']['Pred_pIC50'] - results['test_preds']['Actual_pIC50']
                                ax2.scatter(results['test_preds']['Pred_pIC50'], residuals, alpha=0.6)
                                ax2.axhline(y=0, color='r', linestyle='--')
                                ax2.set_xlabel('Predicted pIC50')
                                ax2.set_ylabel('Residuals')
                                ax2.set_title('Residual Plot')
                                st.pyplot(fig2)
                            else:
                                st.info("No visualization data available")
                        
                        with results_tabs[2]:
                            interpret_results(results, df_train)
                else:
                    st.error("Failed to prepare training data. Please check your input data.")
                        
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# -----------------------------
# Predict Tab - FIXED VERSION
# -----------------------------
with tabs[2]:
    st.subheader(" Predict New Molecules")
    
    # Unified function to get trained model
    def get_trained_model():
        """Get trained model from session state with consistent logic"""
        if st.session_state.trained_state['model'] is not None:
            return st.session_state.trained_state
        return None
    
    # Get model using unified function
    model_info = get_trained_model()
    trained_model = model_info['model'] if model_info else None
    model_name = model_info['model_name'] if model_info else "No model available"
    training_data = model_info.get('training_data', []) if model_info else []
    
    # Model selection
    if trained_model is None:
        st.warning("No trained model found. Please train a model first.")
        
        # Option to upload a pre-trained model
        uploaded_model = st.file_uploader("Or upload a trained model", type=["pkl", "joblib"])
        if uploaded_model:
            try:
                model = joblib.load(uploaded_model)
                st.session_state.trained_state = {
                    'model': model,
                    'model_name': "Uploaded Model",
                    'training_data': [],
                    'results': None
                }
                trained_model = model
                model_name = "Uploaded Model"
                training_data = []
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        st.success(f"Using trained model: {model_name}")
        
        # Add reset button here
        if st.button("Train New Model (Start Fresh)", type="secondary", use_container_width=True, key="analysis_reset_btn"):
            reset_training_state()
            st.rerun()
    
    # Input methods
    input_method = st.radio(
        "Input Method",
        ["Paste SMILES", "Upload CSV"],
        horizontal=True,
        key="input_method_radio"
    )
    
    smiles_list = []
    
    if input_method == "Paste SMILES":
        smiles_text = st.text_area("Enter one SMILES string per line", height=100)
        if smiles_text:
            smiles_list = [line.strip() for line in smiles_text.split('\n') if line.strip()]
    
    elif input_method == "Upload CSV":
        smiles_file = st.file_uploader("Upload CSV with SMILES column", type=["csv"])
        if smiles_file:
            df_smiles = pd.read_csv(smiles_file)
            # More robust column detection
            possible_smiles_cols = ["smiles", "smile", "canonical_smiles", "structure", "mol", "molecule"]
            smiles_col = None
            
            for col in df_smiles.columns:
                col_lower = col.strip().lower()
                if col_lower in possible_smiles_cols:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                # Use the first column if no obvious SMILES column found
                smiles_col = df_smiles.columns[0]
                st.warning(f"Using '{smiles_col}' as SMILES column. Please verify this is correct.")
            
            smiles_list = df_smiles[smiles_col].astype(str).tolist()
            st.write(f"Found {len(smiles_list)} molecules")
    
    if smiles_list and trained_model:
        if st.button("Predict Activities", type="primary", use_container_width=True, key="predict_activities_btn"):
            with st.spinner("Making predictions..."):
                try:
                    predictions = predict_with_pipeline(trained_model, smiles_list)
                    
                    # Calculate applicability domain if we have training data
                    similarities = []
                    if len(training_data) > 0:
                        similarities = calculate_similarity(smiles_list, training_data)
                    else:
                        similarities = [0.0] * len(smiles_list)  # Default value
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'SMILES': smiles_list,
                        'Predicted_pIC50': predictions,
                        'Similarity_to_Training': similarities
                    })
                    
                    # Convert pIC50 back to IC50
                    results_df['Predicted_IC50_nM'] = 10**(9 - results_df['Predicted_pIC50'])
                    
                    # Store the prediction results in session state for the Analysis tab
                    st.session_state.prediction_results = results_df
                    
                    st.success("Predictions completed!")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualize molecules
                    st.subheader("Molecule Visualization")
                    sample_size = min(8, len(results_df))
                    sample_df = results_df.head(sample_size)
                    
                    legends = [
                        f"pIC50: {p:.2f}\nSimilarity: {s:.2f}" 
                        for p, s in zip(sample_df['Predicted_pIC50'], sample_df['Similarity_to_Training'])
                    ]
                    
                    show_mols_grid(sample_df['SMILES'].tolist(), legends=legends)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# -----------------------------
# Analysis Tab - COMPLETE VERSION
# -----------------------------
with tabs[3]:
    st.subheader(" Model Analysis and Interpretation")
    
    # Use the same unified function as Predict tab
    model_info = get_trained_model()
    trained_model = model_info['model'] if model_info else None
    model_name = model_info['model_name'] if model_info else "No model available"
    training_results = model_info.get('results') if model_info else None
    training_data = model_info.get('training_data', []) if model_info else []
    feature_names = training_results.get('feature_names', []) if training_results else []

    # Check for prediction results
    prediction_results = st.session_state.get('prediction_results', None)

    if trained_model is None:
        st.warning("No trained model available for analysis. Please train a model first.")
    
    else:
        st.success(f"Analyzing model: {model_name}")
        
        # Add reset button here
        if st.button("Train New Model", type="secondary", key="analysis_tab_reset_btn"):
            reset_training_state()
            st.rerun()
        
        # Debug information
        with st.expander("Debug Information"):
            st.write("Model available:", trained_model is not None)
            st.write("Training results available:", training_results is not None)
            st.write("Feature names available:", len(feature_names) if feature_names else 0)
            st.write("Training data available:", len(training_data) if training_data is not None and len(training_data) > 0 else 0)
        
        # Create tabs for different types of analysis
        analysis_tab_names = ["Training Diagnostics", "Feature Importance"]
        if prediction_results is not None:
            analysis_tab_names.insert(0, "Prediction Results")
        if SHAP_AVAILABLE:
            analysis_tab_names.append("SHAP Analysis")
        analysis_tab_names.extend(["PCA Visualization"])
        
        analysis_tabs = st.tabs(analysis_tab_names)
        
        tab_index = 0
        
        # Prediction Results Tab
        if prediction_results is not None:
            with analysis_tabs[tab_index]:
                st.subheader("Prediction Results Analysis")
                
                # Display prediction results
                st.dataframe(prediction_results, use_container_width=True)
                
                # Similarity distribution
                if 'Similarity_to_Training' in prediction_results.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(prediction_results['Similarity_to_Training'], bins=20, alpha=0.7)
                    ax.set_xlabel('Similarity to Training Set')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Similarity Scores')
                    ax.axvline(x=0.5, color='red', linestyle='--', label='Similarity Threshold (0.5)')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Identify compounds outside applicability domain
                    low_similarity = prediction_results[prediction_results['Similarity_to_Training'] < 0.5]
                    if len(low_similarity) > 0:
                        st.warning(f"{len(low_similarity)} compounds have low similarity (<0.5) to training set")
                        st.write("These predictions may be less reliable:")
                        st.dataframe(low_similarity[['SMILES', 'Predicted_pIC50', 'Similarity_to_Training']])
                
                # Predicted activity distribution
                if 'Predicted_pIC50' in prediction_results.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(prediction_results['Predicted_pIC50'], bins=20, alpha=0.7)
                    ax.set_xlabel('Predicted pIC50')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predicted Activities')
                    st.pyplot(fig)
                    
                    # Scatter plot of similarity vs predicted activity
                    if 'Similarity_to_Training' in prediction_results.columns and 'Predicted_pIC50' in prediction_results.columns:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        scatter = ax2.scatter(prediction_results['Similarity_to_Training'], 
                                            prediction_results['Predicted_pIC50'], 
                                            alpha=0.6)
                        ax2.set_xlabel('Similarity to Training Set')
                        ax2.set_ylabel('Predicted pIC50')
                        ax2.set_title('Similarity vs Predicted Activity')
                        ax2.axvline(x=0.5, color='red', linestyle='--', label='Similarity Threshold (0.5)')
                        ax2.legend()
                        st.pyplot(fig2)
                    
                    # Identify most/least active compounds
                    most_active = prediction_results.nlargest(5, 'Predicted_pIC50')
                    least_active = prediction_results.nsmallest(5, 'Predicted_pIC50')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Most Active Compounds (Highest pIC50):")
                        st.dataframe(most_active[['SMILES', 'Predicted_pIC50', 'Similarity_to_Training']])
                    with col2:
                        st.write("Least Active Compounds (Lowest pIC50):")
                        st.dataframe(least_active[['SMILES', 'Predicted_pIC50', 'Similarity_to_Training']])
            
            tab_index += 1
        
        # Training Diagnostics Tab
        with analysis_tabs[tab_index]:
            st.subheader("Training Model Diagnostics")
            
            if training_results and 'test_preds' in training_results:
                # Residual analysis
                residuals = (training_results['test_preds']['Pred_pIC50'] - 
                            training_results['test_preds']['Actual_pIC50'])
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                ax[0].hist(residuals, bins=20, alpha=0.7)
                ax[0].axvline(0, color='red', linestyle='--')
                ax[0].set_xlabel('Residuals')
                ax[0].set_ylabel('Frequency')
                ax[0].set_title('Residual Distribution')
                
                ax[1].scatter(training_results['test_preds']['Pred_pIC50'], residuals, alpha=0.6)
                ax[1].axhline(0, color='red', linestyle='--')
                ax[1].set_xlabel('Predicted Values')
                ax[1].set_ylabel('Residuals')
                ax[1].set_title('Residuals vs Predicted')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Normality test
                from scipy.stats import shapiro
                if len(residuals) > 3:  # Shapiro-Wilk requires at least 3 samples
                    stat, p_value = shapiro(residuals)
                    st.write(f"Shapiro-Wilk normality test: p-value = {p_value:.4f}")
                    if p_value > 0.05:
                        st.success("Residuals appear normally distributed")
                    else:
                        st.warning("Residuals may not be normally distributed")
                else:
                    st.warning("Not enough residuals for normality test")
                
                # Actual vs Predicted plot
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.scatter(training_results['test_preds']['Actual_pIC50'], 
                           training_results['test_preds']['Pred_pIC50'], alpha=0.6)
                min_val = min(training_results['test_preds']['Actual_pIC50'].min(), 
                             training_results['test_preds']['Pred_pIC50'].min())
                max_val = max(training_results['test_preds']['Actual_pIC50'].max(), 
                             training_results['test_preds']['Pred_pIC50'].max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax2.set_xlabel('Actual pIC50')
                ax2.set_ylabel('Predicted pIC50')
                ax2.set_title('Actual vs Predicted Values')
                st.pyplot(fig2)
            else:
                st.warning("Training results not available for diagnostics.")
            tab_index += 1
        
        # Feature Importance Tab
        with analysis_tabs[tab_index]:
            st.subheader("Feature Importance")
            
            if trained_model and 'model' in trained_model.named_steps:
                model_step = trained_model.named_steps['model']
                
                if model_supports_interpretation(model_step):
                    fig = create_feature_importance_plot(model_step, feature_names)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Feature importance visualization not available.")
                else:
                    st.info("Feature importance not supported for this model type.")
            else:
                st.warning("Model not available for feature importance analysis.")
            tab_index += 1
        
        # SHAP Analysis Tab (if available)
        if SHAP_AVAILABLE and (tab_index < len(analysis_tabs)) and analysis_tabs[tab_index]._label == "SHAP Analysis":
            with analysis_tabs[tab_index]:
                st.subheader("SHAP Analysis")
                
                if training_results and 'test_preds' in training_results and len(training_data) > 0:
                    if st.button("Calculate SHAP Values", type="secondary", key="calculate_shap_btn"):
                        with st.spinner("Calculating SHAP values (this may take a while)..."):
                            fig = create_shap_analysis(
                                trained_model,
                                training_data,
                                training_results['test_preds']['SMILES'].values,
                                feature_names
                            )
                            
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.warning("SHAP analysis not available for this model type.")
                else:
                    st.warning("Test predictions or training data not available for SHAP analysis.")
            tab_index += 1
        
        # PCA Visualization Tab
        if tab_index < len(analysis_tabs):
            with analysis_tabs[tab_index]:
                st.subheader("PCA Visualization")
                
                if len(training_data) > 0 and training_results and 'y_train' in training_results:
                    # Get feature matrix
                    featurizer = trained_model.named_steps['feats']
                    X_train_transformed = featurizer.transform(training_data)
                    y_train = training_results['y_train']
                    
                    if len(y_train) > 0:
                        fig, variance = create_pca_plot(X_train_transformed, y_train, feature_names)
                        if fig:
                            st.pyplot(fig)
                            st.write(f"Explained variance: PC1 = {variance[0]:.2%}, PC2 = {variance[1]:.2%}")
                    else:
                        st.warning("Training labels not available for PCA visualization.")
                else:
                    st.warning("Training data not available for PCA visualization.")

# -----------------------------
# LEARN TAB - PROFESSIONAL QSAR GUIDE
# -----------------------------
with tabs[4]:
    st.header("QSAR Modeling Educational Resources")
    
    learn_tabs = st.tabs(["QSAR Fundamentals", "Machine Learning", "Molecular Descriptors", "Best Practices"])
    
    with learn_tabs[0]:
        st.subheader("QSAR Fundamentals")
        
        st.markdown("""
        Quantitative Structure-Activity Relationship (QSAR) modeling is a computational methodology that establishes mathematical relationships between chemical structures and biological activities.

        Core Applications:
        - Drug discovery and lead optimization
        - Environmental toxicology assessment
        - Compound property prediction and screening

        Standard QSAR Workflow:
        1. Data Collection: Curated datasets with experimental measurements
        2. Descriptor Calculation: Transformation of structures to numerical features
        3. Model Building: Application of machine learning algorithms
        4. Validation: Rigorous performance evaluation
        5. Prediction: Estimation of activities for novel compounds

        Essential Concepts:
        - pIC50: Negative logarithm of half-maximal inhibitory concentration
        - Applicability Domain: Chemical space where models provide reliable predictions
        - Validation: Critical assessment of model generalizability
        """)
    
    with learn_tabs[1]:
        st.subheader("Machine Learning in QSAR")
        
        st.markdown("""
        Common Algorithm Categories:

        Ensemble Methods:
        - Random Forest: Ensemble of decision trees with robust performance
        - Gradient Boosting: Sequential model improvement with strong predictive power
        - Extra Trees: Highly randomized tree ensemble with computational efficiency

        Linear Methods:
        - Ridge Regression: L2 regularization for improved stability
        - Lasso Regression: L1 regularization with inherent feature selection
        - Elastic Net: Combined L1 and L2 regularization approach

        Performance Evaluation Metrics:
        - R-squared: Proportion of variance explained by the model
        - Root Mean Squared Error: Measure of prediction error magnitude
        - Mean Absolute Error: Average absolute prediction error
        - Q-squared: Cross-validated measure of predictive capability
        """)
    
    with learn_tabs[2]:
        st.subheader("Molecular Descriptors")
        
        st.markdown("""
        Molecular descriptors convert chemical structures into quantitative features for machine learning applications.

        2D Descriptors (Implemented in this platform):
        
        Structural Fingerprints:
        - MACCS Keys: 166 predefined structural patterns
        - Morgan Fingerprints: Circular fingerprints capturing local atomic environments
        - Atom Pair Descriptors: Atomic relationship and distance features
        - Topological Torsion: Molecular torsion pattern descriptors

        Physicochemical Properties:
        - Molecular Weight: Fundamental mass property
        - LogP: Partition coefficient measuring lipophilicity
        - Topological Polar Surface Area: Polar surface area estimation
        - Hydrogen Bond Donors/Acceptors: Hydrogen bonding capacity
        - Rotatable Bond Count: Molecular flexibility indicator

        Note: 3D descriptors require molecular conformation data and are not included in this implementation.
        """)
    
    with learn_tabs[3]:
        st.subheader("QSAR Best Practices")
        
        st.markdown("""
        Data Quality Assurance:
        - Experimental data validation and standardization
        - SMILES string verification and standardization
        - Outlier detection and appropriate handling
        - Consistent unit implementation (nM concentration recommended)

        Model Validation Protocols:
        - Independent test set evaluation
        - Cross-validation performance assessment
        - Applicability domain analysis
        - Overfitting detection and prevention

        Model Interpretation:
        - Feature importance analysis for mechanistic insight
        - SHAP values for prediction explanation
        - Integration of domain expertise with computational results

        Important Limitations:
        - Predictions are reliable only within the model's applicability domain
        - Extrapolation beyond training data carries significant risk
        - Models may capture experimental artifacts rather than true structure-activity relationships
        """)
    
    # Professional resources section
    with st.expander("Additional Resources"):
        st.markdown("""
        Recommended Reading:
        - OECD QSAR Validation Principles
        - Journal of Chemical Information and Modeling
        - Journal of Computer-Aided Molecular Design
        
        Training Materials:
        - QSAR modeling best practices guidelines
        - Molecular descriptor calculation methodologies
        - Machine learning validation techniques
        """)
# -----------------------------
# Settings Tab
# -----------------------------
with tabs[5]:
    st.subheader(" Application Settings")
    
    st.warning("Changing these settings may affect performance and results.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Settings")
        chunk_size = st.slider("Processing chunk size", 100, 5000, CHUNK_SIZE, 100,
                            help="Smaller chunks use less memory but may be slower")
        max_display = st.slider("Max compounds to display", 5, 50, MAX_DISPLAY_COMPOUNDS, 1,
                            help="Limit for molecule visualization")
        max_features = st.slider("Max features to show", 5, 30, MAX_FEATURES_DISPLAY, 1,
                            help="Limit for feature importance display")
    
    with col2:
        st.subheader("Feature Settings")
        default_fingerprints = st.multiselect(
            "Default fingerprint types",
            ["MACCS", "Morgan", "Physicochemical", "Atom Pair", "Topological"],
            default=["MACCS", "Morgan", "Physicochemical"]
        )
        morgan_bits = st.slider("Default Morgan bits", 256, 2048, 1024, 256)
        morgan_radius = st.slider("Default Morgan radius", 1, 4, 2, 1)
    
    st.subheader("Advanced Settings")
    advanced_col1, advanced_col2 = st.columns(2)
    
    with advanced_col1:
        random_seed = st.number_input("Random seed", value=42, min_value=0, max_value=1000,
                                    help="For reproducible results")
        cv_folds = st.slider("Default CV folds", 2, 10, CV_FOLDS, 1,
                            help="Number of cross-validation folds")
    
    with advanced_col2:
        test_size = st.slider("Default test size (%)", 10, 40, 25, 1,
                            help="Percentage of data to use for testing")
        enable_shap = st.checkbox("Enable SHAP analysis", value=True,
                                help="SHAP can be computationally intensive")
    
    if st.button("Save Settings", type="primary",key="save_settings_btn"):
        # Update global constants (in a real app, these would be saved to config)
        CHUNK_SIZE = chunk_size
        MAX_DISPLAY_COMPOUNDS = max_display
        MAX_FEATURES_DISPLAY = max_features
        CV_FOLDS = cv_folds
        
        st.success("Settings saved! Note: some changes may require restarting the application.")
    
    if st.button("Reset to Defaults", type="secondary",key="reset_defaults_btn"):
        st.info("Settings reset to default values.")
    
    st.markdown("---")
    st.subheader("Application Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("Version:2.0")
        st.write("Last Updated:August 2025")
        st.write("RDKit Version:", "Temporarily disabled")
    
    with info_col2:
        st.write("Developer:Bioquantify")
        st.write("License:Educational Use Only")
        st.write("Citation:Please cite this tool if used in teaching")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("Bioquantify  \nAdvanced QSAR Modeling for Education")

with footer_col2:
    st.markdown("Contact  \nindhushree.p1@gmail.com")

with footer_col3:
    st.markdown("Academic Use  \nDeveloped by Indhushree P  \nBioquantify")

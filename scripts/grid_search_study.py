import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (update the file path as needed)
df = pd.read_csv("out/evaluation/tts/iemocap_fishspeech/es/f1_macro.csv")
df = df.drop(['fold', 'denoise', 'total_samples', 'top_p'],axis=1)
# Quick look at the data
print(df.head())

# Convert boolean columns to numeric (if necessary)
if df['denoise'].dtype == bool:
    df['denoise'] = df['denoise'].astype(int)

# Identify categorical variables and convert them into dummy/indicator variables.
categorical_vars = ['embedder_model', 'ser_name']
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# --- Exploratory Data Analysis ---

# Plot correlation matrix for numerical features (including f1_macro)
numeric_features = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_features.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig('corr_matrix.png')

# Print correlation of all features with f1_macro
print("Correlation with f1_macro:")
print(numeric_features.corr()['f1_macro'].sort_values(ascending=False))

# --- Regression Analysis using OLS ---
# Separate features and target variable
X = df.drop("f1_macro", axis=1)
y = df["f1_macro"]

# Add constant term for intercept
X_const = sm.add_constant(X)

# Fit the model
ols_model = sm.OLS(y, X_const).fit()

# Print OLS summary to see coefficients and p-values
print(ols_model.summary())

# --- Feature Importance using Random Forest ---

# Split the data into training and test sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and train the Random Forest regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances from the model
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)
print("Random Forest Feature Importances:")
print(importances_sorted)

# Plot feature importances
plt.figure(figsize=(8, 10))
importances_sorted.plot(kind='bar')
# rotate x axis and increate its font size
plt.xticks(rotation=30, fontsize=10)
plt.ylabel("Importance")
plt.title("Feature Importances from Random Forest")
plt.savefig('grid_search.png')




import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# --- Data Loading and Preprocessing ---

# Load CSV file (replace "your_file.csv" with your actual CSV filepath)
df = pd.read_csv("out/evaluation/tts/iemocap_fishspeech/es/f1_macro.csv")

print(df.groupby(['ser_name']).agg({'f1_macro':'mean'}))

# Just select the rows with the ser_name that has the highest mean f1_macro
ser_name = df.groupby(['ser_name']).agg({'f1_macro':'mean'}).idxmax().iloc[0]
df = df[df['ser_name'] == ser_name]


# --- Step 2: Encode Categorical Variables ---
# Define mappings for categorical variables
embedder_mapping = {'hubert': 0, 'wav2vec2': 1, 'xvector': 2}
denoise_mapping = {False: 0, True: 1}

df['embedder_model_enc'] = df['embedder_model'].map(embedder_mapping)
df['denoise_enc'] = df['denoise'].map(denoise_mapping)

# --- Step 3: Prepare Feature Matrix and Target ---
# We use the hyperparameters that you want to adjust.
features = ['wer_threshold', 'cosine_similarity_threshold', 'max_regenerations',
            'temperature', 'embedder_model_enc', 'denoise_enc']
X = df[features].values
y = df['f1_macro'].values

# --- Step 4: Fit a Gaussian Process Surrogate Model ---
# Define a Matern kernel (you can experiment with other kernels)
kernel = Matern()
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X, y)

# --- Step 5: Define the Bayesian Optimization Search Space ---
# Note: We use the encoded version for categorical variables.
space = [
    Real(0.0, 1.0, name='wer_threshold'),
    Real(0.0, 1.0, name='cosine_similarity_threshold'),
    Integer(1, 20, name='max_regenerations'),
    Real(0.0, 1.0, name='temperature'),
    Integer(0, 2, name='embedder_model_enc'),
    Integer(0, 1, name='denoise_enc')
]

# --- Step 6: Define the Objective Function ---
# We want to maximize f1_macro, so we minimize its negative.
def surrogate(x):
    x = np.array(x).reshape(1, -1)
    pred = gp.predict(x)
    return -pred[0]

@use_named_args(space)
def objective(**params):
    x = [params['wer_threshold'],
         params['cosine_similarity_threshold'],
         params['max_regenerations'],
         params['temperature'],
         params['embedder_model_enc'],
         params['denoise_enc']]
    return surrogate(x)

# --- Step 7: Run Bayesian Optimization ---
res = gp_minimize(objective, space, n_calls=50, random_state=42)

# --- Step 8: Decode and Report the Best Hyperparameters ---
best_enc = res.x
# Decode categorical variables
best_embedder = [k for k, v in embedder_mapping.items() if v == best_enc[4]][0]
best_denoise = True if best_enc[5] == 1 else False

print("Best hyperparameters (encoded):", res.x)
print("Best hyperparameters (decoded):")
print("  wer_threshold:", best_enc[0])
print("  cosine_similarity_threshold:", best_enc[1])
print("  max_regenerations:", best_enc[2])
print("  temperature:", best_enc[3])
print("  embedder_model:", best_embedder)
print("  denoise:", best_denoise)
print("Best predicted f1_macro:", -res.fun)

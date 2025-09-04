
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EtOH_Water_VLE_Predictor:
    """
    Artificial Neural Network for predicting vapor compositions in ethanol-water mixtures
    """
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.trained = False

    def generate_training_data(self, n_points=320):
        """Generate training data from experimental VLE data"""
        # Original experimental data points
        original_data = {
            "xEtoh": [0, 0.019, 0.0721, 0.099, 0.1238, 0.1661, 0.2337, 0.2608, 
                     0.3273, 0.3965, 0.5198, 0.5732, 0.6763, 0.7472, 0.8943, 1],
            "yEtoh": [0, 0.17, 0.3891, 0.4375, 0.4704, 0.5089, 0.5445, 0.558, 
                     0.5826, 0.6122, 0.6599, 0.6841, 0.7385, 0.7815, 0.8943, 1],
            "T_C": [100, 95.5, 89, 86.7, 85.3, 84.1, 82.7, 82.3, 81.5, 80.7, 
                   79.7, 79.3, 78.74, 78.41, 78.15, 78.3],
            "P_kPa": [101.325] * 16
        }
        df_original = pd.DataFrame(original_data)

        # Interpolate to create more training points
        x_interp = np.linspace(0, 1, n_points)
        T_interp = np.interp(x_interp, df_original["xEtoh"], df_original["T_C"])
        y_interp = np.interp(x_interp, df_original["xEtoh"], df_original["yEtoh"])
        P_interp = np.full_like(x_interp, 101.325)

        return pd.DataFrame({
            "xEtoh": x_interp, 
            "T_C": T_interp, 
            "P_kPa": P_interp, 
            "yEtoh": y_interp
        })

    def train(self, data_df=None):
        """Train the ANN model"""
        if data_df is None:
            data_df = self.generate_training_data()

        # Prepare features and target
        X = data_df[["xEtoh", "T_C", "P_kPa"]].values
        y = data_df["yEtoh"].values

        # Scale features and target
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        # Create and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=10000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # Convert back to original scale for R2 calculation
        y_train_orig = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_train_orig = self.scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test_orig = self.scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)

        self.trained = True

        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_iterations': self.model.n_iter_
        }

        print(f"Training completed in {self.model.n_iter_} iterations")
        print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

        return results

    def predict_vapor_composition(self, x_ethanol, temperature, pressure=101.325):
        """Predict vapor mole fraction for given conditions"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        # Validate inputs
        if not (0 <= x_ethanol <= 1):
            raise ValueError("Liquid mole fraction must be between 0 and 1")
        if not (78 <= temperature <= 100):
            raise ValueError("Temperature must be between 78°C and 100°C")
        if pressure != 101.325:
            print("Warning: Model trained only for atmospheric pressure (101.325 kPa)")

        # Prepare input
        X_input = np.array([[x_ethanol, temperature, pressure]])
        X_scaled = self.scaler_X.transform(X_input)

        # Predict
        y_scaled_pred = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()[0]

        # Ensure physical constraints (0 6#8804; y 6#8804; 1)
        return np.clip(y_pred, 0, 1)

    def batch_predict(self, conditions_df):
        """Predict vapor compositions for multiple conditions"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        X = conditions_df[["xEtoh", "T_C", "P_kPa"]].values
        X_scaled = self.scaler_X.transform(X)
        y_scaled_pred = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()

        return np.clip(y_pred, 0, 1)

    def save_model(self, filename_prefix="ethanol_water_vle"):
        """Save trained model and scalers"""
        if not self.trained:
            raise ValueError("Model must be trained before saving")

        joblib.dump(self.model, f"{filename_prefix}_model.joblib")
        joblib.dump(self.scaler_X, f"{filename_prefix}_scaler_X.joblib")
        joblib.dump(self.scaler_y, f"{filename_prefix}_scaler_y.joblib")
        print(f"Model saved as {filename_prefix}_*.joblib")

    def load_model(self, filename_prefix="ethanol_water_vle"):
        """Load trained model and scalers"""
        try:
            self.model = joblib.load(f"{filename_prefix}_model.joblib")
            self.scaler_X = joblib.load(f"{filename_prefix}_scaler_X.joblib")
            self.scaler_y = joblib.load(f"{filename_prefix}_scaler_y.joblib")
            self.trained = True
            print(f"Model loaded from {filename_prefix}_*.joblib")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found with prefix {filename_prefix}")

    def get_azeotrope_prediction(self):
        """Get prediction at azeotrope conditions"""
        azeotrope_x = 0.8943
        azeotrope_T = 78.15
        y_pred = self.predict_vapor_composition(azeotrope_x, azeotrope_T)
        return {
            'liquid_mole_fraction': azeotrope_x,
            'vapor_mole_fraction': y_pred,
            'temperature_C': azeotrope_T,
            'deviation_from_ideal': abs(y_pred - azeotrope_x)
        }

# Training and prediction example
if __name__ == "__main__":
    predictor = EtOH_Water_VLE_Predictor()
    predictor.train()

    # Predict vapor mole fraction at azeotrope
    azeo_pred = predictor.get_azeotrope_prediction()
    print(f"Azeotrope prediction: {azeo_pred}")

    # Sample prediction
    vapor_frac = predictor.predict_vapor_composition(0.5, 80.0)
    print(f"Predicted vapor mole fraction at x=0.5, T=80.0C: {vapor_frac:.4f}")

    # Save model
    predictor.save_model()

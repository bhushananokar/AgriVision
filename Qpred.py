import sys
from pathlib import Path
import numpy as np
import torch
import pennylane as qml
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib
import time
import argparse

# Configure PyTorch to use device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print(f"Using device: {device}")

class QuantumModel(torch.nn.Module):
    """
    Hybrid quantum-classical model with configurable architecture
    """
    def __init__(self, n_qubits, n_layers=2, entanglement='linear'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        
        # Calculate number of parameters for the quantum circuit
        self.num_params = n_qubits * 3 * n_layers
        
        # Classical pre-processing with configurable size
        pre_hidden_size = max(16, n_qubits * 4)
        self.pre_net = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, pre_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(pre_hidden_size, n_qubits),
            torch.nn.Tanh()
        )
        
        # Trainable parameters for the quantum circuit
        self.q_params = torch.nn.Parameter(
            torch.Tensor(self.num_params).uniform_(-0.1, 0.1)
        )
        
        # Classical post-processing with configurable size
        post_hidden_size = max(8, n_qubits * 2)
        self.post_net = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, post_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(post_hidden_size, 1)
        )
        
        # Create the quantum device - try to use faster devices if available
        try:
            self.q_dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"Using lightning.gpu quantum device with {n_qubits} qubits")
        except:
            try:
                self.q_dev = qml.device("lightning.qubit", wires=n_qubits)
                print(f"Using lightning.qubit quantum device with {n_qubits} qubits")
            except:
                self.q_dev = qml.device("default.qubit", wires=n_qubits)
                print(f"Using default.qubit quantum device with {n_qubits} qubits")
        
        # Define the quantum circuit with torch interface
        self.q_circuit = qml.QNode(self.circuit_definition, self.q_dev, interface="torch")
        
    def circuit_definition(self, inputs, weights):
        """Quantum circuit with configurable depth and entanglement"""
        # Data embedding
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            
        # Parameterized quantum layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RX(weights[param_idx], wires=i)
                param_idx += 1
                qml.RY(weights[param_idx], wires=i)
                param_idx += 1
                qml.RZ(weights[param_idx], wires=i)
                param_idx += 1
            
            # Apply entanglement based on selected pattern
            if self.entanglement == 'linear':
                # Linear entanglement (nearest neighbor)
                for i in range(self.n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                    
                # Connect first and last qubit for circular entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits-1, 0])
                    
            elif self.entanglement == 'full':
                # Full entanglement (all-to-all)
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
                        
            elif self.entanglement == 'star':
                # Star topology (central qubit entangled with all)
                center = 0
                for i in range(1, self.n_qubits):
                    qml.CNOT(wires=[center, i])
                
        # Return measurement expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """
        Forward pass through the hybrid model with correct output shape
        """
        batch_size = x.shape[0]
        results = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Pre-processing
            pre_processed = self.pre_net(x[i])
            
            # Get quantum outputs
            q_out_raw = self.q_circuit(pre_processed, self.q_params)
            
            # Ensure q_out is a proper tensor with correct shape
            if isinstance(q_out_raw, list):
                q_out = torch.tensor(q_out_raw, dtype=torch.float32)
            elif isinstance(q_out_raw, np.ndarray):
                q_out = torch.from_numpy(q_out_raw).float()
            else:
                q_out = q_out_raw.float()
            
            # Post-processing
            result = self.post_net(q_out)
            results.append(result)
        
        # Stack results into a batch and ensure shape matches target [batch_size, 1]
        if results:
            stacked_results = torch.cat(results)
            return stacked_results.reshape(batch_size, 1)
        else:
            # Return empty tensor with correct shape if batch is empty
            return torch.zeros((batch_size, 1), device=device)


class QuantumMLTrainer:
    """Class for training QML models using processed pipeline data"""
    
    def __init__(self, base_dir):
        """
        Initialize the QML trainer
        
        Args:
            base_dir: Path to project base directory
        """
        self.base_dir = Path(base_dir)
        self.processed_data_dir = self.base_dir / "Data_Processors" / "processed_timeseries"
        self.engineered_features_dir = self.base_dir / "Data_Processors" / "engineered_features"
        self.fused_data_dir = self.base_dir / "Data_Processors" / "fused_data"
        
        self.output_dir = self.base_dir / "ML_Pipeline" / "trained_qml_models"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "qml_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.target_scaler = StandardScaler()
        
        self.logger.info(f"QML Trainer initialized with base directory: {self.base_dir}")
        
    def load_pipeline_data(self, region_name, target_var, data_source="engineered"):
        """
        Load data from the processing pipeline
        
        Args:
            region_name: Name of the region
            target_var: Target variable for prediction
            data_source: Source of data ("engineered", "fused", or "processed")
            
        Returns:
            Tuple of (features dataframe, target series)
        """
        try:
            # Determine which data to use based on source parameter
            if data_source == "engineered":
                data_path = self.engineered_features_dir / region_name.lower() / "all_features.csv.gz"
                self.logger.info(f"Loading engineered features from {data_path}")
                df = pd.read_csv(data_path)
                
            elif data_source == "fused":
                data_path = self.fused_data_dir / region_name.lower() / "fused_dataset.csv.gz"
                self.logger.info(f"Loading fused data from {data_path}")
                df = pd.read_csv(data_path)
                
            elif data_source == "processed":
                data_path = self.processed_data_dir / region_name.lower() / "processed_timeseries.csv.gz"
                self.logger.info(f"Loading processed timeseries from {data_path}")
                df = pd.read_csv(data_path)
                
            else:
                raise ValueError(f"Unknown data source: {data_source}")
                
            self.logger.info(f"Loaded dataframe with shape: {df.shape}")
            
            # Check if target variable exists
            if target_var not in df.columns:
                self.logger.error(f"Target variable '{target_var}' not found in dataset")
                available_vars = [col for col in df.columns if not col.startswith('time') 
                                 and not col in ['latitude', 'longitude']]
                self.logger.info(f"Available variables: {available_vars[:10]}...")
                raise ValueError(f"Target variable '{target_var}' not found")
                
            # Extract features and target
            # Exclude non-feature columns and derivatives of target
            exclude_cols = ['time', 'latitude', 'longitude', target_var]
            exclude_cols.extend([col for col in df.columns if col.startswith(f"{target_var}_")])
            
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_cols]
            
            # Handle missing values with modern pandas methods
            df_features = df[feature_cols].copy()
            df_features = df_features.ffill()
            df_features = df_features.bfill()
            df_features = df_features.fillna(df_features.mean())
            
            # Extract X and y
            y = df[target_var]
            
            self.logger.info(f"Extracted features (shape: {df_features.shape}) and target: {target_var}")
            
            # Store feature column names
            self.feature_names = feature_cols
            
            return df_features, y
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline data: {str(e)}")
            raise
            
    def prepare_data_for_training(self, X, y, max_qubits=8, test_size=0.2, use_feature_selection=True):
        """
        Prepare data for quantum ML training with feature selection
        
        Args:
            X: Feature dataframe
            y: Target series
            max_qubits: Maximum number of qubits to use
            test_size: Proportion of data for testing
            use_feature_selection: Whether to use feature selection
            
        Returns:
            Tuple of (train tensors, test tensors, number of qubits)
        """
        try:
            n_features = X.shape[1]
            
            # Use feature selection if needed and requested
            if n_features > max_qubits and use_feature_selection:
                self.logger.info(f"Using feature selection to reduce from {n_features} to {max_qubits} features")
                
                # Handle NaN values safely before feature selection
                X_for_selection = X.fillna(0)
                y_for_selection = y.fillna(y.mean())
                
                try:
                    selector = SelectKBest(score_func=f_regression, k=max_qubits)
                    X_selected = selector.fit_transform(X_for_selection, y_for_selection)
                    
                    # Get selected feature names
                    selected_indices = selector.get_support(indices=True)
                    self.feature_names = [self.feature_names[i] for i in selected_indices]
                    
                    # Create new dataframe with selected features
                    X = pd.DataFrame(X_selected, columns=self.feature_names)
                except Exception as selection_error:
                    self.logger.warning(f"Feature selection failed: {str(selection_error)}. Using first {max_qubits} features.")
                    X = X.iloc[:, :max_qubits]
                    self.feature_names = self.feature_names[:max_qubits]
            elif n_features > max_qubits:
                self.logger.info(f"Limiting features from {n_features} to {max_qubits}")
                X = X.iloc[:, :max_qubits]
                self.feature_names = self.feature_names[:max_qubits]
            
            n_qubits = X.shape[1]
            self.logger.info(f"Using {n_qubits} qubits for quantum circuit")
            self.logger.info(f"Selected features: {', '.join(self.feature_names)}")
            
            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values.reshape(-1, 1)
            
            # Scale features to [-π, π] for quantum circuit
            X_scaled = self.feature_scaler.fit_transform(X_array)
            y_scaled = self.target_scaler.fit_transform(y_array)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=test_size, random_state=42
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            self.logger.info(f"Data prepared: train={X_train_tensor.shape}, test={X_test_tensor.shape}")
            
            # Save test data for later visualization
            self.X_test = X_test
            self.y_test = y_test
            
            return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor), n_qubits
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_model(self, train_data, n_qubits, n_layers=2, entanglement='linear',
                   learning_rate=0.005, epochs=100, batch_size=8):
        """
        Train a quantum-classical hybrid model with dimension fixes
        
        Args:
            train_data: Tuple of training and testing tensors
            n_qubits: Number of qubits to use
            n_layers: Number of quantum circuit layers
            entanglement: Entanglement pattern ('linear', 'full', or 'star')
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple of (trained model, training history, evaluation metrics)
        """
        try:
            X_train, y_train, X_test, y_test = train_data
            
            # Create model with specified architecture
            model = QuantumModel(n_qubits=n_qubits, n_layers=n_layers, entanglement=entanglement)
            
            # Optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            loss_fn = torch.nn.MSELoss()
            
            # Early stopping setup
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            best_model_state = None
            
            # Training loop
            train_losses = []
            test_losses = []
            
            self.logger.info(f"Starting training with {n_qubits} qubits, {n_layers} layers, "
                             f"and {entanglement} entanglement")
            start_time = time.time()
            
            for epoch in range(epochs):
                model.train()
                epoch_start = time.time()
                
                # Process in smaller batches for quantum circuit
                permutation = torch.randperm(X_train.size(0))
                total_loss = 0
                
                for i in range(0, X_train.size(0), batch_size):
                    batch_indices = permutation[i:i+batch_size]
                    batch_x, batch_y = X_train[batch_indices], y_train[batch_indices]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    try:
                        y_pred = model(batch_x)
                        
                        # Ensure dimensions match - both should be [batch_size, 1]
                        if y_pred.shape != batch_y.shape:
                            if len(y_pred.shape) == 1:
                                y_pred = y_pred.reshape(-1, 1)
                        
                        loss = loss_fn(y_pred, batch_y)
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_indices)
                    except Exception as batch_error:
                        self.logger.warning(f"Error processing batch at epoch {epoch}: {str(batch_error)}")
                        continue
                
                if X_train.size(0) > 0:
                    avg_train_loss = total_loss / X_train.size(0)
                    train_losses.append(avg_train_loss)
                else:
                    avg_train_loss = float('inf')
                    train_losses.append(avg_train_loss)
                
                # Evaluate on test set
                model.eval()
                try:
                    with torch.no_grad():
                        # Process test data in smaller batches
                        test_total_loss = 0
                        for i in range(0, X_test.size(0), batch_size):
                            end_idx = min(i + batch_size, X_test.size(0))
                            test_batch = X_test[i:end_idx]
                            test_targets = y_test[i:end_idx] 
                            
                            # Ensure test results have correct shape
                            batch_preds = model(test_batch)
                            if batch_preds.shape != test_targets.shape:
                                if len(batch_preds.shape) == 1:
                                    batch_preds = batch_preds.reshape(-1, 1)
                                    
                            test_loss = loss_fn(batch_preds, test_targets)
                            test_total_loss += test_loss.item() * (end_idx - i)
                        
                        if X_test.size(0) > 0:
                            avg_test_loss = test_total_loss / X_test.size(0)
                            test_losses.append(avg_test_loss)
                        else:
                            avg_test_loss = float('inf')
                            test_losses.append(avg_test_loss)
                except Exception as eval_error:
                    self.logger.warning(f"Error during evaluation at epoch {epoch}: {str(eval_error)}")
                    avg_test_loss = float('inf')
                    test_losses.append(avg_test_loss)
                
                # Update learning rate scheduler
                scheduler.step(avg_test_loss)
                
                # Early stopping check
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    self.logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                                     f"Test Loss = {avg_test_loss:.6f}, Time = {epoch_time:.2f}s")
                
                # Check early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Final evaluation
            metrics = self.evaluate_model(model, X_test, y_test, batch_size)
            metrics['training_time'] = total_time
            
            return model, (train_losses, test_losses), metrics
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test, y_test, batch_size=8):
        """
        Evaluate the trained model with proper array reshaping
        
        Args:
            model: Trained model
            X_test: Test features tensor
            y_test: Test target tensor
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            model.eval()
            all_preds = []
            
            with torch.no_grad():
                # Process test data in batches
                for i in range(0, X_test.size(0), batch_size):
                    end_idx = min(i + batch_size, X_test.size(0))
                    test_batch = X_test[i:end_idx]
                    batch_preds = model(test_batch)
                    all_preds.append(batch_preds)
                
                if all_preds:
                    y_pred = torch.cat(all_preds, dim=0)
                else:
                    raise ValueError("No predictions generated during evaluation")
                    
            # Convert to numpy and ensure correct shape
            y_pred_np = y_pred.detach().cpu().numpy()
            
            # Ensure predictions are 2D for sklearn
            if len(y_pred_np.shape) == 1:
                y_pred_np = y_pred_np.reshape(-1, 1)
                
            # Convert y_test to numpy if it's a tensor and ensure 2D
            if isinstance(y_test, torch.Tensor):
                y_test_np = y_test.cpu().numpy()
            else:
                y_test_np = y_test
                
            if len(y_test_np.shape) == 1:
                y_test_np = y_test_np.reshape(-1, 1)
            
            # Inverse transform to original scale
            y_pred_original = self.target_scaler.inverse_transform(y_pred_np)
            y_test_original = self.target_scaler.inverse_transform(y_test_np)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred_original)
            mae = np.mean(np.abs(y_test_original - y_pred_original))
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae)
            }
            
            self.logger.info(f"Model evaluation results:")
            self.logger.info(f"- RMSE: {rmse:.4f}")
            self.logger.info(f"- R²: {r2:.4f}")
            self.logger.info(f"- MAE: {mae:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model, metrics, history, region_name, target_var, config):
        """
        Save the trained model and related artifacts
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            history: Training history
            region_name: Region name
            target_var: Target variable
            config: Dictionary of model configuration
            
        Returns:
            Path to saved model
        """
        try:
            # Create model ID 
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = f"qml_{region_name}_{target_var}_{timestamp}"
            
            # Create model directory
            model_dir = self.output_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_qubits': model.n_qubits,
                'n_layers': model.n_layers,
                'entanglement': model.entanglement
            }, model_dir / "quantum_model.pt")
            
            # Save scalers
            joblib.dump(self.feature_scaler, model_dir / "feature_scaler.joblib")
            joblib.dump(self.target_scaler, model_dir / "target_scaler.joblib")
            
            # Save feature names
            with open(model_dir / "feature_names.json", 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            
            # Save training history
            train_losses, test_losses = history
            history_data = {
                'train_loss': train_losses,
                'test_loss': test_losses
            }
            with open(model_dir / "training_history.json", 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history_data.items()}, f, indent=2)
            
            # Save metrics
            with open(model_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save metadata with configuration
            metadata = {
                'model_id': model_id,
                'region': region_name,
                'target_variable': target_var,
                'created_date': timestamp,
                'architecture': {
                    'model_type': 'quantum_hybrid',
                    'n_qubits': model.n_qubits,
                    'n_layers': model.n_layers,
                    'entanglement': model.entanglement,
                    'n_features': len(self.feature_names)
                },
                'performance': metrics,
                'training_config': {
                    'feature_scaling': 'min_max_to_pi',
                    'target_scaling': 'standard',
                    'quantum_circuit': f'parameterized_rotation_{model.entanglement}_entanglement',
                    'device_used': str(device),
                    'total_training_time': metrics.get('training_time', 0),
                    'learning_rate': config.get('learning_rate', 0.005),
                    'batch_size': config.get('batch_size', 8),
                    'max_epochs': config.get('epochs', 100)
                }
            }
            
            with open(model_dir / "model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Plot and save training curves
            self.plot_training_curves(train_losses, test_losses, metrics, model_dir / "training_curves.png")
            
            # Save additional visualizations
            self.plot_predictions(model, model_dir / "predictions.png")
            
            self.logger.info(f"Model saved to {model_dir}")
            return model_dir
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def plot_training_curves(self, train_losses, test_losses, metrics, output_path):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - Final RMSE: {metrics["rmse"]:.4f}, R²: {metrics["r2"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)

    def plot_predictions(self, model, output_path):
        """Plot actual vs predicted values"""
        try:
            # Create test tensor
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                # Process in smaller batches
                batch_size = 8
                all_preds = []
                for i in range(0, len(X_test_tensor), batch_size):
                    end_idx = min(i + batch_size, len(X_test_tensor))
                    test_batch = X_test_tensor[i:end_idx]
                    batch_preds = model(test_batch)
                    all_preds.append(batch_preds)
                
                if all_preds:
                    y_pred = torch.cat(all_preds, dim=0)
                    
                    # Ensure proper shape
                    if len(y_pred.shape) == 1:
                        y_pred = y_pred.reshape(-1, 1)
                    
                    # Convert to numpy
                    y_pred_np = y_pred.cpu().numpy()
                    
                    # Inverse transform
                    y_pred_original = self.target_scaler.inverse_transform(y_pred_np)
                    y_test_original = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1))
                    
                    # Create scatter plot
                    # Create scatter plot
                    plt.figure(figsize=(10, 8))
                    plt.scatter(y_test_original, y_pred_original, alpha=0.7)
                    
                    # Add perfect prediction line
                    min_val = min(np.min(y_test_original), np.min(y_pred_original))
                    max_val = max(np.max(y_test_original), np.max(y_pred_original))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    
                    # Add regression line
                    m, b = np.polyfit(y_test_original.flatten(), y_pred_original.flatten(), 1)
                    plt.plot(y_test_original, m*y_test_original + b, 'g-', alpha=0.7)
                    
                    # Add labels and title
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.title(f'Quantum ML Predictions\nR² = {r2_score(y_test_original, y_pred_original):.4f}')
                    plt.grid(True, alpha=0.3)
                    
                    # Add text for metrics
                    metrics_text = (
                        f"RMSE: {np.sqrt(mean_squared_error(y_test_original, y_pred_original)):.4f}\n"
                        f"MAE: {np.mean(np.abs(y_test_original - y_pred_original)):.4f}\n"
                        f"R²: {r2_score(y_test_original, y_pred_original):.4f}"
                    )
                    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                            fontsize=12, va='top', bbox=dict(boxstyle='round', alpha=0.5))
                    
                    plt.tight_layout()
                    plt.savefig(output_path)
        except Exception as e:
            self.logger.warning(f"Could not generate prediction plot: {str(e)}")
        
    def train_for_region(self, region_name, target_var, data_source="engineered",
                       max_qubits=8, n_layers=2, entanglement='linear',
                       learning_rate=0.005, epochs=100, batch_size=8):
        """
        Complete training pipeline for a region with configurable architecture
        
        Args:
            region_name: Name of the region
            target_var: Target variable for prediction
            data_source: Source of data ("engineered", "fused", or "processed")
            max_qubits: Maximum number of qubits to use
            n_layers: Number of quantum circuit layers
            entanglement: Entanglement pattern ('linear', 'full', or 'star')
            learning_rate: Learning rate for optimization
            epochs: Training epochs
            batch_size: Training batch size
            
        Returns:
            Path to saved model
        """
        try:
            self.logger.info(f"Starting QML training for {region_name}, target: {target_var}")
            self.logger.info(f"Using data source: {data_source}")
            self.logger.info(f"Model configuration: {max_qubits} qubits, {n_layers} layers, "
                           f"{entanglement} entanglement")
            
            # 1. Load data from pipeline
            X, y = self.load_pipeline_data(region_name, target_var, data_source)
            
            # 2. Prepare data for quantum processing
            train_data, n_qubits = self.prepare_data_for_training(
                X, y, max_qubits, use_feature_selection=True
            )
            
            # 3. Train the model
            model, history, metrics = self.train_model(
                train_data=train_data,
                n_qubits=n_qubits,
                n_layers=n_layers,
                entanglement=entanglement,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # 4. Save the model and artifacts
            config = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'entanglement': entanglement
            }
            model_dir = self.save_model(
                model=model,
                metrics=metrics,
                history=history,
                region_name=region_name,
                target_var=target_var,
                config=config
            )
            
            self.logger.info(f"QML training completed successfully. Model saved to {model_dir}")
            return model_dir
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def main():
    """Main function with command-line arguments for configurable training"""
    parser = argparse.ArgumentParser(description='Train Quantum ML models for climate prediction')
    
    # Data parameters
    parser.add_argument('--region', type=str, default='NorthernCalifornia', 
                      help='Region name to process')
    parser.add_argument('--target', type=str, default='TMAX',
                      help='Target variable to predict')
    parser.add_argument('--data-source', type=str, default='engineered',
                      choices=['engineered', 'fused', 'processed'],
                      help='Source of input data')
    
    # Model architecture
    parser.add_argument('--qubits', type=int, default=8,
                      help='Maximum number of qubits to use')
    parser.add_argument('--layers', type=int, default=2,
                      help='Number of quantum circuit layers')
    parser.add_argument('--entanglement', type=str, default='linear',
                      choices=['linear', 'full', 'star'],
                      help='Entanglement pattern')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.005,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Training batch size')
    parser.add_argument('--multi-target', action='store_true',
                      help='Train models for multiple target variables')
    
    args = parser.parse_args()
    
    try:
        # Get project directory
        project_dir = Path(__file__).parent.parent
        
        print(f"Project directory: {project_dir}")
        print("Looking for processed data in:")
        print(f"- {project_dir / 'Data_Processors' / 'engineered_features'}")
        
        # Initialize trainer
        trainer = QuantumMLTrainer(project_dir)
        
        # Determine target variables
        if args.multi_target:
            target_variables = ["TMAX", "TMIN", "PRCP"]
        else:
            target_variables = [args.target]
        
        # Train models for each target variable
        results = {}
        for target_var in target_variables:
            print(f"\nTraining QML model for {target_var}...")
            print(f"Configuration: {args.qubits} qubits, {args.layers} layers, {args.entanglement} entanglement")
            
            model_dir = trainer.train_for_region(
                region_name=args.region,
                target_var=target_var,
                data_source=args.data_source,
                max_qubits=args.qubits,
                n_layers=args.layers,
                entanglement=args.entanglement,
                learning_rate=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            if model_dir:
                results[target_var] = str(model_dir)
                print(f"Successfully trained model for {target_var}: {model_dir}")
            else:
                print(f"Failed to train model for {target_var}")
        
        # Print summary
        print("\nTraining Summary:")
        for target, path in results.items():
            print(f"- {target}: {path}")
            
    except Exception as e:
        print(f"Fatal error in QML training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

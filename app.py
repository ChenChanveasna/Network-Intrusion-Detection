import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.api.models import load_model
import pickle
import datetime
import os
import matplotlib.pyplot as plt

def change_columns_name(df):
    """
    Rename columns based on predefined mapping
    """
    column_mapping = {
        'Flow ID': 'Flow ID',
        'Src IP': 'Source IP',
        'Src Port': 'Source Port',
        'Dst IP': 'Destination IP',
        'Dst Port': 'Destination Port',
        'Protocol': 'Protocol',
        'Timestamp': 'Timestamp',
        'Flow Duration': 'Flow Duration',
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Bwd IAT Tot': 'Bwd IAT Total',
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Bwd PSH Flags': 'Bwd PSH Flags',
        'Fwd URG Flags': 'Fwd URG Flags',
        'Bwd URG Flags': 'Bwd URG Flags',
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'CWE Flag Count': 'CWE Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count',
        'Down/Up Ratio': 'Down/Up Ratio',
        'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
        'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
        'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
        'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
        'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
        'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
        'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
        'Active Mean': 'Active Mean',
        'Active Std': 'Active Std',
        'Active Max': 'Active Max',
        'Active Min': 'Active Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min',
        'Label': 'Label'
    }
    return df.rename(columns=column_mapping)

def preprocess_imbalanced_model(df):
    """
    Preprocess for imbalanced model
    """
    df = df.replace([-np.inf, np.inf], np.nan)
    df = df.fillna(0)
    #df = df.drop_duplicates()
    
    column_mapping = {
        'Avg Bwd Segment Size': 'Bwd Segment Size Avg',
        'Subflow Bwd Bytes': 'Subflow Bwd Bytes',
        'Init_Win_bytes_backward': 'Bwd Init Win Bytes',
        'Bwd Packet Length Max': 'Bwd Packet Length Max',
        'Destination Port': 'Dst Port',
        'Flow IAT Min': 'Flow IAT Min',
        'Bwd Header Length': 'Bwd Header Length',
        'RST Flag Count': 'RST Flag Count',
        'Bwd Packet Length Mean': 'Bwd Packet Length Mean',
        'Fwd Packets/s': 'Fwd Packets/s'
    }
    df = df.rename(columns=column_mapping)
    
    features = [
        'Bwd Segment Size Avg',
        'Subflow Bwd Bytes',
        'Bwd Init Win Bytes',
        'Bwd Packet Length Max',
        'Dst Port',
        'Flow IAT Min',
        'Bwd Header Length',
        'RST Flag Count',
        'Bwd Packet Length Mean',
        'Fwd Packets/s'
    ]
    
    return df[features]

def preprocess_balanced_model(df):
    """
    Preprocess for balanced model
    """
    df = df.replace([-np.inf, np.inf], np.nan)
    df = df.fillna(0)
    #df = df.drop_duplicates()
    
    column_mapping = {
        'RST Flag Count': 'RST Flag Count',
        'Destination Port': 'Dst Port',
        'Bwd Packet Length Max': 'Bwd Packet Length Max',
        'Bwd Packet Length Mean': 'Bwd Packet Length Mean',
        'Init_Win_bytes_backward': 'Bwd Init Win Bytes',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd Packet Length Min': 'Fwd Packet Length Min',
        'Avg Bwd Segment Size': 'Bwd Segment Size Avg',
        'Subflow Bwd Packets': 'Subflow Bwd Packets',
        'Init_Win_bytes_forward': 'FWD Init Win Bytes'
    }
    df = df.rename(columns=column_mapping)
    
    features = [
        'RST Flag Count',
        'Dst Port',
        'Bwd Packet Length Max',
        'Bwd Packet Length Mean',
        'Bwd Init Win Bytes',
        'Flow IAT Min',
        'Fwd Packet Length Min',
        'Bwd Segment Size Avg',
        'Subflow Bwd Packets',
        'FWD Init Win Bytes'
    ]
    
    return df[features]

class NetworkTrafficClassifier:
    def __init__(self, model_path, scaler_path):
        """
        Initialize the classifier with model and scaler
        """
        try:
            self.model = load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Determine preprocessing function based on model name
            if 'imbalanced' in model_path:
                self.preprocess = preprocess_imbalanced_model
            elif 'balanced' in model_path:
                self.preprocess = preprocess_balanced_model
            else:
                raise ValueError("Model type could not be determined")
        
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")
            raise
    
    def predict(self, df):
        """
        Preprocess and predict labels
        """
        # Preprocess using model-specific function
        X = self.preprocess(df)
        
        # Scale the features
        #X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X)
        return (predictions > 0.25).astype(int).flatten()

def create_log_entry(df, predictions):
    """
    Create a comprehensive log of predictions
    """
    log_columns = ['Flow ID', 'Timestamp', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Predicted Label']
    
    # Ensure we have all required columns
    required_columns = ['Flow ID', 'Timestamp', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port']
    for col in required_columns:
        if col not in df.columns:
            st.warning(f"Warning: {col} column not found in input data.")
            df[col] = 'N/A'
    
    # Create log dataframe
    log_df = df[required_columns].copy()
    log_df['Predicted Label'] = predictions
    
    return log_df[log_columns]

def main():
    st.title('Network Traffic Classification')
    
    # Sidebar for model and scaler upload
    st.sidebar.header('Model Configuration')
    model_file = st.sidebar.file_uploader("Upload H5 Model", type=['h5'])
    scaler_file = st.sidebar.file_uploader("Upload Scaler (pkl)", type=['pkl'])
    
    # Main file upload
    st.header('CSV File Upload')
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Analyze button
    analyze_button = st.button('Analyze')
    
    if uploaded_file and model_file and scaler_file and analyze_button:
        # Save uploaded files temporarily
        file_name = model_file.name
        with open(f'{file_name}_temp_model.h5', 'wb') as f:
            f.write(model_file.getbuffer())
        with open(f'{file_name}_temp_scaler.pkl', 'wb') as f:
            f.write(scaler_file.getbuffer())
        
        try:
            # Read CSV and rename columns
            df = pd.read_csv(uploaded_file)
            df = change_columns_name(df)
            
            # Initialize classifier
            classifier = NetworkTrafficClassifier(f'{file_name}_temp_model.h5', f'{file_name}_temp_scaler.pkl')
            
            # Prediction
            st.subheader('Prediction Results')
            predictions = classifier.predict(df)
            
            # Distribution of predictions
            malicious_count = np.sum(predictions)
            benign_count = len(predictions) - malicious_count
            
            col1, col2, col3= st.columns(3)
            with col1:
                st.metric('Benign Samples', benign_count)
            with col2:
                st.metric('Malicious Samples', malicious_count)
            with col3:
                st.metric('Sample Size', len(predictions))
            
            # Create bar chart to visualize distribution
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create bar chart
            categories = ['Benign', 'Malicious']
            counts = [benign_count, malicious_count]
            colors = ['green', 'red']

            bars = ax.bar(categories, counts, color=colors)

            # Customize the plot
            ax.set_title('Distribution of Prediction Classes', fontsize=15)
            ax.set_ylabel('Number of Samples', fontsize=12)

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}',
                        ha='center', va='bottom', fontsize=10)

            # Add percentage labels
            # total = len(predictions)
            # for i, count in enumerate(counts):
            #     percentage = (count / total) * 100
            #     ax.text(i, count, f'{percentage:.1f}%', 
            #             ha='center', va='bottom', fontsize=10, color='black')

            # Display the plot in Streamlit
            plt.tight_layout()
            st.pyplot(fig)
            # Create log
            log_df = create_log_entry(df, predictions)
            
            # Logging and Export section
            st.subheader('Logging')
            
            # Display log preview
            st.dataframe(log_df.head())
            
            # Export options
            if st.button('Generate Full Log CSV'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f'network_traffic_log_{timestamp}.csv'
                
                # Download button
                csv = log_df.to_csv(index=False)
                st.download_button(
                    label="Download Log CSV",
                    data=csv,
                    file_name=log_filename,
                    mime='text/csv'
                )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary files
            if os.path.exists(f'{file_name}_temp_model.h5'):
                os.remove(f'{file_name}_temp_model.h5')
            if os.path.exists(f'{file_name}_temp_scaler.pkl'):
                os.remove(f'{file_name}_temp_scaler.pkl')

if __name__ == '__main__':
    main()
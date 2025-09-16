import os
import json
import pickle
import joblib
from datetime import datetime
import pandas as pd
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import git
from github import Github
import io
from config import Config

class Utils:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{Config.LOGS_DIR}/utils.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [Config.MODELS_DIR, Config.DATA_DIR, Config.LOGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def save_model(self, model, filename):
        """Save model to file"""
        try:
            filepath = os.path.join(Config.MODELS_DIR, filename)
            
            if filename.endswith('.h5'):
                model.save(filepath)
            elif filename.endswith('.pkl'):
                joblib.dump(model, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                    
            self.logger.info(f"Model saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None
            
    def load_model(self, filename):
        """Load model from file"""
        try:
            filepath = os.path.join(Config.MODELS_DIR, filename)
            
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return None
                
            if filename.endswith('.h5'):
                import tensorflow as tf
                model = tf.keras.models.load_model(filepath)
            elif filename.endswith('.pkl'):
                model = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                    
            self.logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
            
    def sync_with_github(self, files_to_sync=None, commit_message=None):
        """Sync files with GitHub"""
        try:
            # Initialize git repo
            repo_path = os.getcwd()
            
            try:
                repo = git.Repo(repo_path)
            except git.InvalidGitRepositoryError:
                repo = git.Repo.init(repo_path)
                origin = repo.create_remote('origin', f'https://github.com/{Config.GITHUB_REPO}.git')
            
            # Add files
            if files_to_sync:
                for file in files_to_sync:
                    repo.index.add([file])
            else:
                repo.index.add(['*.py', 'models/*', 'data/*'])
                
            # Commit
            if not commit_message:
                commit_message = f"Auto-sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
            repo.index.commit(commit_message)
            
            # Push
            origin = repo.remote('origin')
            origin.push()
            
            self.logger.info(f"Synced with GitHub: {commit_message}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing with GitHub: {str(e)}")
            return False
            
    def sync_with_gdrive(self, files_to_sync=None):
        """Sync files with Google Drive"""
        try:
            # Authenticate
            creds = service_account.Credentials.from_service_account_file(
                'gdrive_credentials.json',
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            service = build('drive', 'v3', credentials=creds)
            
            # Get or create folder
            folder_id = Config.GDRIVE_FOLDER_ID
            
            if not folder_id:
                # Create folder
                folder_metadata = {
                    'name': 'TradingBot',
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                folder_id = folder.get('id')
                
            # Upload files
            if not files_to_sync:
                files_to_sync = []
                for directory in [Config.MODELS_DIR, Config.DATA_DIR]:
                    for file in os.listdir(directory):
                        files_to_sync.append(os.path.join(directory, file))
                        
            for filepath in files_to_sync:
                if os.path.exists(filepath):
                    file_metadata = {
                        'name': os.path.basename(filepath),
                        'parents': [folder_id]
                    }
                    
                    media = MediaFileUpload(
                        filepath,
                        resumable=True
                    )
                    
                    # Check if file exists
                    results = service.files().list(
                        q=f"name='{os.path.basename(filepath)}' and '{folder_id}' in parents",
                        fields="files(id)"
                    ).execute()
                    
                    items = results.get('files', [])
                    
                    if items:
                        # Update existing file
                        file_id = items[0]['id']
                        service.files().update(
                            fileId=file_id,
                            media_body=media
                        ).execute()
                    else:
                        # Create new file
                        service.files().create(
                            body=file_metadata,
                            media_body=media,
                            fields='id'
                        ).execute()
                        
                    self.logger.info(f"Uploaded {filepath} to Google Drive")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing with Google Drive: {str(e)}")
            return False
            
    def download_from_gdrive(self, filename, destination):
        """Download file from Google Drive"""
        try:
            creds = service_account.Credentials.from_service_account_file(
                'gdrive_credentials.json',
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            service = build('drive', 'v3', credentials=creds)
            
            # Search for file
            results = service.files().list(
                q=f"name='{filename}'",
                fields="files(id)"
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                file_id = items[0]['id']
                request = service.files().get_media(fileId=file_id)
                
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    
                fh.seek(0)
                
                with open(destination, 'wb') as f:
                    f.write(fh.read())
                    
                self.logger.info(f"Downloaded {filename} from Google Drive")
                return True
            else:
                self.logger.warning(f"File {filename} not found in Google Drive")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading from Google Drive: {str(e)}")
            return False
            
    def encrypt_credentials(self, credentials):
        """Encrypt sensitive credentials"""
        from cryptography.fernet import Fernet
        
        # Generate key
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        
        # Encrypt credentials
        encrypted_creds = {}
        for key_name, value in credentials.items():
            encrypted_creds[key_name] = cipher_suite.encrypt(value.encode()).decode()
            
        # Save key securely (in production, use a key management service)
        with open('.encryption_key', 'wb') as f:
            f.write(key)
            
        return encrypted_creds
        
    def decrypt_credentials(self, encrypted_creds):
        """Decrypt credentials"""
        from cryptography.fernet import Fernet
        
        # Load key
        with open('.encryption_key', 'rb') as f:
            key = f.read()
            
        cipher_suite = Fernet(key)
        
        # Decrypt credentials
        decrypted_creds = {}
        for key_name, value in encrypted_creds.items():
            decrypted_creds[key_name] = cipher_suite.decrypt(value.encode()).decode()
            
        return decrypted_creds
        
    def generate_report(self, trades, predictions, performance):
        """Generate trading report"""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trades': trades,
                'predictions': predictions,
                'performance': performance
            }
            
            # Save as JSON
            filename = f"{Config.DATA_DIR}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4, default=str)
                
            # Save as CSV
            if trades:
                df_trades = pd.DataFrame(trades)
                df_trades.to_csv(filename.replace('.json', '_trades.csv'), index=False)
                
            if predictions:
                df_predictions = pd.DataFrame(predictions)
                df_predictions.to_csv(filename.replace('.json', '_predictions.csv'), index=False)
                
            self.logger.info(f"Report generated: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None

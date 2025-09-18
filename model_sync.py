import os
import json
import base64
import requests
from datetime import datetime
import joblib
import shutil
import zipfile
from github import Github
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import schedule
import time
import logging
from config import Config

class ModelSyncManager:
    def __init__(self):
        self.setup_logging()
        self.github_client = Github(Config.GITHUB_TOKEN)
        self.repo = self.github_client.get_repo(Config.GITHUB_REPO)
        self.setup_gdrive()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gdrive(self):
        """Configurar Google Drive usando API"""
        try:
            # Para Colab, usar montaje directo
            if 'google.colab' in str(globals().get('__name__')):
                self.gdrive_path = '/content/drive/MyDrive/TradingBot'
                os.makedirs(self.gdrive_path, exist_ok=True)
            else:
                self.gdrive_path = None
        except:
            self.gdrive_path = None
    
    def compress_models(self):
        """Comprimir modelos para reducir tamaño"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_path = f'models_{timestamp}.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk('models'):
                for file in files:
                    if file.endswith(('.h5', '.pkl', '.json')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start='.')
                        zipf.write(file_path, arcname)
        
        return zip_path
    
    def sync_to_github(self, force_push=False):
        """Sincronizar modelos con GitHub usando API"""
        try:
            self.logger.info("Iniciando sincronización con GitHub...")
            
            # Comprimir modelos primero
            zip_path = self.compress_models()
            
            # Leer archivo comprimido
            with open(zip_path, 'rb') as f:
                content = f.read()
            
            # Codificar en base64 para GitHub
            encoded_content = base64.b64encode(content).decode('utf-8')
            
            # Crear path en repo
            file_path = f"models/trained/models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            # Verificar si existe
            try:
                existing_file = self.repo.get_contents(file_path)
                # Actualizar archivo existente
                self.repo.update_file(
                    path=file_path,
                    message=f"Update models - {datetime.now()}",
                    content=encoded_content,
                    sha=existing_file.sha
                )
                self.logger.info(f"Archivo actualizado en GitHub: {file_path}")
            except:
                # Crear nuevo archivo
                self.repo.create_file(
                    path=file_path,
                    message=f"Add models - {datetime.now()}",
                    content=encoded_content
                )
                self.logger.info(f"Archivo creado en GitHub: {file_path}")
            
            # Actualizar archivo de metadatos
            self.update_model_registry()
            
            # Limpiar archivo temporal
            os.remove(zip_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sincronizando con GitHub: {e}")
            return False
    
    def update_model_registry(self):
        """Actualizar registro de modelos en GitHub"""
        try:
            registry = {
                'last_update': datetime.now().isoformat(),
                'models': {},
                'performance': {}
            }
            
            # Listar todos los modelos
            for file in os.listdir('models'):
                if file.endswith(('.h5', '.pkl')):
                    model_info = {
                        'file': file,
                        'size': os.path.getsize(f'models/{file}'),
                        'timestamp': datetime.fromtimestamp(
                            os.path.getmtime(f'models/{file}')
                        ).isoformat()
                    }
                    
                    # Extraer info del nombre
                    parts = file.split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        agent_type = parts[2].replace('.h5', '').replace('.pkl', '')
                        
                        key = f"{symbol}_{timeframe}_{agent_type}"
                        registry['models'][key] = model_info
            
            # Guardar registro localmente
            with open('models/registry.json', 'w') as f:
                json.dump(registry, f, indent=2)
            
            # Subir a GitHub
            content = json.dumps(registry, indent=2)
            encoded = base64.b64encode(content.encode()).decode()
            
            try:
                # Intentar actualizar
                existing = self.repo.get_contents('models/registry.json')
                self.repo.update_file(
                    path='models/registry.json',
                    message='Update model registry',
                    content=encoded,
                    sha=existing.sha
                )
            except:
                # Crear nuevo
                self.repo.create_file(
                    path='models/registry.json',
                    message='Create model registry',
                    content=encoded
                )
            
            self.logger.info("Registro de modelos actualizado")
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando registro: {e}")
            return False
    
    def sync_to_gdrive(self):
        """Sincronizar con Google Drive"""
        try:
            if not self.gdrive_path:
                self.logger.warning("Google Drive no disponible")
                return False
            
            self.logger.info("Sincronizando con Google Drive...")
            
            # Crear carpeta de backup con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.gdrive_path}/backups/{timestamp}"
            os.makedirs(backup_path, exist_ok=True)
            
            # Copiar modelos
            models_dest = f"{backup_path}/models"
            if os.path.exists('models'):
                shutil.copytree('models', models_dest, dirs_exist_ok=True)
            
            # Copiar datos
            data_dest = f"{backup_path}/data"
            if os.path.exists('data'):
                shutil.copytree('data', data_dest, dirs_exist_ok=True)
            
            # Guardar metadata
            metadata = {
                'timestamp': timestamp,
                'models_count': len([f for f in os.listdir('models') if f.endswith(('.h5', '.pkl'))]),
                'total_size_mb': sum(
                    os.path.getsize(f'models/{f}') 
                    for f in os.listdir('models')
                ) / (1024*1024)
            }
            
            with open(f"{backup_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Backup guardado en Drive: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sincronizando con Drive: {e}")
            return False
    
    def download_models_from_github(self):
        """Descargar modelos desde GitHub para Streamlit"""
        try:
            self.logger.info("Descargando modelos desde GitHub...")
            
            # Obtener el archivo de registro
            try:
                registry_content = self.repo.get_contents('models/registry.json')
                registry = json.loads(
                    base64.b64decode(registry_content.content).decode()
                )
            except:
                self.logger.warning("No se encontró registro de modelos")
                return False
            
            # Obtener el último archivo de modelos
            models_dir = self.repo.get_contents('models/trained')
            
            if not models_dir:
                self.logger.warning("No hay modelos en GitHub")
                return False
            
            # Ordenar por fecha y tomar el más reciente
            latest_file = sorted(
                models_dir, 
                key=lambda x: x.name, 
                reverse=True
            )[0]
            
            # Descargar archivo
            content = self.repo.get_contents(latest_file.path)
            decoded = base64.b64decode(content.content)
            
            # Guardar temporalmente
            temp_zip = 'temp_models.zip'
            with open(temp_zip, 'wb') as f:
                f.write(decoded)
            
            # Extraer
            with zipfile.ZipFile(temp_zip, 'r') as zipf:
                zipf.extractall('.')
            
            # Limpiar
            os.remove(temp_zip)
            
            self.logger.info(f"Modelos descargados: {latest_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error descargando modelos: {e}")
            return False
    
    def auto_sync(self, interval_minutes=30):
        """Sincronización automática periódica"""
        def sync_task():
            self.logger.info("Ejecutando sincronización automática...")
            
            # Sincronizar con GitHub
            github_success = self.sync_to_github()
            
            # Sincronizar con Drive
            drive_success = self.sync_to_gdrive()
            
            if github_success and drive_success:
                self.logger.info("✅ Sincronización completa exitosa")
            else:
                self.logger.warning("⚠️ Sincronización parcial")
        
        # Programar tarea
        schedule.every(interval_minutes).minutes.do(sync_task)
        
        # Ejecutar primera vez
        sync_task()
        
        # Loop principal
        while True:
            schedule.run_pending()
            time.sleep(60)

# Singleton para usar en toda la aplicación
sync_manager = ModelSyncManager()

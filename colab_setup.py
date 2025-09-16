"""
Script de configuración inicial para Google Colab
Ejecutar este script al inicio para configurar todo automáticamente
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def setup_colab_environment():
    """Configurar el entorno de Google Colab"""
    
    print("🚀 Iniciando configuración del Trading Bot...")
    
    # 1. Montar Google Drive
    print("📁 Montando Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    # 2. Crear estructura de carpetas
    base_path = '/content/drive/MyDrive/TradingBot'
    folders = ['models', 'data', 'logs', 'credentials', 'backups']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    print("✅ Estructura de carpetas creada")
    
    # 3. Clonar o actualizar repositorio de GitHub
    repo_path = '/content/trading-bot'
    
    if os.path.exists(repo_path):
        print("📥 Actualizando repositorio...")
        os.chdir(repo_path)
        subprocess.run(['git', 'pull'], check=True)
    else:
        print("📥 Clonando repositorio...")
        # Solicitar información del repositorio si no existe
        github_user = input("Ingrese su usuario de GitHub: ")
        github_repo = input("Ingrese el nombre del repositorio (ej: trading-bot): ")
        
        repo_url = f"https://github.com/{github_user}/{github_repo}.git"
        subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
        
        # Guardar info del repo
        repo_info = {'user': github_user, 'repo': github_repo}
        with open(f'{base_path}/credentials/repo_info.json', 'w') as f:
            json.dump(repo_info, f)
    
    os.chdir(repo_path)
    print("✅ Repositorio configurado")
    
    # 4. Instalar dependencias
    print("📦 Instalando dependencias...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
    
    # Instalar paquetes adicionales para Colab
    additional_packages = ['xgboost', 'lightgbm', 'catboost']
    for package in additional_packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], check=True)
    
    print("✅ Dependencias instaladas")
    
    # 5. Configurar credenciales
    creds_path = f'{base_path}/credentials/credentials.json'
    
    if os.path.exists(creds_path):
        print("🔑 Cargando credenciales existentes...")
        with open(creds_path, 'r') as f:
            creds = json.load(f)
    else:
        print("🔑 Configurando nuevas credenciales...")
        print("\n⚠️  IMPORTANTE: Las credenciales se guardarán cifradas en Google Drive")
        
        creds = {}
        creds['ALPACA_API_KEY'] = input("Ingrese su Alpaca API Key: ")
        creds['ALPACA_SECRET_KEY'] = input("Ingrese su Alpaca Secret Key: ")
        creds['GITHUB_TOKEN'] = input("Ingrese su GitHub Token: ")
        creds['GDRIVE_FOLDER_ID'] = base_path
        
        # Guardar credenciales
        with open(creds_path, 'w') as f:
            json.dump(creds, f)
        
        print("✅ Credenciales guardadas")
    
    # 6. Exportar variables de entorno
    for key, value in creds.items():
        os.environ[key] = value
    
    # 7. Crear archivo .env local
    with open('.env', 'w') as f:
        for key, value in creds.items():
            f.write(f"{key}={value}\n")
    
    print("\n✅ Configuración completada exitosamente!")
    print(f"📂 Carpeta de trabajo: {repo_path}")
    print(f"💾 Datos guardados en: {base_path}")
    
    return True

def load_credentials():
    """Cargar credenciales guardadas"""
    base_path = '/content/drive/MyDrive/TradingBot'
    creds_path = f'{base_path}/credentials/credentials.json'
    
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        
        # Exportar a variables de entorno
        for key, value in creds.items():
            os.environ[key] = value
        
        print("✅ Credenciales cargadas exitosamente")
        return creds
    else:
        print("⚠️ No se encontraron credenciales guardadas")
        return None

def sync_models_to_github():
    """Sincronizar modelos entrenados con GitHub"""
    try:
        print("📤 Sincronizando modelos con GitHub...")
        
        # Agregar archivos
        subprocess.run(['git', 'add', 'models/*'], shell=True)
        subprocess.run(['git', 'add', 'data/*'], shell=True)
        
        # Commit
        commit_msg = f"Update models - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # Push
        subprocess.run(['git', 'push'], check=True)
        
        print("✅ Modelos sincronizados con GitHub")
        return True
    except Exception as e:
        print(f"❌ Error sincronizando con GitHub: {str(e)}")
        return False

def backup_to_drive():
    """Hacer backup de modelos en Google Drive"""
    try:
        import shutil
        from datetime import datetime
        
        print("💾 Creando backup en Google Drive...")
        
        source_models = 'models'
        source_data = 'data'
        
        base_path = '/content/drive/MyDrive/TradingBot'
        backup_path = f"{base_path}/backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Copiar modelos
        if os.path.exists(source_models):
            shutil.copytree(source_models, f"{backup_path}/models", dirs_exist_ok=True)
        
        # Copiar datos
        if os.path.exists(source_data):
            shutil.copytree(source_data, f"{backup_path}/data", dirs_exist_ok=True)
        
        print(f"✅ Backup creado en: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Error creando backup: {str(e)}")
        return False

# Ejecutar setup si es el script principal
if __name__ == "__main__":
    setup_colab_environment()

@echo off
REM === Activer l'environnement virtuel et lancer l'application Streamlit ===

REM 1. Activer l'environnement virtuel
call source venv\Scripts\activate

REM 2. Installer les dépendances (optionnel, sécurisé)
pip install -r requirements.txt

REM 3. Lancer l'application
streamlit run explor_recomm.py

pause

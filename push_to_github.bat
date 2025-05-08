@echo off
echo === 📦 Accès au dossier du projet ===
cd my_reco-app

echo === 🧪 Activation de l'environnement virtuel ===
call venv\Scripts\activate

echo === 🗂️ Initialisation Git ===
git init

echo === 🧹 Ajout des fichiers du projet ===
git add api.py requirements.txt README.md data/

echo === 💬 Commit initial ===
git commit -m "🚀 Déploiement initial du projet my_reco-app"

echo === 🌐 Configuration du dépôt distant ===
git branch -M main
git remote remove origin 2>nul
git remote add origin   https://github.com/IALeila75/recommand_api

echo === 🚀 Push vers GitHub ===
git push -u origin main

echo ✅ Push terminé avec succès.
pause

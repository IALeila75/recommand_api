@echo off
echo === Accès au dossier du projet ===
cd my_reco-app

echo === Activation de l'environnement virtuel ===
call venv\Scripts\activate

echo === Initialisation Git ===
git init

echo === Ajout des fichiers du projet ===
git add .

git commit -m "🚀 Déploiement initial du projet my_reco-app"
git branch -M main

REM Ajoute ou remplace l'URL du dépôt distant
git remote remove origin 2>nul
git remote add origin https://github.com/IALEILA75/recommand_api.git

echo === Push vers GitHub ===
git push -u origin main

echo ✅ Push terminé avec succès.
pause

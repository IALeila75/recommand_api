@echo off
echo === Activation de l'environnement virtuel ===
call venv\Scripts\activate

echo === Initialisation Git ===
git init
git add .
git commit -m "🔄 Mise à jour du projet"
git branch -M main
git remote remove origin
git remote add origin https://github.com/TON-UTILISATEUR/recommand_api.git
git push -u origin main

echo ✅ Projet poussé avec succès sur GitHub !
pause

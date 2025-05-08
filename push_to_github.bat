@echo off
title 🚀 Initialisation et push vers GitHub
echo ================================
echo   PUSH VERS GITHUB (MAIN)
echo ================================

REM Supprimer l'ancien dossier .git s'il existe
IF EXIST .git (
    rmdir /s /q .git
    echo 🔁 Ancien historique Git supprimé.
)

REM Initialisation Git
git init
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de git init
    pause
    exit /b
)

REM Ajout des fichiers
git add .
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de git add
    pause
    exit /b
)

REM Commit initial
git commit -m "Initial commit"
IF %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Aucun fichier à committer. Vérifie qu'ils sont bien suivis.
    pause
    exit /b
)

REM Création de la branche main
git branch -M main

REM Lien vers ton dépôt GitHub
git remote add origin https://github.com/IALeila75/recommand_api.git

REM Push avec vérification
git push -u origin main
IF %ERRORLEVEL% EQU 0 (
    echo ✅ Projet pousse avec succes vers GitHub !
) ELSE (
    echo ❌ Le push a echoue. Verifie ton repository ou ta connexion.
)

pause

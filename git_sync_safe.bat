#!/bin/bash
echo "=== Git pull avec rebase ==="
git pull origin main --rebase

echo
echo "=== Ajout des fichiers ==="
git add .

echo
echo "=== Commit (si nécessaire) ==="
git commit -m 'Mise à jour locale avant push' || echo "Aucun commit à faire."

echo
echo "=== Push vers GitHub ==="
git push origin main

echo
echo "✅ Terminé sans forçage destructif."

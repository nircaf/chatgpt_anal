#!/bin/bash
pre-commit run --all-files
read MESSAGE
git pull
git add .
git commit -m "$MESSAGE"
git push
echo "Done"
read -p "Press any key to continue..."

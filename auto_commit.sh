#!/bin/bash
for i in {1..4};do pre-commit run --all-files;done
read MESSAGE
git pull
git add .
git commit -m "$MESSAGE"
git push
echo "Done"
read -p "Press any key to continue..."

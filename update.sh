#!/bin/bash
find . -name '.DS_Store' -type f -delete
git add .                
git commit -a -m "update"
git push             
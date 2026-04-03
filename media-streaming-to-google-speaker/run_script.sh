#!/bin/bash
source /srv/storage/custom-tools/python-venv/bin/activate
python3 /srv/storage/custom-tools/Adhan/play_adhan.py >> /srv/storage/custom-tools/Adhan/logs.txt 2>&1
deactivate
#!/bin/bash
python move_data.py
python create_db.py
python create_dialog_acts.py
python create_delex_data.py
python create_soloist_data.py
cp ./train.soloist.json ../../raw_data/
cp ./valid.soloist.json ../../raw_data/
cp ./test.soloist.json ../../raw_data/
mkdir  ../../data/db
cp -r ./db/* ../../data/db
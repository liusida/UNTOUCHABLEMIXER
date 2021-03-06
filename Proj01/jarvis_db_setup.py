#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3


conn = sqlite3.connect('jarvis.db')
print("Opened database successfully")

conn.execute('''CREATE TABLE IF NOT EXISTS training_data
             (txt TEXT PRIMARY KEY NOT NULL,
             action TEXT NOT NULL);''')
print("Table created successfully")


c = conn.cursor()

#c.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", ('Test', 'test action',))

for row in c.execute('SELECT * FROM training_data'):
    print(row)
    
conn.close()

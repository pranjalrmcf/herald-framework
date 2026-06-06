import sqlite3

conn = sqlite3.connect('data/evaluation.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', tables)
conn.close()
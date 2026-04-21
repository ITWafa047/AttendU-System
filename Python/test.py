import mysql.connector

conn = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "test"
)

print("Connected to database successfully!")
conn.close()
import sqlite3
import hashlib

DB_FILE = "patients.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # PATIENTS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS Patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            FOREIGN KEY(user_id) REFERENCES Users(id)
        )
    """)

    # PREDICTIONS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS Predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            filename TEXT,
            prediction_class TEXT,
            confidence DECIMAL(10,2)
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES Patients(patient_id)
        )
    """)

    conn.commit()
    conn.close()


# ---------------- PASSWORD HASHING ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------- USER FUNCTIONS ----------------
def create_user(email, password):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO Users (email, password) VALUES (?, ?)",
            (email, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False


def authenticate_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id FROM Users WHERE email=? AND password=?",
        (email, hash_password(password))
    )
    user = c.fetchone()
    conn.close()

    if user:
        return user[0]
    return None


# ---------------- PATIENT FUNCTIONS ----------------
def add_patient(user_id, name, age, gender):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO Patients (user_id, name, age, gender) VALUES (?, ?, ?, ?)",
        (user_id, name, age, gender)
    )
    conn.commit()
    conn.close()
    
    




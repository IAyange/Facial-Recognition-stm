import mysql.connector
from mysql.connector import Error
from datetime import datetime
import hashlib
from config import DB_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — password hashing (never store plain text passwords!)
# ─────────────────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Convert a plain password into a secure hash."""
    return hashlib.sha256(password.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_connection():
    """Create and return a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"[DB ERROR] Could not connect to database: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  SETUP — run once to create all tables
# ─────────────────────────────────────────────────────────────────────────────
def setup_database():
    """
    Create all required tables if they don't already exist.
    Call this once when the app starts.
    """
    conn = get_connection()
    if not conn:
        print("[DB ERROR] Setup failed — could not connect.")
        return False

    cursor = conn.cursor()

    try:
        # ── Table 1: users (login accounts) ──────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                username      VARCHAR(50)  UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role          VARCHAR(20)  DEFAULT 'staff',
                created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Table 2: employees (face recognition records) ────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                employee_id   INT          UNIQUE NOT NULL,
                name          VARCHAR(100) NOT NULL,
                registered_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
                is_active     BOOLEAN      DEFAULT TRUE
            )
        """)

        # ── Table 3: auth_logs (every authentication attempt) ────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_logs (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                employee_id   INT          NULL,
                employee_name VARCHAR(100) NULL,
                confidence    FLOAT        NULL,
                status        VARCHAR(20)  NOT NULL,
                attempted_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

        # ── Create a default admin account if none exists ────────────────────
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]

        if count == 0:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s)
            """, ("admin", hash_password("admin123"), "admin"))

            cursor.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s)
            """, ("ian", hash_password("ian2024"), "staff"))

            conn.commit()
            print("[DB] Default accounts created: admin / admin123  and  ian / ian2024")

        print("[DB] Database setup complete ✅")
        return True

    except Error as e:
        print(f"[DB ERROR] Setup error: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  USERS — login / authentication of system users
# ─────────────────────────────────────────────────────────────────────────────
def verify_login(username: str, password: str):
    """
    Check if username + password match a record in the users table.

    Returns:
        dict with user info if valid, None if invalid
    """
    conn = get_connection()
    if not conn:
        return None

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT id, username, role
            FROM users
            WHERE username = %s AND password_hash = %s
        """, (username.lower().strip(), hash_password(password)))

        user = cursor.fetchone()
        return user  # None if not found

    except Error as e:
        print(f"[DB ERROR] Login check failed: {e}")
        return None

    finally:
        cursor.close()
        conn.close()


def add_user(username: str, password: str, role: str = "staff"):
    """Add a new login user account."""
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, %s, %s)
        """, (username.lower().strip(), hash_password(password), role))
        conn.commit()
        print(f"[DB] User '{username}' added successfully.")
        return True

    except Error as e:
        print(f"[DB ERROR] Could not add user: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def get_all_users():
    """Return all user accounts (without passwords)."""
    conn = get_connection()
    if not conn:
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, username, role, created_at FROM users")
        return cursor.fetchall()
    except Error as e:
        print(f"[DB ERROR] Could not fetch users: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  EMPLOYEES — face recognition employee records
# ─────────────────────────────────────────────────────────────────────────────
def save_employee(employee_id: int, name: str):
    """
    Save a new employee record to the database.
    Replaces writing to employee_mapping.txt
    """
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO employees (employee_id, name)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE name = %s
        """, (employee_id, name, name))
        conn.commit()
        print(f"[DB] Employee '{name}' (ID: {employee_id}) saved.")
        return True

    except Error as e:
        print(f"[DB ERROR] Could not save employee: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def get_all_employees():
    """
    Return all active employees as a dict {employee_id: name}.
    Replaces reading from employee_mapping.txt
    """
    conn = get_connection()
    if not conn:
        return {}

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT employee_id, name
            FROM employees
            WHERE is_active = TRUE
        """)
        rows = cursor.fetchall()
        return {row["employee_id"]: row["name"] for row in rows}

    except Error as e:
        print(f"[DB ERROR] Could not fetch employees: {e}")
        return {}

    finally:
        cursor.close()
        conn.close()


def remove_employee(employee_id: int):
    """Soft-delete an employee (marks as inactive instead of deleting)."""
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE employees SET is_active = FALSE
            WHERE employee_id = %s
        """, (employee_id,))
        conn.commit()
        print(f"[DB] Employee ID {employee_id} deactivated.")
        return True

    except Error as e:
        print(f"[DB ERROR] Could not remove employee: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def get_next_employee_id():
    """Get the next available employee ID for registration."""
    conn = get_connection()
    if not conn:
        return 0

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(employee_id) FROM employees")
        result = cursor.fetchone()[0]
        return (result + 1) if result is not None else 0

    except Error as e:
        print(f"[DB ERROR] Could not get next ID: {e}")
        return 0

    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH LOGS — record every authentication attempt
# ─────────────────────────────────────────────────────────────────────────────
def log_auth_attempt(status: str, employee_id=None,
                     employee_name=None, confidence=None):
    """
    Log an authentication attempt to the database.
    Replaces writing to face_recognition.log

    status: 'success' | 'failed' | 'unknown'
    """
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO auth_logs
                (employee_id, employee_name, confidence, status)
            VALUES (%s, %s, %s, %s)
        """, (employee_id, employee_name, confidence, status))
        conn.commit()
        return True

    except Error as e:
        print(f"[DB ERROR] Could not log attempt: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def get_auth_logs(limit: int = 50):
    """Return the most recent authentication logs."""
    conn = get_connection()
    if not conn:
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT employee_name, confidence, status, attempted_at
            FROM auth_logs
            ORDER BY attempted_at DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()

    except Error as e:
        print(f"[DB ERROR] Could not fetch logs: {e}")
        return []

    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK TEST — run this file directly to test the connection
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing database connection...")
    conn = get_connection()
    if conn:
        print("✅ Connected to MySQL successfully!")
        conn.close()
        print("Setting up tables...")
        setup_database()
    else:
        print("❌ Connection failed. Check config.py and make sure XAMPP MySQL is running.")
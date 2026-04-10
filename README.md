# Employee Face Recognition System

A desktop application for employee facial recognition and authentication built with Python, OpenCV, and Tkinter.

## Features

- **User Authentication**: Secure login with username/password
- **Employee Registration**: Capture 20 face samples per employee via webcam
- **Face Recognition**: LBPH (Local Binary Patterns Histograms) algorithm for recognition
- **Authentication Logging**: Complete audit trail of all login attempts
- **Dual Interface**: Both CLI and GUI available

## Requirements

- Python 3.8+
- MySQL Server (XAMPP recommended)
- OpenCV (`opencv-python`)
- PIL (Pillow)
- mysql-connector-python

Install dependencies:
```bash
pip install opencv-python pillow mysql-connector-python
```

## Database Setup

1. Start XAMPP MySQL server
2. Update `config.py` with your database credentials
3. Run `python database.py` to create tables and default users

Default login accounts:
- Username: `admin`, Password: `admin123`
- Username: `ian`, Password: `ian2024`

## Project Structure

```
facialrecognition/
├── main.py                    # CLI entry point
├── login.py                  # Login GUI
├── face_recognition_gui.py   # Main GUI application
├── face_recognition_system.py # Core face recognition logic
├── database.py              # MySQL operations
├── config.py               # Database configuration
├── employee_faces/         # Training data directory
│   ├── employee_model.yml   # Trained model
│   └── training/          # Face images
└── README.md
```

## Usage

### GUI Mode (Recommended)
```bash
python login.py
```

### CLI Mode
```bash
python main.py
```

## GUI Operations

1. **Login**: Enter credentials to access the system
2. **Register New Employee**: Enter name and capture 20 face samples
3. **Train Model**: Train/retrain the recognition model
4. **Authenticate Employee**: Scan face to verify identity
5. **View Auth Logs**: View recent authentication history

## Configuration

Edit `config.py` to modify database settings:
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",      # Your MySQL password
    "database": "facerecognition"
}
```

## Notes

- Ensure good lighting during registration and authentication
- The system captures 20 face samples during registration
- Model retrains automatically after new employee registration
- Authentication timeout is 10 seconds
- All authentication attempts are logged to the database
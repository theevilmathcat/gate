# GATE - Facial Recognition Access Control System

GATE (Geolocation and Attendance Tracking with Ease) is a web application that provides facial recognition-based access control and employee management. It's built using Python, Flask, Docker, PostgreSQL, and TensorFlow, with a responsive front-end using HTML, CSS, and JavaScript.

## Features

- User authentication (local and OAuth with Gmail)
- Employee management (add, remove, view)
- Facial recognition model training
- Real-time employee recognition
- Responsive web interface

## Tech Stack

- Backend: Python, Flask
- Database: PostgreSQL
- ORM: SQLAlchemy
- Authentication: Flask-Bcrypt, Authlib
- Machine Learning: TensorFlow, OpenCV
- Frontend: HTML, CSS, JavaScript
- Containerization: Docker, Docker Compose

## Setup and Installation

1. Clone the repository
2. Install Docker and Docker Compose
3. Create a `.env` file with the necessary environment variables (see below)
4. Run `docker-compose up --build` to start the application

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://username:password@db/dbname
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret
BASE_DIR=/app/model_inputs
MODEL_PATH=/app/model_outputs
```

## Usage

1. Register a new account or log in with existing credentials
2. Add employees and their photos
3. Train the facial recognition model
4. Use the "Recognize Employee" feature to test the system

## API Endpoints

- `/add_employee`: Add a new employee
- `/api/capture_photos`: Capture and save employee photos
- `/train_model`: Train the facial recognition model
- `/api/recognize`: Recognize an employee from a photo
- `/remove_employee`: Remove an employee from the system
- `/clear_model`: Clear the trained model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
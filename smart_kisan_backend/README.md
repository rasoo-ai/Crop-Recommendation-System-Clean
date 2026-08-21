# Smart Kisan Backend

## Run
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

## API Docs
http://localhost:8000/docs

## Endpoints
- POST /register — create account
- POST /login    — get token
- POST /predict  — crop prediction (saved)
- POST /predict/guest — predict without login
- GET  /history  — prediction history
- GET  /me       — my profile
- GET  /farm     — farm profile
- GET  /stats    — app statistics

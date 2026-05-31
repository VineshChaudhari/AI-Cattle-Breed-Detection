# AI Cattle Breed Identification System
Built solo by Vinesh Chaudhari | B.Tech EE, 2nd Year

## What it does
Identifies cattle breeds in real-time from images using 
computer vision. Built to reduce manual errors during 
livestock registration by Field Level Workers (FLWs) 
and farmers. Integrated as a decision-support tool 
compatible with government platforms like BPA.

## Tech Stack
- YOLOv8 — object detection model
- Roboflow — image labelling & dataset management
- Python — model training & optimization
- Web App — upload image, get breed prediction

## Dataset
- 1600+ images across 9 cattle breeds
- Sourced from Kaggle, labelled using Roboflow

## Results
- 70–95% accuracy across 9 breeds
- Real-time breed prediction via web interface

## How to Run
1. Clone the repo
2. Install dependencies — `pip install -r requirements.txt`
3. Run the web app — `python app.py`
4. Upload a cattle image and get breed prediction

## Future Improvements
- Mobile app integration
- Camera-based real-time detection
- Expanding to more breeds

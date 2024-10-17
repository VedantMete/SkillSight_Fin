

from __future__ import print_function # In python 2.7
import numpy as np
import pickle
import os
from flask import Flask,redirect,url_for,render_template,request,jsonify
from flask import Response
import time
import os
# !python -m spacy download en_core_web_md
import spacy
nlp = spacy.load("en_core_web_md")
import cv2
import subprocess
from deepface import DeepFace
import pandas as pd
import json
import sys
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
import speech_recognition as speechrecognizer
import speech_recognition as sr
from flask import send_from_directory
import math
from pydub import AudioSegment
import re
#import fitz 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
df = pd.read_csv('data/normalized_dataset.csv')
similarities = []
Questions_Arr = []
Correct_Answer_Arr = []
User_Answers = []
All_Video_Details = []
All_Text_Details = []

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ogg', 'wav', 'mp3', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




camera = cv2.VideoCapture(0)
emotion_counts = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0,
    'no_face': 0,
}
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




class ResumeScanner:
    def __init__(self, resume_text):
        self.resume_text = resume_text
        self.required_skills = {"Python", "Java", "SQL", "Machine Learning", "Communication", "Finance", "Business Development", "Critical Thinking", "Research", "Data Mining", "Project Management"}
        self.required_certifications = {"AWS Certified Developer", "Microsoft Certified: Azure AI Engineer Associate", "Finance", "Web3"}
        self.required_test_scores = {"GRE": 320, "TOEFL": 110}
        self.job_fit_score = 0

    def scan_experience(self):
        # Search for experience using regular expression
        experience_match = re.search(r'(\d+)\s*years? experience', self.resume_text, re.IGNORECASE)
        if experience_match:
            experience = int(experience_match.group(1))
            # Increase score for more experience, cap at 5 years
            self.job_fit_score += min(experience, 5) * 10

    def scan_certifications(self):
        # Search for certifications
        for cert in self.required_certifications:
            if cert.lower() in self.resume_text.lower():
                self.job_fit_score += 20

    def scan_test_scores(self):
        # Search for test scores
        for test, score in self.required_test_scores.items():
            score_match = re.search(rf'{test}:?\s*(\d+)', self.resume_text, re.IGNORECASE)
            if score_match:
                test_score = int(score_match.group(1))
                if test_score >= score:
                    self.job_fit_score += 15

    def scan_skills(self):
        # Search for skills
        for skill in self.required_skills:
            if skill.lower() in self.resume_text.lower():
                self.job_fit_score += 25

    def calculate_job_fit_score(self):
        self.scan_experience()
        self.scan_certifications()
        self.scan_test_scores()
        self.scan_skills()
        # Limit the score to 100
        self.job_fit_score = min(self.job_fit_score, 100)
        return self.job_fit_score

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'resume' not in request.files:
            return redirect(request.url)
        file = request.files['resume']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Calculate job fit score
            resume_text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            scanner = ResumeScanner(resume_text)
            job_fit_score = scanner.calculate_job_fit_score()
            return redirect(url_for('job_fit_score', filename=filename, job_fit_score=job_fit_score))
    return render_template('upload.html')

@app.route('/job_fit_score/<filename>/<int:job_fit_score>')
def job_fit_score(filename, job_fit_score):
    return render_template('job_fit_score.html', filename=filename, job_fit_score=job_fit_score)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text








def sentences_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    similarity_score = similarity_matrix[0][0]
    return similarity_score
    
        

@app.route('/')
def index():
    """
    Render the main page.
    """
    global similarities
    global Questions_Arr
    global Correct_Answer_Arr
    global User_Answers
    global All_Text_Details

    similarities = []
    Questions_Arr = []
    Correct_Answer_Arr = []
    User_Answers = []
    All_Text_Details = []

    global emotion_counts
    global All_Video_Details


    All_Video_Details = []
    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0,
        'no_face': 0,
    } 
    return render_template('Main_page.html')

@app.route('/home')
def home():
    global similarities
    global Questions_Arr
    global Correct_Answer_Arr
    global User_Answers
    global All_Text_Details

    similarities = []
    Questions_Arr = []
    Correct_Answer_Arr = []
    User_Answers = []
    All_Text_Details = []

    global emotion_counts
    global All_Video_Details


    All_Video_Details = []
    emotion_counts = {
        'angry': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0,
        'no_face': 0,
    } 
    return render_template('Main_page.html')

@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('Main_page.html')

@app.route('/resume')
def resume():
    """
    Render the about page.
    """
    return render_template('resume.html')

@app.route('/video_demo')
def video_demo():
    """
    Render the demo page.
    """
    return render_template('Video_Demo.html')

@app.route('/text_test_instructions')
def text_test_instructions():
    """
    Render the instructions for the text test.
    """
    return render_template('Instructions_text.html')

@app.route('/video_test_instructions')
def video_test_instructions():
    """
    Render the instructions for the video test.
    """
    return render_template('Instructions_video.html')

@app.route('/Text_Test')
def Text_Test():
    """
    Render the text test page.
    """
    return render_template('Text_Test.html')

@app.route('/Text_Test_Results')
def Text_Test_Results():
    """
    Render the text test results page.
    """
    return render_template('Text_Test_Results.html')

@app.route('/Video_Test')
def Video_Test():
    """
    Render the video test page.
    """
    return render_template('Video_Test.html')

@app.route('/Video_Test_Results')
def Video_Test_Results():
    """
    Render the video test results page.
    """
    return render_template('Video_Test_Results.html')

    
@app.route('/favicon.ico')
def favicon():
    """
    Serve the favicon.
    """
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/Questions')
def Text_Questions():
    """
    Get random questions from the dataset.
    """
    global Questions_Arr
    global Correct_Answer_Arr
    random_rows = df.sample(n=10)
    Questions_Arr = random_rows['Questions'].tolist()
    Correct_Answer_Arr = random_rows['Answers'].tolist()
    return Questions_Arr

@app.route('/Text_Answers/<int:Qindex>', methods=['POST'])
def text_answers(Qindex):
    """
    Receive and process user answers for text questions.
    """
    global All_Text_Details
    global Questions_Arr
    global Correct_Answer_Arr

    answer = request.data.decode('utf-8')
    temp_list = []
    temp_list.append(Questions_Arr[Qindex])
    temp_list.append(Correct_Answer_Arr[Qindex])
    temp_list.append(answer)
    temp_list.append(sentences_similarity(str(answer),str(Correct_Answer_Arr[Qindex])))
    All_Text_Details.append(temp_list)
    return jsonify(answer)

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-audio/<int:Qindex>', methods=['POST'])
def upload_audio(Qindex):
    """
    Handle audio file upload and process it.
    """
    AudioSegment.ffmpeg
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert MP3 to WAV
        wav_file_path = os.path.splitext(file_path)[0] + '.wav'
        AudioSegment.from_file(file_path).export(wav_file_path, format="wav")

        # Load WAV file
        audio_data, sample_rate = librosa.load(wav_file_path)

        # Remove the uploaded files
       #woh remove statements add karna 

        # Speech recognition
        r = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = r.record(source)
        
        text = r.recognize_google(audio_data)

        # Store information
        global Questions_Arr
        global Correct_Answer_Arr
        global All_Video_Details
        similarity = sentences_similarity(text, Correct_Answer_Arr[Qindex])
        temp_list = [Questions_Arr[Qindex], Correct_Answer_Arr[Qindex], text, similarity]
        All_Video_Details.append(temp_list)
        
        return jsonify({'success': text}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video_results')
def video_results():
    temp_Emotion_Counts = emotion_counts
    temp_Results = All_Video_Details

    temp_Emotion_Counts['angry'] = emotion_counts['angry'] * 0.2
    temp_Emotion_Counts['disgust'] = emotion_counts['disgust'] * 0.2
    temp_Emotion_Counts['fear'] = emotion_counts['fear'] * 0.2
    temp_Emotion_Counts['happy'] = emotion_counts['happy'] * 1.3
    temp_Emotion_Counts['sad'] = emotion_counts['sad'] * 0.2
    temp_Emotion_Counts['neutral'] = emotion_counts['neutral'] * 1
    temp_Emotion_Counts['no_face'] = 0

    new_temp = [] 
    new_temp.append(temp_Results)
    new_temp.append(list(temp_Emotion_Counts.values()))
    return jsonify(new_temp)


@app.route('/text_results')
def text_results():
    # All_Text_Details_temp = []
    All_Text_Details_temp = All_Text_Details    
    return jsonify(All_Text_Details_temp)

def generate_frames():
    global camera
    while True:
        if camera is not None:
            success, frame = camera.read()
            if not success:
                break
        else:
            break
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_copy = frame
        frame = buffer.tobytes()
 
        # convert to grayscale
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

        # detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # loop over faces
        for (x, y, w, h) in faces:
            # extract face
            face = frame_copy[y:y+h, x:x+w]
            # recognize emotion if a face is detected
            if len(face) > 0:
                try:
                    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=True)
                    if result[0]['dominant_emotion'] is not None:
                        emotion_counts[result[0]['dominant_emotion']] += 1
                    else:
                        emotion_counts['no_emotion'] += 1
                except ValueError as err:
                    emotion_counts['no_face'] += 1
            # update the no_face count if no face is detected
            else:
                emotion_counts['no_face'] += 1

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/videofeed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return 'Camera started'

@app.route('/start_again')
def start_again():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    while not camera.isOpened():  # Wait until the camera is ready
        pass
    return 'Camera started'

@app.route('/stop')
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        camera = cv2.VideoCapture(0)
    return 'Camera stopped'

if __name__=='__main__':
    app.run(debug=True,port=7070,host='0.0.0.0')

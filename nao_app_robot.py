import sys
import os
import cv2
import json
import time
import socket
import struct
import threading
import subprocess
import numpy as np
import re
import unicodedata

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_cascade_path():
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'cv2', 'data', 'haarcascade_frontalface_default.xml')

def get_yolo_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'yolo11n.pt')
    return 'yolo11n.pt'

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
    QTabWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

# ===================== STYLE =====================
STYLE = """
QMainWindow, QWidget {
    background-color: #060a14;
    color: #c0d4e8;
    font-family: 'Consolas', 'Courier New', monospace;
}
QTabWidget::pane { border: 1px solid #1a2a4a; background: #060a14; }
QTabBar::tab {
    background: #0a0f1e; color: #6a8aaa;
    padding: 8px 20px; border: none;
    font-size: 11px; letter-spacing: 2px; font-family: 'Consolas', monospace;
}
QTabBar::tab:selected { color: #4fc3f7; border-bottom: 2px solid #4fc3f7; background: #0d1428; }
QTabBar::tab:hover { color: #c0d4e8; }
QPushButton {
    background: rgba(79,195,247,0.1); border: 1px solid rgba(79,195,247,0.3);
    border-radius: 3px; color: #4fc3f7; padding: 6px 14px;
    font-size: 11px; letter-spacing: 1px; font-family: 'Consolas', monospace;
}
QPushButton:hover { background: rgba(79,195,247,0.2); border-color: #4fc3f7; }
QPushButton:pressed { background: rgba(79,195,247,0.3); }
QPushButton#btn-red { color: #f44336; border-color: rgba(244,67,54,0.3); background: rgba(244,67,54,0.05); }
QPushButton#btn-red:hover { background: rgba(244,67,54,0.15); border-color: #f44336; }
QPushButton#btn-green { color: #4caf50; border-color: rgba(76,175,80,0.3); background: rgba(76,175,80,0.05); }
QPushButton#btn-green:hover { background: rgba(76,175,80,0.15); border-color: #4caf50; }
QLineEdit {
    background: #060a14; border: 1px solid #1a2a4a; border-radius: 3px;
    color: #c0d4e8; padding: 6px 10px; font-size: 12px; font-family: 'Consolas', monospace;
}
QLineEdit:focus { border-color: #4fc3f7; }
QTextEdit {
    background: #060a14; border: none; color: #c0d4e8;
    font-size: 11px; font-family: 'Consolas', monospace;
}
QScrollBar:vertical { background: #060a14; width: 4px; border: none; }
QScrollBar::handle:vertical { background: #2a3a5a; border-radius: 2px; }
QFrame#card { background: #0a0f1e; border: 1px solid #1a2a4a; border-radius: 4px; }
"""

CYAN = "#4fc3f7"
GREEN = "#4caf50"
ORANGE = "#ff9800"
RED = "#f44336"
BG = "#060a14"
BG2 = "#0a0f1e"
BG3 = "#0d1428"
BORDER = "#1a2a4a"

FOCALE = 600
TAILLES_REELLES = {
    "person":0.45,"bottle":0.07,"cup":0.08,"chair":0.50,
    "book":0.20,"laptop":0.35,"cell phone":0.07,"keyboard":0.45,
    "mouse":0.073,"pen":0.01,"vase":0.10,"teddy bear":0.20,
    "tv":1.27,"monitor":0.50,"bowl":0.15,"apple":0.08,
    "banana":0.20,"orange":0.08,"scissors":0.10,"remote":0.05,
    "clock":0.30,"umbrella":0.60,"bag":0.35,"backpack":0.35,
    "sports ball":0.133,"wine glass":0.08,"fork":0.02,
    "knife":0.02,"spoon":0.02,"sandwich":0.15,"table":1.20,
}

TRADUCTIONS = {
    "person":"personne","bottle":"bouteille","cup":"tasse","chair":"chaise",
    "book":"livre","laptop":"ordinateur","cell phone": "téléphone","keyboard":"clavier",
    "mouse":"souris","pen":"stylo","bag":"sac","apple":"pomme","banana":"banane",
    "tv":"television","monitor":"ecran","remote":"telecommande","clock":"horloge",
    "vase":"vase","teddy bear":"peluche","sports ball":"balle","scissors":"ciseaux",
    "bowl":"bol","umbrella":"parapluie","backpack":"sac a dos","sandwich":"sandwich",
    "fork":"fourchette","knife":"couteau","spoon":"cuillere","table":"table",
    "wine glass":"verre","sports ball":"balle",
}

nao_port   = 9561
nao_ip     = "10.126.205.184"  # IP du robot NAO
cam_port   = 9562              # port stream camera
audio_port = 9563              # port stream audio
current_angles = {}
visages_detectes = 0
objets_detectes_count = 0
serveur_process = None
imitation_active = False
patrouille_active = False
historique_mistral = []

# Memoire des personnes (LBPH)
recognizer        = cv2.face.LBPHFaceRecognizer_create()
memoire_personnes = {}
prochain_id       = 0
modele_entraine   = False
dernier_prenom_vu = ""

def envoyer_commande(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((nao_ip, nao_port))
        s.send(json.dumps(cmd).encode("utf-8"))
        reponse = s.recv(4096)
        s.close()
        try:
            data = json.loads(reponse.decode("utf-8"))
            if "angles" in data and data["angles"]:
                global current_angles
                current_angles = data["angles"]
            return data.get("status") == "ok"
        except:
            return reponse == b'ok'
    except:
        return False

def recv_exact(sock, n):
    """Recevoir exactement n octets depuis un socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise Exception("Connexion fermee")
        data += chunk
    return data

# ===================== THREAD VIDEO (camera NAO) =====================
class VideoThread(QThread):
    frame_signal  = pyqtSignal(QImage)
    stats_signal  = pyqtSignal(int, int)
    prenom_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.detection_visages = True
        self.detection_objets  = True
        self.objets_detectes   = {}
        self.capture_pour_apprentissage = False
        self.faces_apprentissage = []
        self.ids_apprentissage   = []
        self.id_apprentissage    = 0
        self.imitation_thread_ref = None
        self.last_frame           = None

    def run(self):
        global visages_detectes, objets_detectes_count, dernier_prenom_vu

        try:
            from ultralytics import YOLO
            yolo = YOLO("yolo11n.pt")
        except:
            yolo = None

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Connexion au stream camera du robot
        cam_sock = None
        while self.running:
            try:
                cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cam_sock.settimeout(5)
                cam_sock.connect((nao_ip, cam_port))
                print("Stream camera NAO connecte !")
                break
            except:
                print("Attente stream camera...")
                time.sleep(2)

        while self.running:
            try:
                # Recevoir taille puis JPEG
                taille_bytes = recv_exact(cam_sock, 4)
                taille       = struct.unpack(">I", taille_bytes)[0]
                jpeg_data    = recv_exact(cam_sock, taille)

                # Decoder le JPEG en frame OpenCV
                import numpy as np_
                arr   = np_.frombuffer(jpeg_data, dtype=np_.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                nb_visages = 0
                nb_objets  = 0
                hauteur, largeur = frame.shape[:2]
                nouveaux_objets  = {}

                # Partager avec ImitationThread et FaceTrackThread
                if self.imitation_thread_ref is not None:
                    self.imitation_thread_ref.frame = frame.copy()
                self.last_frame = frame.copy()

                if self.detection_visages:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # CLAHE : meilleure egalisation locale pour camera de mauvaise qualite
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray  = clahe.apply(gray)

                    # Parametres très assouplis pour camera NAO
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=4,
                        minSize=(20, 20),
                        maxSize=(400, 400)
                    )
                    nb_visages = len(faces)

                    if self.capture_pour_apprentissage:
                        for (x, y, w, h) in faces:
                            roi = gray[y:y+h, x:x+w]
                            roi = cv2.resize(roi, (200, 200))
                            self.faces_apprentissage.append(roi)
                            self.ids_apprentissage.append(self.id_apprentissage)

                    for (x, y, w, h) in faces:
                        nom     = "personne"
                        couleur = (0, 255, 100)
                        if modele_entraine and len(memoire_personnes) > 0:
                            try:
                                roi = gray[y:y+h, x:x+w]
                                roi = cv2.resize(roi, (200, 200))
                                id_predit, confiance = recognizer.predict(roi)
                                if confiance < 70:  # seuil plus tolerant pour camera NAO
                                    nom     = memoire_personnes.get(id_predit, "personne")
                                    couleur = (0, 200, 255)
                                    if nom != dernier_prenom_vu:
                                        dernier_prenom_vu = nom
                                        self.prenom_signal.emit(nom)
                            except:
                                pass
                        cv2.rectangle(frame, (x, y), (x+w, y+h), couleur, 2)
                        cv2.putText(frame, nom, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

                if yolo and self.detection_objets:
                    results = yolo(frame, verbose=False)
                    for res in results:
                        for box in res.boxes:
                            if float(box.conf) > 0.5:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                classe_en = yolo.names[int(box.cls)]
                                nom_fr    = TRADUCTIONS.get(classe_en, classe_en)
                                nb_objets += 1
                                w_boite   = x2 - x1
                                lr        = TAILLES_REELLES.get(classe_en, 0.20)
                                distance  = round((lr * FOCALE) / w_boite, 2) if w_boite > 0 else 1.0
                                centre_x  = (x1 + x2) / 2 / largeur
                                nouveaux_objets[classe_en] = {"distance": distance, "centre_x": centre_x, "nom_fr": nom_fr}
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (79, 195, 247), 2)
                                cv2.putText(frame, f"{nom_fr} {distance}m", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 195, 247), 1)

                self.objets_detectes  = nouveaux_objets
                visages_detectes      = nb_visages
                objets_detectes_count = nb_objets
                self.stats_signal.emit(nb_visages, nb_objets)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_signal.emit(img)

            except Exception as e:
                print("Erreur reception camera : " + str(e))
                # Reconnexion
                time.sleep(1)
                try:
                    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    cam_sock.settimeout(5)
                    cam_sock.connect((nao_ip, cam_port))
                except:
                    pass

    def stop(self):
        self.running = False

# ===================== THREAD MICRO =====================
class MicroThread(QThread):
    texte_signal = pyqtSignal(str)
    commande_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str, str)
    chat_signal = pyqtSignal(str, str)

    def __init__(self, api_key, device_index=2):
        super().__init__()
        self.running = True
        self.api_key = api_key
        self.device_index = device_index

    def comprendre(self, texte):
        global historique_mistral
        from mistralai import Mistral
        client = Mistral(api_key=self.api_key)
        historique_mistral.append({"role": "user", "content": texte})
        if len(historique_mistral) > 20:
            historique_mistral = historique_mistral[-20:]
        system = """Tu es un assistant intelligent qui controle le robot NAO.
Tu DOIS repondre avec UN SEUL JSON brut sans aucun texte avant ou apres.
Si c'est une commande robot -> utilise les actions. Si c'est une question -> {"action": "say", "text": "reponse"}.

Actions:
{"action": "say", "text": "..."}
{"action": "stand"}
{"action": "sit"}
{"action": "wave"}
{"action": "walk", "direction": "forward"}
{"action": "walk", "direction": "backward"}
{"action": "walk", "direction": "left"}
{"action": "walk", "direction": "right"}
{"action": "walk", "direction": "demi_tour"}
{"action": "pick_up", "objet": "nom de l objet en francais"}
{"action": "imitate_start"}
{"action": "imitate_stop"}
{"action": "greet"}
{"action": "patrol_start"}
{"action": "patrol_stop"}
{"action": "apprendre_visage", "prenom": "prenom de la personne"}
{"action": "qui_est_la"}

Exemples IMPORTANTS :
"ramasse la bouteille" -> {"action": "pick_up", "objet": "bouteille"}
"prends le stylo" -> {"action": "pick_up", "objet": "stylo"}
"recupere la tasse" -> {"action": "pick_up", "objet": "tasse"}
"ramasse ca" -> {"action": "pick_up", "objet": "objet"}
"bonjour" -> {"action": "greet"}
"leve toi" -> {"action": "stand"}
"imite moi" -> {"action": "imitate_start"}
"commence la patrouille" -> {"action": "patrol_start"}
"retiens-moi je m'appelle Mateo" -> {"action": "apprendre_visage", "prenom": "Mateo"}
"souviens-toi de moi je m'appelle Robin" -> {"action": "apprendre_visage", "prenom": "Robin"}
"memorise-moi mon prenom est Lucas" -> {"action": "apprendre_visage", "prenom": "Lucas"}
"qui est devant toi" -> {"action": "qui_est_la"}
"qui vois-tu" -> {"action": "qui_est_la"}
"fais demi-tour" -> {"action": "walk", "direction": "demi_tour"}
"retourne-toi" -> {"action": "walk", "direction": "demi_tour"}
"tourne-toi" -> {"action": "walk", "direction": "demi_tour"}

IMPORTANT: Toute phrase contenant ramasse, prends, recupere, attrape -> pick_up.
IMPORTANT: Toute phrase contenant retiens, souviens, memorise avec un prenom -> apprendre_visage.
IMPORTANT: Toute phrase contenant demi-tour, retourne-toi, fais demi -> walk demi_tour."""
        messages = [{"role": "system", "content": system}] + historique_mistral
        response = client.chat.complete(model="mistral-small-latest", messages=messages)
        contenu = response.choices[0].message.content
        historique_mistral.append({"role": "assistant", "content": contenu})
        contenu = re.sub(r"```json|```", "", contenu).strip()
        match = re.search(r'\{.*?\}', contenu, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(contenu)

    def run(self):
        import speech_recognition as sr
        import struct as st_
        import io

        r = sr.Recognizer()
        self.log_signal.emit("Connexion au micro du robot...", "info")

        # Connexion au stream audio du robot
        audio_sock = None
        while self.running:
            try:
                audio_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                audio_sock.settimeout(5)
                audio_sock.connect((nao_ip, audio_port))
                self.log_signal.emit("Micro NAO connecte !", "success")
                break
            except:
                self.log_signal.emit("Attente micro NAO...", "info")
                time.sleep(2)

        while self.running:
            try:
                # Recevoir taille puis chunk WAV
                taille_bytes = recv_exact(audio_sock, 4)
                taille       = st_.unpack(">I", taille_bytes)[0]
                wav_data     = recv_exact(audio_sock, taille)

                # Ecrire dans un fichier temporaire (BytesIO non supporte dans .exe)
                import tempfile, os as os_
                import sys as sys_
                if getattr(sys_, 'frozen', False):
                    tmp_dir = os_.path.dirname(sys_.executable)
                else:
                    tmp_dir = tempfile.gettempdir()
                tmp_wav = os_.path.join(tmp_dir, "nao_audio_chunk.wav")
                with open(tmp_wav, "wb") as f:
                    f.write(wav_data)

                try:
                    audio_file = sr.AudioFile(tmp_wav)
                    with audio_file as source:
                        audio = r.record(source)
                finally:
                    try: os_.remove(tmp_wav)
                    except: pass

                try:
                    texte = r.recognize_google(audio, language="fr-FR")
                    if texte.strip():
                        self.log_signal.emit("Reconnu: " + texte, "info")
                        self.texte_signal.emit(texte)
                        self.chat_signal.emit("user", texte)
                        commande = self.comprendre(texte)
                        self.log_signal.emit("Commande: " + json.dumps(commande), "info")
                        self.commande_signal.emit(commande)
                except sr.UnknownValueError:
                    pass

            except Exception as e:
                self.log_signal.emit("Erreur audio : " + str(e), "error")
                # Reconnexion
                time.sleep(1)
                try:
                    audio_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    audio_sock.settimeout(5)
                    audio_sock.connect((nao_ip, audio_port))
                    self.log_signal.emit("Micro NAO reconnecte !", "success")
                except:
                    pass

    def stop(self):
        self.running = False

# ===================== THREAD IMITATION =====================
class ImitationThread(QThread):
    angles_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.frame   = None
        self.NEUTRE_LATERAL_D  = 0.03
        self.NEUTRE_LATERAL_G  = 0.20
        self.NEUTRE_VERTICAL_D = -0.18
        self.NEUTRE_VERTICAL_G = -0.18
        # Angles precedents pour le lissage
        self.angles_prev = {}
        # Facteur de lissage (0.0 = aucun lissage, 1.0 = bloque)
        self.LISSAGE = 0.5
        # Vitesse max de changement par frame (en radians)
        self.VITESSE_MAX = 0.15

    def lisser(self, angles_cibles):
        """Applique un filtre exponentiel + limite de vitesse entre frames."""
        if not self.angles_prev:
            self.angles_prev = angles_cibles.copy()
            return angles_cibles.copy()
        result = {}
        for k, v in angles_cibles.items():
            prev = self.angles_prev.get(k, v)
            # Lissage exponentiel
            lisse = prev + (1.0 - self.LISSAGE) * (v - prev)
            # Limite de vitesse
            delta = lisse - prev
            if abs(delta) > self.VITESSE_MAX:
                lisse = prev + self.VITESSE_MAX * (1 if delta > 0 else -1)
            result[k] = lisse
        self.angles_prev = result.copy()
        return result

    def run(self):
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        def calc_angle(a, b, c):
            a = np.array([a.x, a.y, a.z])
            b = np.array([b.x, b.y, b.z])
            c = np.array([c.x, c.y, c.z])
            ba = a - b; bc = c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return float(np.arccos(np.clip(cos, -1.0, 1.0)))

        derniere = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.running:
                if self.frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lm  = res.pose_landmarks.landmark
                    rsh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    lsh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    rel = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    lel = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                    rwr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                    lwr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                    nez = lm[mp_pose.PoseLandmark.NOSE]
                    od  = lm[mp_pose.PoseLandmark.RIGHT_EAR]
                    og  = lm[mp_pose.PoseLandmark.LEFT_EAR]

                    acd = calc_angle(rsh, rel, rwr)
                    acg = calc_angle(lsh, lel, lwr)
                    ld  = float(rsh.x - rel.x) - self.NEUTRE_LATERAL_D
                    vrd = float(rsh.y - rel.y) - self.NEUTRE_VERTICAL_D
                    lg  = float(lsh.x - rel.x) - self.NEUTRE_LATERAL_G
                    vrg = float(lsh.y - lel.y) - self.NEUTRE_VERTICAL_G
                    cx  = (od.x + og.x) / 2
                    cy2 = (rsh.y + lsh.y) / 2
                    yaw   = float((nez.x - cx) * 5)
                    pitch = float((nez.y - cy2) * 3)

                    # Calcul des angles bruts
                    rsp = float(np.clip(1.5 - vrd * 8, -1.5, 2.0))  # limite max reduite
                    lsp = float(np.clip(1.5 - vrg * 8, -1.5, 2.0))

                    # Compensation equilibre : si bras vers l'avant (ShoulderPitch negatif)
                    # les hanches compensent legerement vers l'arriere
                    hip_compensation = 0.0
                    if rsp < 0.5 or lsp < 0.5:
                        # Bras vers l'avant -> incliner les hanches pour contrebalancer
                        bras_avant = max(0.5 - rsp, 0.5 - lsp, 0.0)
                        hip_compensation = float(np.clip(bras_avant * 0.15, 0.0, 0.2))

                    maintenant = time.time()
                    if maintenant - derniere >= 0.1:
                        angles_cibles = {
                            "HeadYaw":        float(np.clip(yaw, -2.0, 2.0)),
                            "HeadPitch":      float(np.clip(pitch - 0.5, -0.7, 0.5)),
                            "RShoulderPitch": rsp,
                            "RShoulderRoll":  float(np.clip(-ld * 8, -1.3, 0.3)),
                            "RElbowRoll":     float(np.clip(acd - 1.5, 0.0, 1.5)),
                            "LShoulderPitch": lsp,
                            "LShoulderRoll":  float(np.clip(lg * 8, -0.3, 1.3)),
                            "LElbowRoll":     float(np.clip(-(acg - 1.5), -1.5, 0.0)),
                            # Hanches avec compensation
                            "RHipPitch": float(-0.04 - hip_compensation),
                            "LHipPitch": float(-0.04 - hip_compensation),
                            "RHipRoll": 0.0, "LHipRoll": 0.0,
                            "RKneePitch": float(np.clip(hip_compensation * 0.5, 0.0, 0.15)),
                            "LKneePitch": float(np.clip(hip_compensation * 0.5, 0.0, 0.15)),
                            "RAnklePitch": 0.0, "LAnklePitch": 0.0,
                            "RAnkleRoll":  0.0, "LAnkleRoll":  0.0
                        }
                        # Appliquer lissage + limite de vitesse
                        angles = self.lisser(angles_cibles)
                        envoyer_commande({"action": "imitate", "angles": angles})
                        self.angles_signal.emit(angles)
                        derniere = maintenant

    def stop(self):
        self.running = False

# ===================== THREAD SUIVI DE VISAGES =====================

# ===================== WIDGETS UTILITAIRES =====================
def make_card():
    f = QFrame()
    f.setObjectName("card")
    return f

def make_label(text, color=CYAN, size=10):
    l = QLabel(text)
    l.setStyleSheet(f"color:{color};font-size:{size}px;letter-spacing:2px;")
    return l

def make_stat_card(label, color=CYAN):
    card = make_card()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(4)
    layout.addWidget(make_label(label))
    val = QLabel("0")
    val.setStyleSheet(f"color:{color};font-size:22px;font-weight:bold;font-family:'Consolas';")
    layout.addWidget(val)
    return card, val

def make_toggle_btn(text, mode, callback):
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet("""
        QPushButton { background:#060a14; border:1px solid #1a2a4a; color:#6a8aaa; padding:6px 10px; font-size:11px; font-family:'Consolas'; border-radius:3px; }
        QPushButton:checked { border-color:#4fc3f7; color:#4fc3f7; background:rgba(79,195,247,0.05); }
        QPushButton:hover { border-color:#2a3a5a; color:#c0d4e8; }
    """)
    btn.clicked.connect(lambda checked: callback(mode, checked))
    return btn

# ===================== ONGLET CONTROLE =====================
class TabControle(QWidget):
    def __init__(self, app_ref):
        super().__init__()
        self.app = app_ref
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        left = QWidget()
        left.setStyleSheet("background:#020408;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#020408;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.video_label)

        cmd_bar = QWidget()
        cmd_bar.setStyleSheet(f"background:{BG3};border-top:1px solid {BORDER};")
        cmd_layout = QHBoxLayout(cmd_bar)
        cmd_layout.setContentsMargins(12, 8, 12, 8)
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("dis quelque chose a NAO...")
        self.cmd_input.returnPressed.connect(self.send_cmd)
        btn_send = QPushButton("ENVOYER")
        btn_send.clicked.connect(self.send_cmd)
        cmd_layout.addWidget(self.cmd_input)
        cmd_layout.addWidget(btn_send)
        left_layout.addWidget(cmd_bar)

        right = QWidget()
        right.setFixedWidth(310)
        right.setStyleSheet(f"background:{BG2};border-left:1px solid {BORDER};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        stats_widget = QWidget()
        stats_widget.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BORDER};")
        stats_layout = QGridLayout(stats_widget)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(8)
        c1, self.val_visages = make_stat_card("VISAGES", GREEN)
        c2, self.val_objets  = make_stat_card("OBJETS",  CYAN)
        c3, self.val_latence = make_stat_card("LATENCE", CYAN)
        c4, self.val_serveur = make_stat_card("SERVEUR", ORANGE)
        self.val_latence.setText("--ms")
        self.val_serveur.setText("OFF")
        stats_layout.addWidget(c1, 0, 0)
        stats_layout.addWidget(c2, 0, 1)
        stats_layout.addWidget(c3, 1, 0)
        stats_layout.addWidget(c4, 1, 1)
        right_layout.addWidget(stats_widget)

        modes_widget = QWidget()
        modes_widget.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BORDER};")
        modes_layout = QVBoxLayout(modes_widget)
        modes_layout.setContentsMargins(12, 10, 12, 10)
        modes_layout.setSpacing(6)
        modes_layout.addWidget(make_label("MODES"))
        self.btn_patrouille   = make_toggle_btn("patrouille",        "patrouille",        self.toggle_mode)
        self.btn_imitation    = make_toggle_btn("imitation",         "imitation",         self.toggle_mode)
        self.btn_det_objets   = make_toggle_btn("detection objets",  "detection_objets",  self.toggle_mode)
        self.btn_det_visages  = make_toggle_btn("detection visages", "detection_visages", self.toggle_mode)
        self.btn_suivi_visage = make_toggle_btn("suivi de visage",   "suivi_visage",      self.toggle_mode)
        self.btn_det_objets.setChecked(True)
        self.btn_det_visages.setChecked(True)
        modes_layout.addWidget(self.btn_patrouille)
        modes_layout.addWidget(self.btn_imitation)
        modes_layout.addWidget(self.btn_det_objets)
        modes_layout.addWidget(self.btn_det_visages)
        modes_layout.addWidget(self.btn_suivi_visage)
        right_layout.addWidget(modes_widget)

        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(12, 10, 12, 10)
        chat_layout.setSpacing(6)
        chat_layout.addWidget(make_label("HISTORIQUE"))
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet(f"background:{BG};border:1px solid {BORDER};border-radius:3px;padding:6px;font-size:11px;")
        chat_layout.addWidget(self.chat_area)
        right_layout.addWidget(chat_widget, 1)

        quick_widget = QWidget()
        quick_widget.setStyleSheet(f"background:{BG2};border-top:1px solid {BORDER};")
        quick_layout = QVBoxLayout(quick_widget)
        quick_layout.setContentsMargins(12, 10, 12, 10)
        quick_layout.setSpacing(6)

        # Header avec label memoire
        hdr_quick = QHBoxLayout()
        hdr_quick.addWidget(make_label("COMMANDES RAPIDES"))
        hdr_quick.addStretch()
        self.lbl_memoire = QLabel("memoire: 0 personne(s)")
        self.lbl_memoire.setStyleSheet(f"color:{ORANGE};font-size:10px;font-family:'Consolas';")
        hdr_quick.addWidget(self.lbl_memoire)
        quick_layout.addLayout(hdr_quick)

        grid = QGridLayout()
        grid.setSpacing(5)
        cmds = [
            ("leve-toi",    "stand",            None,         0, 0),
            ("assieds-toi", "sit",              None,         0, 1),
            ("salue",       "wave",             None,         0, 2),
            ("avance",      "walk",             "forward",    1, 0),
            ("recule",      "walk",             "backward",   1, 1),
            ("demi-tour",   "walk",             "demi_tour",  1, 2),
            ("bonjour",     "greet",            None,         2, 0),
            ("patrouille",  "patrol_toggle",    None,         2, 1),
            ("imite-moi",   "imitation_toggle", None,         2, 2),
            ("STOP",        "stand",            None,         3, 0),
            ("qui est la ?","qui_est_la",       None,         3, 1),
            ("memoriser",   "memoriser_ui",     None,         3, 2),
            ("oublier tt",  "oublier",          None,         4, 0),
        ]
        for label, action, direction, row, col in cmds:
            btn = QPushButton(label)
            if label == "STOP":
                btn.setStyleSheet("QPushButton{background:#060a14;border:1px solid rgba(244,67,54,0.3);color:#f44336;padding:5px;font-size:10px;border-radius:3px;}QPushButton:hover{background:rgba(244,67,54,0.1);border-color:#f44336;}")
            elif label == "oublier tt":
                btn.setStyleSheet("QPushButton{background:#060a14;border:1px solid rgba(255,152,0,0.3);color:#ff9800;padding:5px;font-size:10px;border-radius:3px;}QPushButton:hover{background:rgba(255,152,0,0.1);border-color:#ff9800;}")
            elif label == "memoriser":
                btn.setStyleSheet("QPushButton{background:#060a14;border:1px solid rgba(76,175,80,0.3);color:#4caf50;padding:5px;font-size:10px;border-radius:3px;}QPushButton:hover{background:rgba(76,175,80,0.1);border-color:#4caf50;}")
            elif label == "qui est la ?":
                btn.setStyleSheet("QPushButton{background:#060a14;border:1px solid rgba(79,195,247,0.3);color:#4fc3f7;padding:5px;font-size:10px;border-radius:3px;}QPushButton:hover{background:rgba(79,195,247,0.1);border-color:#4fc3f7;}")
            else:
                btn.setStyleSheet("QPushButton{background:#060a14;border:1px solid #1a2a4a;color:#6a8aaa;padding:5px;font-size:10px;border-radius:3px;}QPushButton:hover{border-color:#4fc3f7;color:#4fc3f7;}")
            btn.clicked.connect(lambda _, a=action, d=direction: self.quick_cmd(a, d))
            grid.addWidget(btn, row, col)
        quick_layout.addLayout(grid)
        right_layout.addWidget(quick_widget)

        layout.addWidget(left, 1)
        layout.addWidget(right)

    def update_frame(self, img):
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.video_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def update_stats(self, visages, objets):
        self.val_visages.setText(str(visages))
        self.val_objets.setText(str(objets))

    def update_memoire_label(self):
        n = len(memoire_personnes)
        self.lbl_memoire.setText(f"memoire: {n} personne(s)")
        if n > 0:
            self.lbl_memoire.setToolTip(", ".join(memoire_personnes.values()))

    def toggle_mode(self, mode, actif):
        if mode == "imitation":
            self.app.toggle_imitation(actif)
        elif mode == "patrouille":
            self.app.toggle_patrouille(actif)
        elif mode == "detection_objets":
            if self.app.video_thread:
                self.app.video_thread.detection_objets = actif
        elif mode == "detection_visages":
            if self.app.video_thread:
                self.app.video_thread.detection_visages = actif
        elif mode == "suivi_visage":
            self.app.toggle_suivi_visage(actif)

    def quick_cmd(self, action, direction=None):
        if action == "patrol_toggle":
            actif = not self.btn_patrouille.isChecked()
            self.btn_patrouille.setChecked(actif)
            self.app.toggle_patrouille(actif)
        elif action == "imitation_toggle":
            actif = not self.btn_imitation.isChecked()
            self.btn_imitation.setChecked(actif)
            self.app.toggle_imitation(actif)
        elif action == "qui_est_la":
            self.app.qui_est_la()
        elif action == "memoriser_ui":
            self.app.memoriser_via_ui()
        elif action == "oublier":
            self.app.oublier_tout()
        else:
            cmd = {"action": action}
            if direction:
                cmd["direction"] = direction
            if action == "greet":
                cmd["visage"] = visages_detectes > 0
            ok = envoyer_commande(cmd)
            self.add_chat("user", action)
            if ok:
                self.add_chat("nao", f"Commande executee: {action}")

    def send_cmd(self):
        texte = self.cmd_input.text().strip()
        if not texte:
            return
        self.cmd_input.clear()
        self.add_chat("user", texte)
        mapping = {
            "leve-toi":    {"action": "stand"},
            "assieds-toi": {"action": "sit"},
            "salue":       {"action": "wave"},
            "avance":      {"action": "walk", "direction": "forward"},
            "recule":      {"action": "walk", "direction": "backward"},
            "bonjour":     {"action": "greet", "visage": visages_detectes > 0},
        }
        cmd = mapping.get(texte.lower(), {"action": "say", "text": texte})
        envoyer_commande(cmd)

    def add_chat(self, role, text):
        color = CYAN if role == "user" else GREEN
        who = "vous" if role == "user" else "NAO"
        self.chat_area.append(f'<span style="color:{color};font-size:10px;">{who}</span>')
        self.chat_area.append(f'<span style="color:#c0d4e8;font-size:11px;">{text}</span>')
        self.chat_area.append("")

# ===================== ONGLET LOGS =====================
class TabLogs(QWidget):
    def __init__(self, app_ref):
        super().__init__()
        self.app = app_ref
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for title, btn_id in [("SERVEUR NAO", "srv"), ("APPLICATION", "app")]:
            panel = QWidget()
            panel.setStyleSheet(f"background:{BG};border-right:1px solid {BORDER};")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)

            hdr = QWidget()
            hdr.setStyleSheet(f"background:{BG3};border-bottom:1px solid {BORDER};")
            hdr_layout = QHBoxLayout(hdr)
            hdr_layout.setContentsMargins(14, 8, 14, 8)
            hdr_layout.addWidget(make_label(title, "#6a8aaa", 12))
            hdr_layout.addStretch()

            if btn_id == "srv":
                btn_start = QPushButton("DEMARRER")
                btn_start.setObjectName("btn-green")
                btn_start.clicked.connect(self.app.start_server)
                btn_stop = QPushButton("ARRETER")
                btn_stop.setObjectName("btn-red")
                btn_stop.clicked.connect(self.app.stop_server)
                hdr_layout.addWidget(btn_start)
                hdr_layout.addWidget(btn_stop)
            else:
                btn_clear = QPushButton("EFFACER")
                btn_clear.clicked.connect(lambda: self.log_app.clear())
                hdr_layout.addWidget(btn_clear)

            panel_layout.addWidget(hdr)

            log_area = QTextEdit()
            log_area.setReadOnly(True)
            log_area.setStyleSheet(f"background:{BG};border:none;padding:8px;font-size:11px;font-family:'Consolas';")
            panel_layout.addWidget(log_area)

            if btn_id == "srv":
                self.log_srv = log_area
            else:
                self.log_app = log_area

            layout.addWidget(panel)

    def add_log(self, message, niveau="info", source="app"):
        t = time.strftime("%H:%M:%S")
        colors = {"info": "#c0d4e8", "success": "#4caf50", "warning": "#ff9800", "error": "#f44336"}
        color = colors.get(niveau, "#c0d4e8")
        line = f'<span style="color:#6a8aaa;">{t}</span> <span style="color:{color};">{message}</span>'
        if source == "serveur":
            self.log_srv.append(line)
        else:
            self.log_app.append(line)

# ===================== FENETRE PRINCIPALE =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NAO CONTROL")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(STYLE)

        self.video_thread      = None
        self.micro_thread      = None
        self.imitation_thread  = None

        self.setup_ui()
        self.start_video()
        self.start_micro()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        topbar = QWidget()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background:{BG3};border-bottom:1px solid {BORDER};")
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)
        tb_layout.setSpacing(20)

        logo = QLabel("● NAO CONTROL")
        logo.setStyleSheet(f"color:{CYAN};font-size:16px;letter-spacing:3px;font-family:'Consolas';")
        tb_layout.addWidget(logo)

        self.conn_dot  = QLabel("●")
        self.conn_dot.setStyleSheet(f"color:{RED};font-size:10px;")
        self.conn_text = QLabel("deconnecte")
        self.conn_text.setStyleSheet(f"color:#6a8aaa;font-size:11px;font-family:'Consolas';")
        tb_layout.addWidget(self.conn_dot)
        tb_layout.addWidget(self.conn_text)

        self.mode_badge = QLabel("MODE NORMAL")
        self.mode_badge.setStyleSheet(f"color:{CYAN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(79,195,247,0.1);border:1px solid rgba(79,195,247,0.3);border-radius:3px;padding:3px 12px;")
        tb_layout.addWidget(self.mode_badge)
        tb_layout.addStretch()

        # Champ IP du robot
        lbl_ip = QLabel("IP :")
        lbl_ip.setStyleSheet(f"color:#6a8aaa;font-size:11px;font-family:'Consolas';")
        self.ip_input = QLineEdit(nao_ip)
        self.ip_input.setFixedWidth(140)
        self.ip_input.setStyleSheet(f"background:{BG};border:1px solid {BORDER};border-radius:3px;color:{CYAN};padding:4px 8px;font-size:11px;font-family:'Consolas';")
        self.ip_input.setPlaceholderText("192.168.x.x")
        btn_ip = QPushButton("APPLIQUER")
        btn_ip.setFixedWidth(80)
        btn_ip.setStyleSheet(f"background:rgba(79,195,247,0.1);border:1px solid rgba(79,195,247,0.3);border-radius:3px;color:{CYAN};padding:4px 8px;font-size:10px;font-family:'Consolas';")
        btn_ip.clicked.connect(self.appliquer_ip)
        tb_layout.addWidget(lbl_ip)
        tb_layout.addWidget(self.ip_input)
        tb_layout.addWidget(btn_ip)

        self.lbl_mic = QLabel("● micro actif")
        self.lbl_mic.setStyleSheet(f"color:{GREEN};font-size:11px;font-family:'Consolas';")
        tb_layout.addWidget(self.lbl_mic)

        main_layout.addWidget(topbar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tab_controle = TabControle(self)
        self.tab_logs     = TabLogs(self)
        self.tabs.addTab(self.tab_controle, "CONTROLE")
        self.tabs.addTab(self.tab_logs,     "LOGS")
        main_layout.addWidget(self.tabs)

        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_connection)
        self.conn_timer.start(2000)

    def appliquer_ip(self):
        global nao_ip
        nouvelle_ip = self.ip_input.text().strip()
        if nouvelle_ip:
            nao_ip = nouvelle_ip
            self.tab_logs.add_log(f"IP robot changee : {nao_ip}", "success", "app")
            self.conn_dot.setStyleSheet(f"color:{RED};font-size:10px;")
            self.conn_text.setText("deconnecte")
            self.tab_controle.val_serveur.setText("OFF")
            self.tab_controle.val_serveur.setStyleSheet(f"color:{ORANGE};font-size:22px;font-weight:bold;font-family:'Consolas';")

    def check_connection(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            result = s.connect_ex((nao_ip, nao_port))
            s.close()
            ok = (result == 0)
        except:
            ok = False
        self.conn_dot.setStyleSheet(f"color:{'#4caf50' if ok else '#f44336'};font-size:10px;")
        self.conn_text.setText("connecte" if ok else "deconnecte")
        if ok:
            self.tab_controle.val_serveur.setText("ON")
            self.tab_controle.val_serveur.setStyleSheet(f"color:{GREEN};font-size:22px;font-weight:bold;font-family:'Consolas';")

    def start_video(self):
        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.tab_controle.update_frame)
        self.video_thread.stats_signal.connect(self.tab_controle.update_stats)
        self.video_thread.prenom_signal.connect(self.on_prenom_reconnu)
        self.video_thread.start()
        self.tab_logs.add_log("Camera demarree", "success", "app")

    def start_micro(self):
        self.micro_thread = MicroThread("snesYeB80tCRwLjnMUeHorIqqpb1jVDP")
        self.micro_thread.log_signal.connect(lambda m, n: self.tab_logs.add_log(m, n, "app"))
        self.micro_thread.chat_signal.connect(self.tab_controle.add_chat)
        self.micro_thread.commande_signal.connect(self.on_commande)
        self.micro_thread.start()

    def on_commande(self, commande):
        action = commande.get("action")

        if action == "imitate_start":
            self.toggle_imitation(True)
            self.tab_controle.btn_imitation.setChecked(True)
            self.tab_controle.add_chat("nao", "Mode imitation active !")

        elif action == "imitate_stop":
            self.toggle_imitation(False)
            self.tab_controle.btn_imitation.setChecked(False)
            self.tab_controle.add_chat("nao", "Mode imitation desactive")

        elif action == "patrol_start":
            self.toggle_patrouille(True)
            self.tab_controle.btn_patrouille.setChecked(True)

        elif action == "patrol_stop":
            self.toggle_patrouille(False)
            self.tab_controle.btn_patrouille.setChecked(False)

        elif action == "apprendre_visage":
            prenom = commande.get("prenom", "inconnu")
            threading.Thread(target=self.apprendre_visage, args=(prenom,), daemon=True).start()

        elif action == "qui_est_la":
            self.qui_est_la()

        elif action == "say":
            self.tab_controle.add_chat("nao", commande.get("text", ""))
            envoyer_commande(commande)

        elif action == "greet":
            commande["visage"] = visages_detectes > 0
            envoyer_commande(commande)
            if dernier_prenom_vu and modele_entraine:
                self.tab_controle.add_chat("nao", f"Bonjour {dernier_prenom_vu} !")
                envoyer_commande({"action": "say", "text": f"Bonjour {dernier_prenom_vu} !"})
            else:
                self.tab_controle.add_chat("nao", "Bonjour !")

        elif action == "pick_up":
            nom_objet = commande.get("objet", "")
            info = None

            def normaliser(t):
                return unicodedata.normalize('NFD', t.lower()).encode('ascii', 'ignore').decode('utf-8')

            if self.video_thread:
                for classe_en, data in self.video_thread.objets_detectes.items():
                    nom_fr = data.get("nom_fr", "")
                    if normaliser(nom_objet) in normaliser(nom_fr) or normaliser(nom_fr) in normaliser(nom_objet):
                        info = data
                        break

            if info:
                distance = info["distance"]
                centre_x = info["centre_x"]
                self.tab_controle.add_chat("nao", f"Je vois {nom_objet} a {distance}m, je vais le ramasser !")
                self.tab_logs.add_log(f"pick_up: {nom_objet} a {distance}m, centre_x={centre_x:.2f}", "info", "app")

                if centre_x < 0.4:
                    envoyer_commande({"action": "walk", "direction": "left"})
                    time.sleep(1)
                elif centre_x > 0.6:
                    envoyer_commande({"action": "walk", "direction": "right"})
                    time.sleep(1)

                envoyer_commande({"action": "pick_up", "distance": distance})
                self.tab_controle.add_chat("nao", "Voila, c'est fait !")
            else:
                self.tab_controle.add_chat("nao", f"Je ne vois pas {nom_objet}, montre-moi ce que tu veux que je ramasse !")
                self.tab_logs.add_log(f"pick_up: {nom_objet} non visible — commande annulee", "warning", "app")
                # Ne rien envoyer au robot
        else:
            envoyer_commande(commande)
            self.tab_controle.add_chat("nao", f"Commande: {action}")

    def toggle_imitation(self, actif):
        if actif:
            if self.imitation_thread is None or not self.imitation_thread.isRunning():
                self.imitation_thread = ImitationThread()
                self.imitation_thread.angles_signal.connect(self.on_angles)
                self.imitation_thread.start()
                self.video_thread.imitation_thread_ref = self.imitation_thread
            self.mode_badge.setText("MODE IMITATION")
            self.mode_badge.setStyleSheet(f"color:{CYAN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(79,195,247,0.15);border:1px solid {CYAN};border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Mode imitation active", "success", "app")
        else:
            if self.imitation_thread:
                self.imitation_thread.stop()
                self.imitation_thread = None
                self.video_thread.imitation_thread_ref = None
            envoyer_commande({"action": "stand"})
            self.mode_badge.setText("MODE NORMAL")
            self.mode_badge.setStyleSheet(f"color:{CYAN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(79,195,247,0.1);border:1px solid rgba(79,195,247,0.3);border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Mode imitation desactive", "info", "app")

    def toggle_patrouille(self, actif):
        if actif:
            envoyer_commande({"action": "patrol_start"})
            self.mode_badge.setText("MODE PATROUILLE")
            self.mode_badge.setStyleSheet(f"color:{ORANGE};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(255,152,0,0.1);border:1px solid rgba(255,152,0,0.3);border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Mode patrouille active", "success", "app")
            self.tab_controle.add_chat("nao", "Je commence ma patrouille !")
        else:
            envoyer_commande({"action": "patrol_stop"})
            self.mode_badge.setText("MODE NORMAL")
            self.mode_badge.setStyleSheet(f"color:{CYAN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(79,195,247,0.1);border:1px solid rgba(79,195,247,0.3);border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Mode patrouille arrete", "info", "app")

    def on_angles(self, angles):
        global current_angles
        current_angles = angles

    def toggle_suivi_visage(self, actif):
        if actif:
            envoyer_commande({"action": "suivi_start"})
            self.mode_badge.setText("MODE SUIVI VISAGE")
            self.mode_badge.setStyleSheet(f"color:{GREEN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(76,175,80,0.1);border:1px solid rgba(76,175,80,0.3);border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Suivi de visage active (cote robot)", "success", "app")
            self.tab_controle.add_chat("nao", "Je vais suivre les visages !")
        else:
            envoyer_commande({"action": "suivi_stop"})
            envoyer_commande({"action": "head_scan", "direction": 0.0})
            self.mode_badge.setText("MODE NORMAL")
            self.mode_badge.setStyleSheet(f"color:{CYAN};font-size:11px;font-family:'Consolas';letter-spacing:1px;background:rgba(79,195,247,0.1);border:1px solid rgba(79,195,247,0.3);border-radius:3px;padding:3px 12px;")
            self.tab_logs.add_log("Suivi de visage desactive", "info", "app")

    # ===================== MEMOIRE DES PERSONNES =====================
    def apprendre_visage(self, prenom):
        global prochain_id, modele_entraine

        self.tab_logs.add_log(f"Apprentissage de '{prenom}' en cours...", "info", "app")
        self.tab_controle.add_chat("nao", f"Je vais te memoriser {prenom}, reste bien en face de moi !")
        envoyer_commande({"action": "say", "text": f"Je vais te memoriser {prenom}, ne bouge pas !"})

        self.video_thread.faces_apprentissage = []
        self.video_thread.ids_apprentissage   = []
        self.video_thread.id_apprentissage    = prochain_id
        self.video_thread.capture_pour_apprentissage = True

        # Plus d'echantillons sur plus longtemps pour compenser la qualite camera NAO
        debut = time.time()
        while len(self.video_thread.faces_apprentissage) < 40 and time.time() - debut < 20:
            nb = len(self.video_thread.faces_apprentissage)
            self.tab_logs.add_log(f"Capture {nb}/40...", "info", "app")
            # Demander a l'utilisateur de bouger legerement la tete pour avoir des variations
            if nb == 10:
                self.tab_controle.add_chat("nao", "Tourne legerement la tete a gauche...")
            elif nb == 20:
                self.tab_controle.add_chat("nao", "Maintenant a droite...")
            elif nb == 30:
                self.tab_controle.add_chat("nao", "Face a la camera...")
            time.sleep(0.4)

        self.video_thread.capture_pour_apprentissage = False
        faces_data = self.video_thread.faces_apprentissage[:]
        ids_data   = self.video_thread.ids_apprentissage[:]

        if len(faces_data) >= 5:
            # Augmentation de donnees : ajouter versions avec variations de luminosite
            faces_augmentees = []
            ids_augmentees   = []
            for roi, id_ in zip(faces_data, ids_data):
                faces_augmentees.append(roi)
                ids_augmentees.append(id_)
                # Version plus lumineuse
                bright = cv2.convertScaleAbs(roi, alpha=1.2, beta=20)
                faces_augmentees.append(bright)
                ids_augmentees.append(id_)
                # Version moins lumineuse
                dark = cv2.convertScaleAbs(roi, alpha=0.8, beta=-20)
                faces_augmentees.append(dark)
                ids_augmentees.append(id_)

            if modele_entraine:
                recognizer.update(faces_augmentees, np.array(ids_augmentees))
            else:
                recognizer.train(faces_augmentees, np.array(ids_augmentees))

            memoire_personnes[prochain_id] = prenom
            prochain_id    += 1
            modele_entraine = True
            total = len(faces_augmentees)
            self.tab_logs.add_log(f"'{prenom}' memorise avec {len(faces_data)} captures ({total} avec variations) !", "success", "app")
            self.tab_controle.add_chat("nao", f"Je me souviendrai de toi, {prenom} !")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.tab_controle.update_memoire_label)
            envoyer_commande({"action": "say", "text": f"Je me souviendrai de toi {prenom} !"})
        else:
            self.tab_logs.add_log("Pas assez de visages captures, reessaie !", "error", "app")
            self.tab_controle.add_chat("nao", "Je n'ai pas reussi a te voir, reessaie !")
            envoyer_commande({"action": "say", "text": "Je n'ai pas reussi a te voir, reessaie !"})

    def memoriser_via_ui(self):
        from PyQt6.QtWidgets import QInputDialog
        prenom, ok = QInputDialog.getText(self, "Memoriser une personne",
                                          "Entre ton prenom :", QLineEdit.EchoMode.Normal)
        if ok and prenom.strip():
            threading.Thread(target=self.apprendre_visage, args=(prenom.strip(),), daemon=True).start()

    def oublier_tout(self):
        global modele_entraine, prochain_id, dernier_prenom_vu
        memoire_personnes.clear()
        modele_entraine   = False
        prochain_id       = 0
        dernier_prenom_vu = ""
        self.tab_controle.update_memoire_label()
        self.tab_logs.add_log("Memoire des personnes effacee", "warning", "app")
        self.tab_controle.add_chat("nao", "J'ai oublie tout le monde !")
        envoyer_commande({"action": "say", "text": "J'ai oublie tout le monde !"})

    def qui_est_la(self):
        if not modele_entraine or len(memoire_personnes) == 0:
            self.tab_controle.add_chat("nao", "Je ne connais personne encore. Presente-toi !")
            envoyer_commande({"action": "say", "text": "Je ne connais personne encore. Presente-toi !"})
            return
        if dernier_prenom_vu:
            self.tab_controle.add_chat("nao", f"Je vois {dernier_prenom_vu} devant moi !")
            envoyer_commande({"action": "say", "text": f"Je vois {dernier_prenom_vu} devant moi !"})
        else:
            connus = ", ".join(memoire_personnes.values())
            self.tab_controle.add_chat("nao", f"Je ne reconnais personne. Je connais : {connus}.")
            envoyer_commande({"action": "say", "text": f"Je ne reconnais personne. Je connais {connus}."})

    def on_prenom_reconnu(self, prenom):
        self.tab_logs.add_log(f"Visage reconnu : {prenom}", "success", "app")

    def start_server(self):
        global serveur_process
        if serveur_process is not None:
            self.tab_logs.add_log("Serveur deja actif", "warning", "serveur")
            return

        def run():
            global serveur_process
            try:
                import paramiko
                self.tab_logs.add_log("Connexion SSH au robot...", "info", "serveur")

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(nao_ip, username="nao", password="SMART59", timeout=10)

                # Tuer les anciens processus sur les ports
                ssh.exec_command("fuser -k 9561/tcp 9562/tcp 9563/tcp 2>/dev/null; sleep 1")
                import time as t_; t_.sleep(2)

                # Lancer le serveur en arriere-plan et garder le canal ouvert
                transport = ssh.get_transport()
                channel   = transport.open_session()
                channel.exec_command("export PYTHONPATH=/opt/aldebaran/lib/python2.7/site-packages:$PYTHONPATH && python /home/nao/nao_server.py 2>&1")

                serveur_process = ssh  # stocker le client SSH

                self.tab_controle.val_serveur.setText("ON")
                self.tab_controle.val_serveur.setStyleSheet(f"color:{GREEN};font-size:22px;font-weight:bold;font-family:'Consolas';")
                self.tab_logs.add_log("Serveur NAO demarre sur le robot !", "success", "serveur")

                # Lire les logs du serveur en temps reel
                while serveur_process is not None:
                    if channel.recv_ready():
                        data = channel.recv(1024).decode("utf-8", errors="ignore")
                        for line in data.splitlines():
                            line = line.strip()
                            if line:
                                self.tab_logs.add_log(line, "info", "serveur")
                    if channel.exit_status_ready():
                        break
                    t_.sleep(0.2)

                self.tab_controle.val_serveur.setText("OFF")
                self.tab_controle.val_serveur.setStyleSheet(f"color:{ORANGE};font-size:22px;font-weight:bold;font-family:'Consolas';")
                self.tab_logs.add_log("Serveur NAO arrete", "warning", "serveur")
                ssh.close()
                serveur_process = None

            except Exception as e:
                self.tab_logs.add_log(f"Erreur SSH : {e}", "error", "serveur")
                serveur_process = None

        threading.Thread(target=run, daemon=True).start()

    def stop_server(self):
        global serveur_process
        if serveur_process is not None:
            try:
                import paramiko
                ssh_kill = paramiko.SSHClient()
                ssh_kill.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh_kill.connect(nao_ip, username="nao", password="SMART59", timeout=5)
                # Attendre que chaque commande se termine
                _, stdout, _ = ssh_kill.exec_command("pkill -9 -f nao_server.py")
                stdout.channel.recv_exit_status()
                _, stdout, _ = ssh_kill.exec_command("fuser -k 9561/tcp 9562/tcp 9563/tcp 2>/dev/null")
                stdout.channel.recv_exit_status()
                ssh_kill.close()
            except:
                pass
            serveur_process = None
            self.tab_controle.val_serveur.setText("OFF")
            self.tab_controle.val_serveur.setStyleSheet(f"color:{ORANGE};font-size:22px;font-weight:bold;font-family:'Consolas';")
            self.tab_logs.add_log("Serveur NAO arrete manuellement", "warning", "serveur")

    def closeEvent(self, event):
        if self.video_thread:     self.video_thread.stop()
        if self.micro_thread:     self.micro_thread.stop()
        if self.imitation_thread: self.imitation_thread.stop()
        envoyer_commande({"action": "suivi_stop"})
        envoyer_commande({"action": "patrol_stop"})
        self.stop_server()
        event.accept()

# ===================== MAIN =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
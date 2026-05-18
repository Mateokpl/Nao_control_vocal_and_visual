# -*- coding: utf-8 -*-
# nao_server_robot.py - Version VRAI ROBOT NAO 6
# Gere : commandes NAO (port 9561), stream camera (port 9562), stream audio (port 9563)
# Lancer sur le robot : python /home/nao/nao_server.py

import sys
import socket
import json
import time
import threading
import struct
import os

from naoqi import ALProxy

NAO_IP   = "127.0.0.1"
NAO_PORT = 9559

CMD_PORT   = 9561
CAM_PORT   = 9562
AUDIO_PORT = 9563

# Flags globaux
suivi_visage_actif = False
patrouille_actif   = False

# ===================== PATROUILLE AUTONOME (cote robot) =====================
def patrouille_thread():
    global patrouille_actif

    motion  = ALProxy("ALMotion",       NAO_IP, NAO_PORT)
    posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)
    tts     = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)

    motion.setStiffnesses("Body", 1.0)
    posture.goToPosture("StandInit", 0.5)

    # Charger le detecteur de visages
    import os
    cascade_paths = [
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    face_cascade = None
    for p in cascade_paths:
        if os.path.exists(p):
            try:
                import cv2
                face_cascade = cv2.CascadeClassifier(p)
                print("Cascade trouvee : " + p)
                break
            except:
                pass

    video  = None
    cam_id = None
    if face_cascade is not None:
        try:
            video  = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
            cam_id = video.subscribeCamera("patrol_cam", 0, 0, 0, 5)
            print("Camera patrouille activee")
        except:
            video = None

    derniere_salutation = 0
    angle_rotation_total = 0.0  # cumul des rotations effectuees

    print("Patrouille demarree")

    while patrouille_actif:
        try:
            # 1. Tourner le corps de 45 degres (0.785 rad)
            motion.setStiffnesses("Body", 1.0)
            motion.moveInit()
            motion.moveTo(0, 0, 0.785)  # rotation sur place 45 degres
            angle_rotation_total += 0.785
            time.sleep(0.5)

            # 2. Scanner la tete gauche/centre/droite
            for angle_tete in [-0.5, 0.0, 0.5]:
                if not patrouille_actif:
                    break
                motion.setAngles("HeadYaw",   angle_tete, 0.1)
                motion.setAngles("HeadPitch", 0.1,        0.1)
                time.sleep(1.0)

                # 3. Detecter les visages si possible
                visage_detecte = False
                if video is not None and face_cascade is not None:
                    try:
                        import numpy as np
                        image = video.getImageRemote(cam_id)
                        if image is not None:
                            w   = image[0]
                            h   = image[1]
                            raw = bytes(bytearray(image[6]))
                            yuv  = np.frombuffer(raw, dtype=np.uint8)
                            gray = yuv[0::2].reshape(h, w)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20,20))
                            if len(faces) > 0:
                                visage_detecte = True
                    except:
                        pass

                # 4. Si visage detecte et anti-spam ok -> saluer
                if visage_detecte and time.time() - derniere_salutation > 8.0:
                    # Remettre la tete au centre
                    motion.setAngles("HeadYaw",   0.0, 0.1)
                    motion.setAngles("HeadPitch", 0.1, 0.1)
                    time.sleep(0.3)
                    # Salutation
                    motion.setAngles("RShoulderPitch", -1.0, 0.4)
                    motion.setAngles("RShoulderRoll",  -0.5, 0.4)
                    motion.setAngles("RElbowRoll",      1.0, 0.4)
                    time.sleep(0.4)
                    motion.setAngles("RWristYaw",  1.0, 0.5)
                    time.sleep(0.3)
                    motion.setAngles("RWristYaw", -1.0, 0.5)
                    time.sleep(0.3)
                    motion.setAngles("RWristYaw",  1.0, 0.5)
                    tts.say("Bonjour, je vous ai repere !")
                    time.sleep(1.5)
                    posture.goToPosture("StandInit", 0.5)
                    time.sleep(0.5)
                    derniere_salutation = time.time()

            # 5. Apres un tour complet (8 x 45 = 360 degres) -> avancer d'un pas
            if angle_rotation_total >= 6.28:  # 2*pi = 360 degres
                angle_rotation_total = 0.0
                motion.moveInit()
                motion.moveTo(0.3, 0, 0)  # avancer de 30cm
                time.sleep(1.0)

        except Exception as e:
            print("Erreur patrouille : " + str(e))
            time.sleep(1)

    # Fin patrouille : remettre tete au centre
    try:
        motion.setAngles("HeadYaw",   0.0, 0.1)
        motion.setAngles("HeadPitch", 0.0, 0.1)
        posture.goToPosture("StandInit", 0.5)
    except:
        pass
    if video is not None and cam_id is not None:
        try:
            video.unsubscribe(cam_id)
        except:
            pass
    print("Patrouille arretee")

# ===================== SUIVI DE VISAGE (cote robot, sans latence reseau) =====================
def suivi_visage_thread():
    import os, sys

    # Haar cascade embarque sur le robot
    cascade_paths = [
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    cascade_path = None
    for p in cascade_paths:
        if os.path.exists(p):
            cascade_path = p
            break

    if cascade_path is None:
        print("Cascade non trouvee sur le robot, suivi desactive")
        return

    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except:
        print("OpenCV non disponible sur le robot")
        return

    video  = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
    motion = ALProxy("ALMotion",      NAO_IP, NAO_PORT)
    motion.setStiffnesses("Head", 1.0)

    # Resolution basse pour aller vite : kQQVGA=0 (160x120)
    cam_id = video.subscribeCamera("suivi_visage", 0, 0, 0, 10)
    # camera 0, resolution 0=QQVGA 160x120, colorspace 0=kYUV422, 10fps

    yaw_courant   = 0.0
    pitch_courant = 0.1
    scan_positions  = [-0.8, -0.4, 0.0, 0.4, 0.8, 0.4, 0.0, -0.4]
    scan_idx        = 0
    derniere_detect = 0

    print("Suivi visage demarre")

    while suivi_visage_actif:
        try:
            image = video.getImageRemote(cam_id)
            if image is None:
                time.sleep(0.1)
                continue

            w   = image[0]
            h   = image[1]
            raw = bytes(bytearray(image[6]))

            # Convertir YUV422 en gris (prendre 1 octet sur 2)
            import numpy as np
            yuv  = np.frombuffer(raw, dtype=np.uint8)
            gray = yuv[0::2].reshape(h, w)  # canal Y = luminance

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15)
            )

            if len(faces) > 0:
                # Visage le plus grand
                faces = sorted(faces.tolist(), key=lambda f: f[2]*f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                cx_visage = fx + fw / 2.0
                cy_visage = fy + fh / 2.0
                cx_image  = w / 2.0
                cy_image  = h / 2.0

                err_x = (cx_visage - cx_image) / cx_image
                err_y = (cy_visage - cy_image) / cy_image

                # Zone morte ±20%
                delta_yaw   = 0.0
                delta_pitch = 0.0
                if abs(err_x) > 0.20:
                    delta_yaw   = -err_x * 0.20
                if abs(err_y) > 0.20:
                    delta_pitch =  err_y * 0.15

                if abs(delta_yaw) > 0.01 or abs(delta_pitch) > 0.01:
                    nouveau_yaw   = max(-1.8, min(1.8,  yaw_courant   + delta_yaw))
                    nouveau_pitch = max(-0.5, min(0.4, pitch_courant + delta_pitch))
                    motion.changeAngles("HeadYaw",   delta_yaw,   0.2)
                    motion.changeAngles("HeadPitch", delta_pitch, 0.2)
                    yaw_courant   = nouveau_yaw
                    pitch_courant = nouveau_pitch

                derniere_detect = time.time()
                scan_idx = 0
                time.sleep(0.3)

            else:
                # Scan apres 3s sans visage
                if time.time() - derniere_detect > 3.0:
                    angle = scan_positions[scan_idx % len(scan_positions)]
                    motion.setAngles("HeadYaw",   angle, 0.08)
                    motion.setAngles("HeadPitch", 0.1,   0.08)
                    yaw_courant = angle
                    scan_idx   += 1
                    time.sleep(1.5)
                else:
                    time.sleep(0.2)

        except Exception as e:
            print("Erreur suivi : " + str(e))
            time.sleep(0.5)

    video.unsubscribe(cam_id)
    # Remettre la tete au centre
    try:
        motion.setAngles("HeadYaw",   0.0, 0.1)
        motion.setAngles("HeadPitch", 0.1, 0.1)
    except:
        pass
    print("Suivi visage arrete")

# ===================== SERVEUR COMMANDES =====================
def serveur_commandes():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", CMD_PORT))
    server.listen(1)
    print("Serveur commandes port " + str(CMD_PORT))

    while True:
        conn, addr = server.accept()
        data = conn.recv(4096).decode("utf-8").strip()
        if not data:
            conn.close()
            continue
        try:
            cmd = json.loads(data)
        except:
            conn.close()
            continue

        try:
            motion  = ALProxy("ALMotion",       NAO_IP, NAO_PORT)
            posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)
            tts     = ALProxy("ALTextToSpeech", NAO_IP, NAO_PORT)
            try:
                tts.setLanguage("French")
            except:
                pass
            try:
                motion.setStiffnesses("Body", 1.0)
            except:
                pass

            action = cmd.get("action", "")

            if action == "stand":
                posture.goToPosture("StandInit", 0.5)

            elif action == "sit":
                posture.goToPosture("Sit", 0.5)

            elif action == "wave":
                motion.setAngles("RShoulderPitch", -1.0, 0.3)
                motion.setAngles("RShoulderRoll", -0.5, 0.3)
                motion.setAngles("RElbowRoll", 1.0, 0.3)
                time.sleep(0.5)
                motion.setAngles("RWristYaw", 1.0, 0.5)
                time.sleep(0.3)
                motion.setAngles("RWristYaw", -1.0, 0.5)
                time.sleep(0.3)
                motion.setAngles("RWristYaw", 1.0, 0.5)

            elif action == "walk":
                motion.moveInit()
                direction = cmd.get("direction", "forward")
                if direction == "forward":
                    motion.moveTo(0.5, 0, 0)
                elif direction == "backward":
                    motion.moveTo(-0.5, 0, 0)
                elif direction == "left":
                    motion.moveTo(0, 0.3, 0)
                elif direction == "right":
                    motion.moveTo(0, -0.3, 0)
                elif direction == "demi_tour":
                    motion.moveTo(0, 0, 3.14159)

            elif action == "pick_up":
                posture.goToPosture("StandInit", 0.5)
                time.sleep(1)
                motion.moveInit()
                distance = cmd.get("distance", 0.3)
                motion.moveTo(distance, 0, 0)
                time.sleep(2)
                posture.goToPosture("Crouch", 0.5)
                time.sleep(1)
                motion.setStiffnesses("RArm", 1.0)
                motion.setAngles("RShoulderPitch", 1.4, 0.2)
                motion.setAngles("RShoulderRoll", -0.3, 0.2)
                motion.setAngles("RElbowRoll", 0.8, 0.2)
                motion.setAngles("RElbowYaw", 1.0, 0.2)
                motion.setAngles("RWristYaw", 0.0, 0.2)
                time.sleep(1)
                motion.setAngles("RHand", 0.0, 0.5)
                time.sleep(0.5)
                motion.setAngles("RShoulderPitch", -0.5, 0.2)
                time.sleep(1)
                posture.goToPosture("StandInit", 0.5)

            elif action == "greet":
                avec_visage = cmd.get("visage", False)
                prenom      = cmd.get("prenom", "")
                if avec_visage:
                    motion.setAngles("RShoulderPitch", -1.0, 0.3)
                    motion.setAngles("RShoulderRoll", -0.5, 0.3)
                    motion.setAngles("RElbowRoll", 1.0, 0.3)
                    time.sleep(0.5)
                    motion.setAngles("RWristYaw", 1.0, 0.5)
                    time.sleep(0.3)
                    motion.setAngles("RWristYaw", -1.0, 0.5)
                    time.sleep(0.3)
                    motion.setAngles("RWristYaw", 1.0, 0.5)
                    if prenom:
                        tts.say("Bonjour " + prenom.encode("utf-8") + " !")
                    else:
                        tts.say("Bonjour !")
                    time.sleep(1)
                    posture.goToPosture("StandInit", 0.5)
                else:
                    tts.say("Bonjour !")

            elif action == "patrol_start":
                global patrouille_actif
                if not patrouille_actif:
                    patrouille_actif = True
                    t = threading.Thread(target=patrouille_thread)
                    t.daemon = True
                    t.start()
                    print("Patrouille lancee")

            elif action == "patrol_stop":
                patrouille_actif = False
                print("Patrouille arretee")

            elif action == "patrol_step":
                # Garde pour compatibilite — rotation simple
                motion.setStiffnesses("Body", 1.0)
                motion.moveTo(0, 0, 0.3)
                time.sleep(1)

            elif action == "head_scan":
                direction = cmd.get("direction", 0.0)
                motion.setStiffnesses("Head", 1.0)
                motion.setAngles("HeadYaw", direction, 0.08)
                motion.setAngles("HeadPitch", 0.1, 0.08)

            elif action == "head_move":
                # Suivi de visage : ajustement fin de la tete
                yaw   = cmd.get("yaw", 0.0)
                pitch = cmd.get("pitch", 0.0)
                speed = cmd.get("speed", 0.15)
                motion.setStiffnesses("Head", 1.0)
                motion.changeAngles("HeadYaw",   yaw,   speed)
                motion.changeAngles("HeadPitch", pitch, speed)

            elif action == "patrol_greet":
                motion.setAngles("RShoulderPitch", -1.0, 0.4)
                motion.setAngles("RShoulderRoll", -0.5, 0.4)
                motion.setAngles("RElbowRoll", 1.0, 0.4)
                time.sleep(0.4)
                motion.setAngles("RWristYaw", 1.0, 0.5)
                time.sleep(0.3)
                motion.setAngles("RWristYaw", -1.0, 0.5)
                time.sleep(0.3)
                motion.setAngles("RWristYaw", 1.0, 0.5)
                tts.say("Bonjour, je vous ai repere !")
                time.sleep(1)
                posture.goToPosture("StandInit", 0.5)

            elif action == "patrol_announce":
                objet = cmd.get("objet", "quelque chose")
                if isinstance(objet, unicode):
                    objet = objet.encode("utf-8")
                tts.say("Je vois " + str(objet))

            elif action == "imitate":
                angles = cmd.get("angles", {})
                motion.setStiffnesses("Body", 1.0)
                for articulation, valeur in angles.items():
                    motion.setAngles(str(articulation), float(valeur), 0.3)

            elif action == "say":
                texte = cmd.get("text", "")
                if isinstance(texte, unicode):
                    texte = texte.encode("utf-8")
                tts.say(texte)

            elif action == "suivi_start":
                global suivi_visage_actif
                if not suivi_visage_actif:
                    suivi_visage_actif = True
                    t = threading.Thread(target=suivi_visage_thread)
                    t.daemon = True
                    t.start()
                    print("Suivi visage active")

            elif action == "suivi_stop":
                suivi_visage_actif = False
                print("Suivi visage desactive")

            try:
                noms    = motion.getBodyNames("Body")
                valeurs = motion.getAngles("Body", True)
                angles_dict = dict(zip(noms, valeurs))
                conn.send(json.dumps({"status": "ok", "angles": angles_dict}).encode("utf-8"))
            except:
                conn.send(json.dumps({"status": "ok", "angles": {}}).encode("utf-8"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            conn.send(json.dumps({"status": "error", "angles": {}}).encode("utf-8"))

        conn.close()

# ===================== STREAM CAMERA =====================
def stream_camera():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", CAM_PORT))
    server.listen(1)
    print("Stream camera port " + str(CAM_PORT))

    video  = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
    cam_id = video.subscribeCamera("stream_pc", 0, 1, 11, 10)
    # camera 0 = frontale, resolution 1 = QVGA 320x240, colorspace 11 = RGB, 10 fps

    while True:
        try:
            conn, addr = server.accept()
            print("Client camera : " + str(addr))
            while True:
                try:
                    image = video.getImageRemote(cam_id)
                    if image is None:
                        time.sleep(0.1)
                        continue

                    w   = image[0]
                    h   = image[1]
                    raw = bytes(bytearray(image[6]))

                    # Encoder en JPEG
                    try:
                        from PIL import Image as PILImage
                        import io
                        img  = PILImage.frombytes("RGB", (w, h), raw)
                        buf  = io.BytesIO()
                        img.save(buf, format="JPEG", quality=70)
                        jpeg = buf.getvalue()
                    except:
                        try:
                            import Image as PILImage
                            import StringIO
                            img  = PILImage.fromstring("RGB", (w, h), raw)
                            buf  = StringIO.StringIO()
                            img.save(buf, format="JPEG", quality=70)
                            jpeg = buf.getvalue()
                        except:
                            time.sleep(0.1)
                            continue

                    # Envoyer taille (4 octets) puis JPEG
                    taille = struct.pack(">I", len(jpeg))
                    conn.sendall(taille + jpeg)
                    time.sleep(0.1)

                except Exception as e:
                    print("Erreur frame : " + str(e))
                    break
            conn.close()

        except Exception as e:
            print("Erreur camera : " + str(e))
            time.sleep(1)

    video.unsubscribe(cam_id)

# ===================== STREAM AUDIO =====================
def stream_audio():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", AUDIO_PORT))
    server.listen(1)
    print("Stream audio port " + str(AUDIO_PORT))

    while True:
        try:
            conn, addr = server.accept()
            print("Client audio : " + str(addr))
            recorder = ALProxy("ALAudioRecorder", NAO_IP, NAO_PORT)
            tmp = "/tmp/nao_chunk.wav"

            while True:
                try:
                    # Enregistre 2 secondes
                    recorder.startMicrophonesRecording(tmp, "wav", 16000, [0, 0, 1, 0])
                    time.sleep(2.0)
                    recorder.stopMicrophonesRecording()
                    time.sleep(0.1)

                    if os.path.exists(tmp):
                        with open(tmp, "rb") as f:
                            data = f.read()
                        os.remove(tmp)
                        taille = struct.pack(">I", len(data))
                        conn.sendall(taille + data)

                except Exception as e:
                    print("Erreur audio : " + str(e))
                    try:
                        recorder.stopMicrophonesRecording()
                    except:
                        pass
                    break

            conn.close()

        except Exception as e:
            print("Erreur serveur audio : " + str(e))
            time.sleep(1)

# ===================== MAIN =====================
if __name__ == "__main__":
    t_cmd   = threading.Thread(target=serveur_commandes)
    t_cam   = threading.Thread(target=stream_camera)
    t_audio = threading.Thread(target=stream_audio)
    t_cmd.daemon   = True
    t_cam.daemon   = True
    t_audio.daemon = True
    t_cmd.start()
    t_cam.start()
    t_audio.start()
    print("Serveurs demares - cmd:" + str(CMD_PORT) + " cam:" + str(CAM_PORT) + " audio:" + str(AUDIO_PORT))
    while True:
        time.sleep(1)
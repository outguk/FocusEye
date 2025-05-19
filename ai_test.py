import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # 얼굴 랜드마크 정교화 옵션 활성화

# YOLO 모델 초기화 함수
def construct_yolo_v3():
    # 클래스 이름을 coco_names.txt에서 읽어와 리스트로 저장
    with open('coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # YOLO 모델 가중치와 구성 파일 로드
    model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # YOLO의 출력 레이어 이름 추출
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    return model, out_layers, class_names

# YOLO 탐지 함수
def yolo_detect(img, yolo_model, out_layers):
    # 입력 이미지를 YOLO에 맞게 전처리
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)

    # YOLO 탐지 실행
    outputs = yolo_model.forward(out_layers)

    # 탐지 결과 정리
    box, conf, class_id = [], [], []
    for output in outputs:
        for vec85 in output:
            scores = vec85[5:]  # 클래스별 신뢰도 점수
            max_id = np.argmax(scores)  # 가장 높은 점수의 클래스 ID
            confidence = scores[max_id]  # 해당 클래스의 신뢰도
            if confidence > 0.5:  # 신뢰도 기준 필터링
                center_x, center_y = int(vec85[0] * width), int(vec85[1] * height)  # 박스 중심 좌표
                w, h = int(vec85[2] * width), int(vec85[3] * height)  # 박스 크기
                x, y = int(center_x - w / 2), int(center_y - h / 2)  # 박스 좌상단 좌표
                box.append([x, y, x + w, y + h])  # 박스 좌표 추가
                conf.append(float(confidence))  # 신뢰도 추가
                class_id.append(max_id)  # 클래스 ID 추가

    # 비최대 억제 (NMS) 적용하여 중복 제거
    indices = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i], class_id[i]] for i in range(len(box)) if i in indices]
    return objects

# EAR (눈 깜빡임 비율) 계산 함수
def calculate_ear(landmarks, eye_indices):
    # 눈의 좌우 및 상하 좌표 계산
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = ((landmarks[eye_indices[1]][0] + landmarks[eye_indices[2]][0]) / 2, 
           (landmarks[eye_indices[1]][1] + landmarks[eye_indices[2]][1]) / 2)
    bottom = ((landmarks[eye_indices[4]][0] + landmarks[eye_indices[5]][0]) / 2, 
              (landmarks[eye_indices[4]][1] + landmarks[eye_indices[5]][1]) / 2)

    # EAR 계산 공식
    ear = (abs(top[1] - bottom[1])) / (abs(left[0] - right[0]) + 1e-6)  # 안정성을 위해 1e-6 추가
    return ear

# YOLO 모델 및 레이어 이름, 클래스 이름 로드
model, out_layers, class_names = construct_yolo_v3()

# YOLO와 Mediapipe 결과를 저장할 변수
yolo_results = []
mediapipe_results = []

# 상태 초기화
blink_count = 0  # 깜빡임 횟수
eye_closed = False  # 눈 감김 여부
elapsed_time = 0  # 탐지된 총 시간
person_detected = False  # 사람 탐지 여부
timer_start = None  # 타이머 시작 시간

# 카메라 시작 (기본 카메라 열기)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라 연결 실패")
    exit()

# 해상도 설정 (320x240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 프레임 속도 설정 (30 FPS)
cap.set(cv2.CAP_PROP_FPS, 30)

# YOLO 프로세스
def process_yolo(frame):
    global yolo_results
    yolo_results = yolo_detect(frame, model, out_layers)

# Mediapipe 프로세스
def process_mediapipe(frame):
    global mediapipe_results
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mediapipe는 RGB 형식 요구
    results = face_mesh.process(rgb_frame)  # 얼굴 랜드마크 처리
    mediapipe_results = results.multi_face_landmarks  # 처리 결과 저장

# 메인 루프
while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("프레임 획득에 실패했습니다.")
        break

    # YOLO와 Mediapipe를 멀티스레드로 실행
    yolo_thread = threading.Thread(target=process_yolo, args=(frame,))
    mediapipe_thread = threading.Thread(target=process_mediapipe, args=(frame,))
    yolo_thread.start()
    mediapipe_thread.start()
    yolo_thread.join()
    mediapipe_thread.join()

    # YOLO 결과 처리
    current_person_detected = False
    for obj in yolo_results:
        x1, y1, x2, y2, confidence, class_id = obj
        if class_names[class_id] == "person":  # 탐지된 객체가 "사람"일 경우
            current_person_detected = True
            text = f"{class_names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 탐지된 박스 그리기
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 탐지 시간 계산
    if current_person_detected:
        if not person_detected:  # 새로 탐지되면 타이머 시작
            timer_start = time.time()
        person_detected = True
    else:
        if person_detected and timer_start:  # 탐지가 종료되면 시간 누적
            elapsed_time += time.time() - timer_start
        person_detected = False
        timer_start = None

    # Mediapipe 결과 처리
    if mediapipe_results:
        for face_landmarks in mediapipe_results:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

            # EAR 계산
            left_ear = calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])  # 왼쪽 눈
            right_ear = calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])  # 오른쪽 눈
            ear = (left_ear + right_ear) / 2  # 평균 EAR 계산
            if ear < 0.2:  # EAR 임계값으로 눈 감김 여부 확인
                if not eye_closed:
                    blink_count += 1  # 눈 감김으로 깜빡임 횟수 증가
                    eye_closed = True
            else:
                eye_closed = False

    # 타이머와 깜빡임 횟수 표시
    display_time = elapsed_time + (time.time() - timer_start if person_detected and timer_start else 0)
    cv2.putText(frame, f"Time: {display_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("YOLO + Mediapipe", frame)

    # 속도 조절 (15 FPS 이하로 유지)
    time.sleep(1 / 15)

    if cv2.waitKey(1) & 0xFF == ord('x'):  # 'x' 키 입력 시 종료
        break

# 카메라 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()

# 마지막 탐지 시간 누적
if person_detected and timer_start:
    elapsed_time += time.time() - timer_start

# 최종 결과 출력
print(f"총 깜빡임 횟수: {blink_count}")
print(f"총 사람 탐지 시간: {elapsed_time:.2f} 초")

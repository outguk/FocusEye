from threading import Lock
import cv2
import mediapipe as mp
import numpy as np
import time
import threading

from queue import Queue

# YOLO 모델 초기화 함수
def construct_yolo_v3():
    # 클래스 이름을 coco.names에서 읽어와 리스트로 저장
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

# YOLO 프로세스
def process_yolo(frame):
    results = yolo_detect(frame, model, out_layers)
    yolo_results_queue.put(results)

# Mediapipe 프로세스
def process_mediapipe(frame):
    global mediapipe_results
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mediapipe는 RGB 형식 요구
    results = face_mesh.process(rgb_frame)  # 얼굴 랜드마크 처리
    mediapipe_results_queue.put(results.multi_face_landmarks) # 처리 결과 저장  

yolo_results_queue = Queue()
mediapipe_results_queue = Queue()

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # 얼굴 랜드마크 정교화 옵션 활성화

# 상태 변수
is_running = False  # 영상 캡처 실행 여부
capture_thread = None  # 영상 캡처 스레드
user_states = {}  # {user_id: {"is_running": bool, "blink_count": int, "elapsed_time": float}}
# Lock 초기화 멀티 스레드인 경우 race condition 방지
user_states_lock = Lock()

# YOLO 모델 및 레이어 이름, 클래스 이름 로드
try:
  model, out_layers, class_names = construct_yolo_v3()
  print("YOLO initialized successfully")
except Exception as e:
  print(f"YOLO initialized failed: {e}")

def main_process(user_id):
  
  # YOLO와 Mediapipe 결과를 저장할 변수
  yolo_results = []
  mediapipe_results = []

  # 상태 초기화
  blink_count = 0  # 깜빡임 횟수
  eye_closed = False  # 눈 감김 여부
  elapsed_time = 0  # 탐지된 총 시간
  person_detected = False  # 사람 탐지 여부
  timer_start = None  # 타이머 시작 시간
    
  global user_states
  # 카메라 시작 (기본 카메라 열기)
  cv2.destroyAllWindows()  # 이전 창 닫기
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  if not cap.isOpened():
      print(f"User {user_id}:카메라 연결 실패")
      exit()

  # 해상도 설정 (320x240)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

  # 프레임 속도 설정 (30 FPS)
  cap.set(cv2.CAP_PROP_FPS, 30)

  # 메인 루프
  while user_states[user_id]["is_running"]: # 실행 상태 확인
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

      # **큐에서 결과 가져오기**
      if not yolo_results_queue.empty():
          yolo_results = yolo_results_queue.get()
      else:
          yolo_results = []

      if not mediapipe_results_queue.empty():
          mediapipe_results = mediapipe_results_queue.get()
      else:
          mediapipe_results = None

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
                      user_states[user_id]["blink_count"] = blink_count # 깜박임 횟수 업데이트
                      eye_closed = True
              else:
                  eye_closed = False

      # 타이머와 깜빡임 횟수 표시 
      display_time = round(elapsed_time + (time.time() - timer_start if person_detected and timer_start else 0),1) # 소수점 1자리 제한
      cv2.putText(frame, f"Time: {display_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(frame, f"Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      # 시간 업데이트
      user_states[user_id]["elapsed_time"] = display_time

      # 프레임 출력
      cv2.imshow("YOLO + Mediapipe", frame)
      cv2.waitKey(1)  # 렌더링 대기 (필수)

      # 속도 조절 (15 FPS 이하로 유지)
      time.sleep(1 / 30)

      # if cv2.getWindowProperty("Video Capture", cv2.WND_PROP_VISIBLE) < 1:
      #     print(f"User {user_id}: 영상 캡쳐 화면이 닫혔습니다.")  
      #     stop_video_capture(user_id)
      #     break

      if cv2.waitKey(1) & 0xFF == ord('x'):  # 'x' 키 입력 시 종료
          break

  # 카메라 및 윈도우 해제
  cap.release()
  cv2.destroyAllWindows() 

  # 마지막 탐지 시간 누적
  if person_detected and timer_start:
      elapsed_time += round(time.time() - timer_start, 1)

  # 측정된 blink_count와 elapsed_time을 업데이트
  # user_states[user_id]["blink_count"] = blink_count
  user_states[user_id]["elapsed_time"] = elapsed_time

  # 최종 결과 출력
  print(f"총 깜빡임 횟수: {blink_count}")
  print(f"총 사람 탐지 시간: {elapsed_time:.1f} 초")
      

async def start_video_capture(user_id):
    """사용자별 영상 캡처 시작"""
    if user_id in user_states and user_states[user_id]["is_running"]:
        print(f"User {user_id}: 이미 실행 중입니다.")
        print("현재 상태:", user_states)
        return

    user_states[user_id] = {"is_running": True, "blink_count": 0, "elapsed_time": 0.0}
    threading.Thread(target=main_process, args=(user_id,)).start()
    print(f"User {user_id}  : 영상 캡처 시작.")
    print("현재 상태:", user_states)

def stop_video_capture(user_id):
    """사용자별 영상 캡처 종료"""
    if user_id not in user_states or not user_states[user_id]["is_running"]:
        print(f"User {user_id}: 실행 중인 캡처가 없습니다.")
        print("현재 상태:", user_states)
        return

    user_states[user_id]["is_running"] = False
    print(f"User {user_id}: 영상 캡처 종료.")
    print("현재 상태:", user_states)

    # 스레드 종료 처리
    capture_thread = user_states[user_id].get("capture_thread")
    if capture_thread and capture_thread.is_alive():
        capture_thread.join()  # 스레드 종료 대기

    # 상태 초기화 (리스트에서 유저 제거)
    del user_states[user_id]
    print(f"User {user_id}: 리스트에서 제거.")
    print("현재 상태:", user_states)

def process_camera_data(user_id):
    """현재 사용자 데이터 반환"""
    return user_states[user_id]["blink_count"], user_states[user_id]["elapsed_time"]

if __name__ == "__main__":
    # 테스트 또는 독립 실행용 코드
    user_id = 1  # 예제 사용자 ID
    start_video_capture(user_id)
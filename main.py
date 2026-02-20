from numpy import byte
import cv2
from datetime import datetime
import pandas as pd
# from ultralytics import YOLO
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

import supervision as sv

from collections import deque
from detect_stopped_car import detect_highway_stopped_vehicle
from get_speed_direction import detect_car_direction
from get_wrong_way_and_speeding import wrong_way_drive, get_real_speed

file_path = "./videos/test2.mp4"
cctv_id = "best_11n_newdataset_v1"
cap = cv2.VideoCapture(file_path)


# 프레임 속도를 1초를 환산
fps = cap.get(cv2.CAP_PROP_FPS)
# 속력을 계산할 때, 사용한 시간 7프레임 간격으로 속력 계산
one_second = 1 * 16 / fps

CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

model = RFDETRNano()
model.optimize_for_inference()
tracker = sv.ByteTrack()

print("Classes:", model.class_names)
# model1 = YOLO("yolov8n.pt")

#  Classes: {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'people', 4: 'person', 5: 'truck'}
# 실시간으로 4초 120개의 frame을 저장할 리스트 dq 설정
dq = deque(maxlen=120)

frame_count = 0

# 탐지된 차량 저장 딕션너리

vehicle_id = {}
recording_start = {}

df = pd.DataFrame(
    columns=["id", "type", "direction", "speed", "datetime", "illegal", "file_name"]
)


while cap.isOpened():
    frame_count += 1
    # print(frame_count)
    direction = None
    if frame_count % 2 != 0:
        continue

    # print("Vehicle ID", vehicle_id)
    success, frame = cap.read()
    if not success:
        break

    # frame[:450, :] = 0  # 화면 상단 하얀색으로 전처리
    frame[:200, :] = 0

    results = model.predict(frame, 0.5) # 탐지
    results = tracker.update_with_detections(results) # id 부여

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in results.class_id]

    # conf=0.5 , tracker="botsort.yaml" 

    print(results)

    print("="*50)

    annotated_frame = sv.BoxAnnotator().annotate(frame, results)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, results, labels)

    # veh[0] = 박스 
    # car[1] = Mask
    # car[2] = class id
    # car[3] = id

    for car in results:
        print(car)
        print("="*50)

        # 차량 ID와 그 차량의 중심값
        tid = int(car[3])

        cx, cy= (car[0][0]+car[0][2])/2, (car[0][1] + car[0][3])/2

        print(cx,cy)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 탐지 존 영역 수정

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # in_zone = 300 < cy < 330  # test1 video        
        in_zone = 510 < cy < 540  # test2 video
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ==========================================================================
        # ==============================  차량 방향 및 속력 탐지 =============================
        # ==========================================================================
        # in_zone 범위를 시각적으로 표시

         # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
         #test1 video
        # cv2.line(result, (0, 300), (1920, 300), (0, 255, 0), 2)  # 상단 경계선
        # cv2.line(result, (0, 320), (1920, 320), (0, 255, 0), 2)  # 하단 경계선
        
        #test2 video
        cv2.line(annotated_frame, (0, 510), (1920, 510), (0, 255, 0), 2)  # 상단 경계선
        cv2.line(annotated_frame, (0, 540), (1920, 540), (0, 255, 0), 2)  # 하단 경계선
         # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        if tid not in vehicle_id and in_zone:
            vehicle_id[tid] = tid
            # 프로그램 작동여부 1번 확인소
            # print(f"처음 Id: {tid}, y1의 좌표: {cy}")
            result_dir_speed = detect_car_direction(
                tid, cx, cy, frame_count, one_second
            )

        if tid in vehicle_id.keys():
            result_dir_speed = detect_car_direction(
                tid, cx, cy, frame_count, one_second
            )

            if result_dir_speed is None:

                continue

            else:
                cls = int(car[3])
                direction, speed_px = result_dir_speed
                # 프로그램 작동여부 확인 2번쩨 포인트
                # print(speed_px)
                data_time = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
                # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]
                cols = ["type", "direction", "speed", "datetime"]
                df.loc[tid, "id"] = tid
                df.loc[tid, "type"] = cls
                df.loc[tid, "direction"] = direction
                df.loc[tid, "speed"] = speed_px
                df.loc[tid, "datetime"] = data_time
                del vehicle_id[tid]

        # ==========================================================================
        # ==============================  역방향 탐지와 및 속력 보정 =============================
        # ==========================================================================

        # # 역방향 차량 탐지 및 속력 보정후 실제 속력
        # # 상행선 하행선의 차량 데이터 50개를 활용하여 최근접분류를 통해 역방향을 찾는다.
        try:
            if df.loc[tid, "speed"] is not None:
                car_direction = df.loc[tid, "direction"]
                speed_px1 = df.loc[tid, "speed"]
                cls = df.loc[tid, "type"]

                detect_wrong_way_car = wrong_way_drive(
                    tid, cls, cx, cy, car_direction, speed_px1
                )

                # 아래의 car_direction은 차선과 관계없는 실제 차량의 주행방향
                # 최근접 분류 머신런닝을 통해 역주행 여부 판단

                if detect_wrong_way_car:
                    print(f"경고!! 역주행 차량{tid}가 발견되었습니다.")

                    wrong_way_datetime = datetime.now()
                    # file_name 예시 parking_25-12-19_13:00:00.mp4

                    file_name = f"wrong_way[{tid}]_{wrong_way_datetime.strftime("%Y_%m_%d_%H_%M_%S")}"
                    recording_start[tid] = [(frame_count + 100), file_name, cx, cy]

                    # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]
                    df.loc[tid, ["illegal", "file_name"]] = [
                        "wrong_way",
                        f"{file_name}.mp4",
                    ]

                    # recording_start[tid][2] = cx
                    # recording_start[tid][3] = cy
                # 차선에 따른 속력보정후 실제 속력
                real_speed = get_real_speed(cx, cy, car_direction)
                if real_speed is None:
                    df.loc[tid, "speed"] = "learning"

                else:

                    df.loc[tid, "speed"] = speed_px1 * real_speed
                    # print(cls, df.loc[tid, "speed"])

        except:
            pass

        # 불법차량 빨간색 원으로 표시할 좌표  # 188, 189줄 참고
        try:
            recording_start[tid][2] = cx
            recording_start[tid][3] = cy
        except:
            pass

    for key in recording_start.keys():
        cx = int(recording_start[key][2])
        cy = int(recording_start[key][3])
        # 불법차량 녹화 작동여부 확인소 3번째
        # print(key, cx, cy, recording_start[key])
        cv2.circle(result, (cx, cy), 20, (0, 0, 255), 3)

    # 리코딩 시작:  아이디 그리고 불법 종류 별로
    if recording_start:
        for key in list(recording_start.keys()):
            if recording_start[key][0] == frame_count:
                print("리코딩을 시작합니다")
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                file_out_path = f"./results/{recording_start[key][1]}.mp4"
                out = cv2.VideoWriter(
                    file_out_path, fourcc, fps, (720, 480)
                )  # (720, 480),  (1920, 1080)

                for f in dq:
                    out.write(f)

                out.release()
                recording_start.pop(key)  # 한 번만 실행

    cv2.imshow("frame", annotated_frame)

    # 녹화할 영상 dq에  담기
    dq.append(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        print(df.tail())

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

    # 30분 간격(프레임 54000)일, 그리고 버튼 [r]을 누르면, 엑셀파일로 데이터 내보낸다.
    if frame_count % 54000 == 0 or (cv2.waitKey(1) & 0xFF == ord("r")):
        print("엑셀로 교통분석을 내보냅니다.")

        file_excel = f"./results/highway[{cctv_id}].xlsx"
        df.to_excel(file_excel, index=False)

# 데이터를 주기적으로 엑셀파일로 방출한다.


# DB로 저장한다.

file_excel = f"./results/highway_traffic[{cctv_id}]_final.xlsx"
df.to_excel(file_excel, index=False)

cap.release()
cv2.destroyAllWindows()

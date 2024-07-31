import cv2
import numpy as np
import tensorflow as tf

# 학습된 LaneNet 모델 로드
model_path = '/Users/seongjinjeong/Downloads/lanenet_.model/'
model = tf.saved_model.load(model_path)

# 이미지 전처리 함수
def preprocess_image(image):
    image = cv2.resize(image, (640, 480))  # 모델 입력 크기에 맞게 조정
    image = image / 127.5 - 1.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 차선 추출 함수
def extract_lanes(image, model):
    preprocessed_image = preprocess_image(image)
    pred = model(preprocessed_image, training=False)
    binary_seg_image = pred['binary_seg_logits']  # 이진 차선 분할 결과
    instance_seg_image = pred['instance_seg_logits']  # 인스턴스 분할 결과
    return binary_seg_image, instance_seg_image

# 결과 후처리 함수
def postprocess_result(binary_seg_image, instance_seg_image):
    binary_seg_image = binary_seg_image[0].numpy()
    instance_seg_image = instance_seg_image[0].numpy()
    binary_seg_image = np.argmax(binary_seg_image, axis=-1)
    return binary_seg_image, instance_seg_image

# 이미지 불러오기 및 차선 추출
image_path = '/Users/seongjinjeong/lane_detection/road_lane_line_detection/test_images/solidWhiteRight.jpg'
image = cv2.imread(image_path)
binary_seg_image, instance_seg_image = extract_lanes(image, model)
binary_seg_image, instance_seg_image = postprocess_result(binary_seg_image, instance_seg_image)

# 차선 결과 시각화
def visualize_lanes(image, binary_seg_image):
    lanes_image = image.copy()
    lanes_image[binary_seg_image == 1] = [0, 255, 0]  # 차선을 초록색으로 표시
    return lanes_image

lanes_image = visualize_lanes(image, binary_seg_image)

# 결과 이미지 저장
output_path = '/Users/seongjinjeong/lane_detection/road_lane_line_detection/test_images/output.jpg'
cv2.imwrite(output_path, lanes_image)

# 결과 이미지 표시
cv2.imshow('Lanes Image', lanes_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

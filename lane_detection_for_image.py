import cv2
import numpy as np
import math
from slidewindow import SlideWindow

class LaneDetection:
    def __init__(self):
        # 초기화
        self.slidewindow = SlideWindow()

        # 트랙바 초기값 설정
        self.lower_yellow = np.array([95, 51, 101])
        self.upper_yellow = np.array([104, 255, 255])
        self.lower_white = np.array([80, 0, 158])
        self.upper_white = np.array([132, 16, 255])

        # 트랙바 창 생성
        cv2.namedWindow("Trackbars")
        self.create_trackbars()

        # 이미지 로드
        self.img = cv2.imread('/Users/seongjinjeong/lane_detection/road_lane_line_detection/test_images/test_image.jpg')  # 이미지 파일 경로를 지정하세요
        if self.img is None:
            print("Image not found!")
            return

        self.process_image()

    def process_image(self):
        y, x = self.img.shape[0:2]

        # HSV 변환 및 마스크 생성
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(self.img_hsv)

        # 트랙바로부터 HSV 값 읽기
        self.read_trackbar_values()

        # 노란색 및 흰색 마스크 생성
        mask_yellow = cv2.inRange(self.img_hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(self.img_hsv, self.lower_white, self.upper_white)
        filtered_yellow = cv2.bitwise_and(self.img, self.img, mask=mask_yellow)
        filtered_white = cv2.bitwise_and(self.img, self.img, mask=mask_white)
        masks = cv2.bitwise_or(mask_yellow, mask_white)
        filtered_img = cv2.bitwise_and(self.img, self.img, mask=masks)

        # Perspective Transform
        left_margin = 140  # 조정된 왼쪽 여백
        top_margin = 200   # 조정된 위쪽 여백

        # 소스 포인트 조정
        src_point1 = [0, y]                # 왼쪽 아래
        src_point2 = [left_margin, top_margin]  # 왼쪽 위
        src_point3 = [x - left_margin, top_margin]  # 오른쪽 위
        src_point4 = [x, y]                # 오른쪽 아래

        src_points = np.float32([src_point1, src_point2, src_point3, src_point4])

        # 대상 포인트 조정
        dst_point1 = [x//6, y]              # 왼쪽 아래
        dst_point2 = [x//6, 0]              # 왼쪽 위
        dst_point3 = [x//6*5, 0]            # 오른쪽 위
        dst_point4 = [x//6*5, y]            # 오른쪽 아래

        dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_img = cv2.warpPerspective(filtered_img, matrix, [x, y])

        # 이미지 평행이동
        # translated_img = self.translate_image(warped_img, tx=90, ty=0)

        # 기존 HSV 방식에서 다시 살리기
        grayed_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        # 이미지 이진화
        bin_img = np.zeros_like(grayed_img)
        bin_img[grayed_img > 20] = 1

        # 슬라이딩 윈도우 차선 검출
        out_img, x_location, _ = self.slidewindow.slidewindow(bin_img, False)

        # 결과 표시
        cv2.imshow('Original Image', self.img)
        cv2.imshow("Yellow Mask", filtered_yellow)
        cv2.imshow("White Mask", filtered_white)
        cv2.imshow("Filtered Image", filtered_img)
        cv2.imshow("Warped Image", warped_img)
        cv2.imshow("Output Image", out_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def translate_image(self, image, tx, ty):
        """
        이미지 평행이동 함수
        :param image: 입력 이미지
        :param tx: x축 평행이동 거리
        :param ty: y축 평행이동 거리
        :return: 평행이동된 이미지
        """
        rows, cols = image.shape[:2]
        
        # 평행이동 행렬 생성
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 이미지 평행이동
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        
        return translated_image

    def create_trackbars(self):
        # 노란색 HSV 범위 트랙바
        cv2.createTrackbar("Yellow Lower H", "Trackbars", self.lower_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Lower S", "Trackbars", self.lower_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Lower V", "Trackbars", self.lower_yellow[2], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper H", "Trackbars", self.upper_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Upper S", "Trackbars", self.upper_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper V", "Trackbars", self.upper_yellow[2], 255, self.nothing)

        # 흰색 HSV 범위 트랙바
        cv2.createTrackbar("White Lower H", "Trackbars", self.lower_white[0], 180, self.nothing)
        cv2.createTrackbar("White Lower S", "Trackbars", self.lower_white[1], 255, self.nothing)
        cv2.createTrackbar("White Lower V", "Trackbars", self.lower_white[2], 255, self.nothing)
        cv2.createTrackbar("White Upper H", "Trackbars", self.upper_white[0], 180, self.nothing)
        cv2.createTrackbar("White Upper S", "Trackbars", self.upper_white[1], 255, self.nothing)
        cv2.createTrackbar("White Upper V", "Trackbars", self.upper_white[2], 255, self.nothing)

    def read_trackbar_values(self):
        self.lower_yellow[0] = cv2.getTrackbarPos("Yellow Lower H", "Trackbars")
        self.lower_yellow[1] = cv2.getTrackbarPos("Yellow Lower S", "Trackbars")
        self.lower_yellow[2] = cv2.getTrackbarPos("Yellow Lower V", "Trackbars")
        self.upper_yellow[0] = cv2.getTrackbarPos("Yellow Upper H", "Trackbars")
        self.upper_yellow[1] = cv2.getTrackbarPos("Yellow Upper S", "Trackbars")
        self.upper_yellow[2] = cv2.getTrackbarPos("Yellow Upper V", "Trackbars")

        self.lower_white[0] = cv2.getTrackbarPos("White Lower H", "Trackbars")
        self.lower_white[1] = cv2.getTrackbarPos("White Lower S", "Trackbars")
        self.lower_white[2] = cv2.getTrackbarPos("White Lower V", "Trackbars")
        self.upper_white[0] = cv2.getTrackbarPos("White Upper H", "Trackbars")
        self.upper_white[1] = cv2.getTrackbarPos("White Upper S", "Trackbars")
        self.upper_white[2] = cv2.getTrackbarPos("White Upper V", "Trackbars")

    def nothing(self, x):
        pass

if __name__ == "__main__":
    LaneDetection()

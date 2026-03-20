"""
original code from thohemp:
https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/utils.py
"""

import numpy as np
import torch
import scipy.io as sio
import cv2
from math import cos, sin

# .mat 파일 경로를 주면 pitch/yaw/roll 세 숫자를 꺼내 준다
def get_ypr_from_mat(mat_path):

    # mat: MATLAB이라는 수치계산 프로그램에서 쓰는 데이터 저장 형식
    mat = sio.loadmat(mat_path)

    # mat 안에 Pose_Para라는 이름의 데이터가 들어있음.
    # pre_pose_params = [pitch, yaw, roll, tdx, tdy, tdz, scale_factor]
    pre_pose_params = mat['Pose_Para'][0]

    # 이 데이터로부터 pitch, yaw, roll 순서로 추출, [:3]를 통해
    pose_params = pre_pose_params[:3]

    # return되는 결과 예시: array([0.1745, -0.5236, 0.0349]) (라디안)
    return pose_params

# .mat 파일을 열고, 'pt2d'라는 키 추출하기.
def get_pt2d_from_mat(mat_path):
    mat = sio.loadmat(mat_path)

    # pt2d는 2D landmark 좌표
    # 얼굴에서 눈 꼬리, 코 끝, 입 꼬리 같은 특징점들의 (x, y) 위치
    # array([[120, 145, 160, 175, ...],   ← 각 점의 x 좌표들
    #        [200, 198, 210, 220, ...]])  ← 각 점의 y 좌표들
    pt2d = mat['pt2d']

    return pt2d

    # 나중에 pt2d를 얼굴 랜드마크들의 최소값, 최대값을 구해 얼굴 영역 crop에 사용

    # x_min = min(pt2d[0, :])  # x좌표 중 극좌
    # x_max = max(pt2d[0, :])  # x좌표 중 극우
        # pt2d[0, :] 에서 min/max를 구하면 얼굴의 좌우 경계, 

    # y_min = min(pt2d[1, :])  # y좌표 중 극상
    # y_max = max(pt2d[1, :])  # y좌표 중 극하
        # pt2d[1, :] 에서 min/max를 구하면 얼굴의 상하 경계

# pitch/yaw/roll 세 각도(라디안) → 3×3 회전행렬
# 용도: 데이터셋에서 정답 레이블 만들 때
def get_R(x,y,z):
    # pitch만 표현하는 3×3 행렬
    Rx = np.array([[1, 0, 0], # x방향은 그대로
                   [0, np.cos(x), -np.sin(x)], # y방향은 cos/sin으로 변환
                   [0, np.sin(x), np.cos(x)]]) # z방향은 cos/sin으로 변환
    
    # yaw만 표현하는 3×3 행렬
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    
    # roll만 표현하는 3×3 행렬
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    
    # 세 행렬을 합쳐 하나로 만듦
    R = Rz.dot(Ry.dot(Rx))

    # 최종 3×3 회전행렬 반환
    # 이 함수의 반환값이 datasets.py에서 학습 데이터의 정답 레이블로 사용됌.
    # 모델이 학습할 때 "이 얼굴 이미지의 정답은 이 3×3 행렬이다"라고 제공되는 것이 바로 get_R의 출력.
    return R

# 모델이 출력한 숫자 6개 → 3×3 회전행렬
# 용도: 모델 예측값을 회전행렬로 변환할 때
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):

    # 왜 6개지? 아깐 9개인데?

    # 회전행렬은 아무 숫자 9개나 되는 게 아님.
    # 수학적으로 엄격한 조건을 만족해야 함.
    # 열벡터 세 개가 서로 수직이어야 하고, 각각의 길이가 정확히 1이어야 함.
    # 딥러닝 모델이 숫자 9개를 자유롭게 출력하면 이 조건이 깨져버림.
    # 그래서 6D 표현을 해결책으로 제시함.
    # 숫자 6개만 예측하게 하고, 나머지 1개 열은 수학적으로 자동 계산.
    # 6개만으로도 회전을 완전히 표현할 수 있거든

    x_raw = poses[:,0:3] # 첫 번째 열 후보 (3개)
    y_raw = poses[:,3:6] # 두 번째 열 후보 (3개)
    # 이 둘이 최종 행렬의 첫 번째, 두 번째 열
    # 세 번째 열은 이 둘로부터 계산


    x = normalize_vector(x_raw, use_gpu) # 길이를 1로 정규화
    z = cross_product(x,y_raw) # x와 y_raw의 외적 → 두 벡터에 수직인 벡터
    # 이 두 단계를 거치면 수학적 조건을 자동으로 만족하는 세 벡터 x, y, z가 완성됩니다.

    z = normalize_vector(z, use_gpu) # 길이를 1로 정규화
    y = cross_product(z,x) # 최종 y열 확정

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)

    matrix = torch.cat((x,y,z), 2) # 세 벡터를 열로 합쳐 3×3 행렬 완성

    return matrix

    # get_R과의 관계는?
    # get_R: pitch/yaw/roll(각도) → 3×3 행렬  ← 정답 레이블 만들 때
    # compute_rotation_matrix  : 6D 벡터(모델 출력)  → 3×3 행렬  ← 모델 예측할 때

# 3×3 회전행렬 → pitch/yaw/roll 세 각도(라디안)
# 용도: 최종 결과를 사람이 읽을 수 있는 각도로 변환할 때
def compute_euler_angles_from_rotation_matrices(rotation_matrices, use_gpu=True):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
    if use_gpu:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
    else:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3))  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler

# 벡터의 길이를 1로 만들기
# compute_rotation_matrix_from_ortho6d 내부에서 호출
def normalize_vector( v, use_gpu=True):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))  
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# 두 벡터에 동시에 수직인 새 벡터 계산
# compute_rotation_matrix_from_ortho6d 내부에서 호출
def cross_product(u, v):

    batch = u.shape[0]

    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)
        
    return out

# 예측된 각도로 얼굴 위에 3D 큐브를 그려서 시각화
# inference.py에서 결과 이미지 저장할 때 사용
def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y


    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

# 예측된 각도로 얼굴 위에 3D 축(x/y/z 화살표)을 그려서 시각화
# test.py에서 결과 확인할 때 사용
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

# .mat 파일 경로를 주면 pitch/yaw/roll/tdx/tdy 다섯 숫자를 꺼내 준다
# 근데 이거 만들어 놓고 안 쓰는 함수임 ㅋ
def get_pose_params_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]

    # tdx, tdy는 얼굴의 이미지 내 위치(translation)를 나타내는 값
    pose_params = pre_pose_params[:5]

    # 반환값: array([0.1745, -0.5236, 0.0349, 112.3, 98.7])
    return pose_params
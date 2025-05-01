#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

# ROS2 메시지 타입들
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped

# TF 퍼블리시
import tf2_ros

# 메시지 동기화 (ROS2용 message_filters)
import message_filters

import torch
import numpy as np
import math
import time

# --- 기존 코드 라이브러리 (프로젝트 내 제공 모듈)
# 전처리 모듈
from preprocessors.no import NoPreprocessing
from preprocessors.orientation import OrientationPreprocessing
from preprocessors.imu_offset import ImuOffestAndRotation

# UKF 관련 모듈 (추론/학습)
from filters.ukf_model_steper_inference import UKFModelStepperInference
from filters.ukf_model_steper_training import UKFModelStepperTrain

# Solver 및 WANDB 모델 로딩 관련 유틸들
from utils.solver_settings import get_solver_settings
from utils.wanb_model_getter import get_wandb_model

# ----------------------------------------------------------------------
# 전역 설정 (WANDB 모델 로딩 및 UKF 관련 설정)
# ----------------------------------------------------------------------
# 훈련된 모델 경로 등은 실제 환경에 맞게 수정
model_dict = {
    'model': '/home/a/Learning-dynamics-models-for-velocity-estimation/code/trained_models/ukf_2025_04_04_13_19_08',
}
pred_model_name = model_dict['model']

# WANDB로부터 모델 및 noise 모델, 그리고 추가 파라미터(args) 로딩 (내부에 device, precision, 등 포함)
args, pred_model, pred_noise_model = get_wandb_model(pred_model_name)

# Noise 모델 확인 (옵션)
Q, R, P = pred_noise_model(torch.zeros(1, 5))
print(f'pred noise model')
print(f"Q: {torch.diag(Q[0])}")
print(f"R: {torch.diag(R[0])}")
print(f"P: {torch.diag(P[0])}")

device = torch.device(args.common_device)

# 상태 및 제어 차원 (기존 코드와 동일)
state_dim = 5
control_dim = 2

# UKF solver 설정
solver_settings = get_solver_settings(args)

# UKF 추론 노드 초기화 (추론 모듈)
ukf_single_stepper = UKFModelStepperInference(
    pred_model, 
    pred_noise_model,
    dt=0.01,  # 센서 주기에 맞게 dt (또는 실제 dt 이용)
    solver_settings=solver_settings,
    q_entr_lb=args.ukf_q_entropy_lb,
    device=device
)
ukf_stepper = UKFModelStepperTrain(ukf_model_steper=ukf_single_stepper)

# UKF 관련 파라미터
args.ukf_start_loss = 200  # 이 값은 기존 손실 계산 시점에 사용되지만, 여기서는 참고용
# state_weights 등도 기존 로스 계산 시 사용되나, real-time TF 퍼블리시에서는 크게 사용하지 않음

# ----------------------------------------------------------------------
# ROS2 실시간 UKF 노드 (실시간 센서 수신, 전처리, UKF 추론, TF 퍼블리시)
# ----------------------------------------------------------------------
class RealTimeUKFNode(Node):
    def __init__(self):
        super().__init__('realtime_ukf_node')

        self.get_logger().info("Initializing RealTimeUKFNode...")

        # TF 브로드캐스터 초기화 (TF 퍼블리시용)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Preprocessing 모듈 초기화
        self.no_preprocessor = NoPreprocessing()
        self.orientation_preprocessor = OrientationPreprocessing()
        self.imu_offset_preprocessor = ImuOffestAndRotation()

        # 실시간 데이터를 누적할 시퀀스 버퍼 (슬라이딩 윈도우)
        self.seq_buffer = []  
        self.sequence_length = args.ukf_test_sequence_length  # 예: 50 (실시간 추론 시 사용 길이)

        # ROS2 센서 토픽 구독 (message_filters 사용)
        # 토픽명은 실제 환경에 맞게 수정하시기 바랍니다.
        imu_sub = message_filters.Subscriber(self, Imu, '/imu')
        odom_sub = message_filters.Subscriber(self, Odometry, '/vesc/odom')
        pose_sub = message_filters.Subscriber(self, PoseStamped, '/opitrack/rigid_body_0')

        # ApproximateTimeSynchronizer: 큐 사이즈와 슬롭 값은 환경에 맞게 조정
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [imu_sub, odom_sub, pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sensor_callback)
        self.get_logger().info("Subscribers and message_filters initialized.")

        # UKF 모델 stepper (전역 ukf_stepper를 사용)
        self.ukf_stepper = ukf_stepper

    def sensor_callback(self, imu_msg, odom_msg, pose_msg):
        """
        동기화된 센서 메시지를 받아 전처리 후 feature 벡터를 구성하고,
        슬라이딩 윈도우 방식으로 UKF 추론을 실행한 후, TF 퍼블리시를 진행.
        """
        # --- 1. Feature 벡터 구성 ---
        # 예시: ground truth state는 Odometry 사용 (실제 환경에선 Optitrack 등 활용 가능)
        # ground truth state (5 dims): [pos.x, pos.y, twist.linear.x, twist.linear.x, orientation.z (대략 yaw)]
        state = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.x,  # wheel_speed: 중복 사용 (원 코드에서 4번째 값을 wheel_speed로 사용)
            odom_msg.pose.pose.orientation.z  # 간단히 yaw 대용 (정확한 yaw 계산은 quaternion->euler 필요)
        ]
        # control input (2 dims): [twist.linear.x, twist.angular.z]
        control = [
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.angular.z
        ]
        # IMU 데이터 (3 dims): 선형 가속도
        imu_data = [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ]
        # 최종 feature vector: 총 5 + 2 + 3 = 10 dims
        feature_vector = state + control + imu_data

        # --- 2. 전처리 적용 ---
        # torch tensor로 변환 (device 적용)
        x_vec = torch.tensor(feature_vector, dtype=torch.float32, device=device)
        # 각 전처리 모듈 적용 (필요에 따라 함수 내부 구현이 다를 수 있음)
        x_vec = self.no_preprocessor(x_vec)
        x_vec = self.orientation_preprocessor(x_vec)
        x_vec = self.imu_offset_preprocessor(x_vec)
        # x_vec shape: (10,)

        # 누적 버퍼에 저장 (각 항목은 (1, feature_dim))
        self.seq_buffer.append(x_vec.unsqueeze(0))
        # 버퍼가 충분히 쌓이면 UKF 추론 실행 (슬라이딩 윈도우 방식)
        if len(self.seq_buffer) >= self.sequence_length:
            # 최근 sequence_length 개 항목을 사용하여 시퀀스 텐서 구성: shape (1, sequence_length, 10)
            seq_tensor = torch.cat(self.seq_buffer[-self.sequence_length:], dim=0).unsqueeze(0)
            # --- 3. UKF 추론을 위한 데이터 구성 ---
            # 초기 상태 (X0): 시퀀스의 첫 타임스텝에서 ground truth 상태 (첫 5개 값)
            X0 = seq_tensor[:, 0, :state_dim]
            # 제어 입력 (u): 시퀀스의 각 타임스텝에 대해, 5번째부터 (5~6) 값
            u = seq_tensor[:, :, state_dim: state_dim+control_dim]
            # IMU 관측 데이터: 시퀀스의 마지막 3개 값
            imu_seq = seq_tensor[:, :, -3:]
            # wheel_speed: ground truth 상태의 4번째 값 (index 3); shape를 맞추기 위해 unsqueeze
            wheel_speed = seq_tensor[:, :, 3].unsqueeze(-1)
            # 관측 벡터 y: IMU 데이터와 wheel_speed를 결합 (3+1=4 dims)
            y = torch.cat((imu_seq, wheel_speed), dim=-1)
            
            # --- 4. UKF 모델 추론 실행 ---
            X_ukf, P, q_entropy, r_entropy = self.ukf_stepper(X0, u, y)
            # 최종 추정 상태: 시퀀스의 마지막 타임스텝 추정값, shape: (1, state_dim)
            est_state = X_ukf[0, -1, :].detach().cpu().numpy()
            
            # --- 5. TF 퍼블리시 (추정된 상태를 사용하여) ---
            self.publish_tf(est_state)
            self.get_logger().info(f"UKF Estimated State: {est_state}")

    def publish_tf(self, state):
        """
        추정된 상태(state)를 사용하여 TF 변환 메시지를 생성하고 퍼블리시함.
        state: numpy array, shape (state_dim,)  
               (예: [x, y, _, _, yaw], 여기서 yaw는 대략 orientation.z)
        """
        x_est = float(state[0])
        y_est = float(state[1])
        # 여기서는 추정 상태의 5번째 값(index 4)을 yaw로 사용 (실제 환경에 따라 수정 필요)
        yaw_est = float(state[4])
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = x_est
        t.transform.translation.y = y_est
        t.transform.translation.z = 0.0

        # 오일러 각 -> 쿼터니언 변환
        q = self.euler_to_quaternion(0.0, 0.0, yaw_est)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """
        오일러 각(roll, pitch, yaw)을 쿼터니언으로 변환.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return (x, y, z, w)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeUKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealTimeUKFNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

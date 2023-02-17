# https://docs.python-requests.org/zh_CN/latest/
# 收发请求控制小车

"""
结合evaluate_process将训练好的模型参数计算出请求发送数值  建议先简单计算模拟小车与实际小车的速度比
"""
import requests
import time

r = requests.get('http://192.168.4.1/read_speed')
print(r.text)
r = requests.get('http://192.168.4.1/read_angle')   #获取实际小车参数
print(r.text)                                   

r = requests.post('http://192.168.4.1/motor_control?speed={speed_final}')
r = requests.post('http://192.168.4.1/servo_control?angle={angel_final}')


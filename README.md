# robotics_ball_bot

### To run on Nano :
```
ssh shanshah@10.0.0.119
cd robotics_ball_bot
python3 run_robot.py True
```

### To run over LAN:
First on the jetson Nano do the following
```
ssh shanshah@10.0.0.119 # SSH into jetson
cd CortexNanoBridge/jetson_nano/cortano
python3 worker.py
```
On your local macbook 
```python3 run_robot.py False```

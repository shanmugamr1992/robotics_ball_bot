# robotics_ball_bot

WORKING VIDEO : https://drive.google.com/drive/folders/1yc_6RI3hc5NVKJ7r_hlfpTO2b2KDAvhl?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto 

### To run on Nano :
```
ssh shanshah@10.0.0.119
cd robotics_ball_bot
python3 run_robot.py True
```
To kill the process
```
ps aux | grep python3 | grep run_robot
Note the second column number (PID)
sudo kill -9 <PID-HERE>
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

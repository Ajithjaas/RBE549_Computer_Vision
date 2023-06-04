# Sexy Semantic Mapping - RBE549 Homework 2

The following commands have to be run in the terminal to install all the libraries required to run the code.
```json
sudo pip install glob2
sudo pip install mxnet
sudo pip install gluoncv
sudo pip install open3d
```



## Run the following code in the terminal to initiate Point Cloud painting and ICP:

Next we have to go the below location in the terminal:
```json
.../ajayamoorthy_hw2/Code/
```

For point cloud painting using Segmented image:
```json
python main.py --flag = 0
```
For point cloud painting using RGB image:
```json
python main.py --flag = 1
```

## To visualize the point clouds results
Next we have to go the below location in the terminal:
```json
.../ajayamoorthy_hw2/Outputs/
```
Open the visualize.py file and replace the file name you want to open. Then run the following code in the terminal:

For point cloud painting using Segmented image:
```json
python visualize.py
```

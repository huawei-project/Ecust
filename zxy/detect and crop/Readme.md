### 0_detect_only.py

- 功能：只作为检测使用，输出**总共图片数目**以及**被成功检测的图片的数目**
- 使用：
  - 在main函数中，设置好图片文件夹的目录（**rootdir**） 
  - 检测室内图片，运行 **detect_RGB(indoors=True)**  **detect_Multi(indoors=True)**
  - 检测室外图片，运行**detect_RGB(indoors=False)**  **detect_Multi(indoors=False)**
- 注意：
  - 如果数据存储的目录结构发生改变，在程序读取文件时的代码也需要进行相应的改变

### 1_detect_and_savetxt.py

- 功能：
  - 作为检测使用，输出**总共图片数目**以及**被成功检测的图片的数目**
  - 对于RGB图片，保存所有未被检测出来的图片的路径储存在**failedrgb.txt**；对于Multi图片，保存所有未被检测出来的图片的路径储存在**failemultifile.txt**，把25个通道的图片看成一张图片，未被检测出来的图片的路径储存在**failemultdir.txt**
  - 对于RGB图片，把每张图片的检测情况以（*imagepath: socre, bbox, landmark*）的形式储存在**rgbdetect.txt**；对于Multi图片，把每张图片的检测情况以（*imgdir: socre, bbox, landmark*）的形式储存在**multidetect.txt**。对于未检测出来的图片，会以（*imgdir: None, None, None*）的形式储存

- 使用：
  - 在main函数中，设置好图片文件夹的目录（**rootdir**） 
  - 检测室内图片，运行 **detect_RGB(indoors=True)**  **detect_Multi(indoors=True)**
  - 检测室外图片，运行**detect_RGB(indoors=False)**  **detect_Multi(indoors=False)**
- 注意：
  - 如果数据存储的目录结构发生改变，在程序读取文件时的代码也需要进行相应的改变

### 2_mark_manually.py

- 功能：
  - 对1中没有检测出来的图片的 bbox和landmark进行手工标注
  - 需要手工剪裁的图片的路径会相应储存在**rgbmanual.txt** 和**multimanual.txt**
  - 对于1中未检测出来的图片，把手工标注的score, bbox和landmark代替之前的None, None, None储存在**rgbdetect.txt**和**multidetect.txt**中（其中score设置为1）
- 使用：
  - 在main函数中，设置好图片文件夹的目录（**rootdir**） 并运行程序
  - 裁剪时，每次回调出一张未成功剪裁的图片，需要依次鼠标左键双击图片回归框的左上角，回归框的右下角，人脸的左眼，人脸的右脸，人脸的鼻尖，人脸的左端嘴唇，人脸的右端嘴唇从而获得7个坐标点。相应的坐标也会在终端显示。标注完一张图片可按回车对下一张图片进行标记。
- 注意：如果在标注中有标注错误，可以在出错之后把剩下的坐标点全部点完，然后重新从回归框的左上角开始对这张图片进行标注，全部标注完之后按回车

### 3_crop.py

- 功能：对图片进行人脸剪裁，同时把尺寸改成(112, 96)

- 使用：在main函数中，设置好图片文件夹的目录（**rootdir**） 以及剪裁完的文件需要保存的目录（**SAVEPATH**）并运行程序

- 注意：

  - 如果需要调整输出的尺寸，需要修改（112，96）
  - 如果裁剪的出来的图片的顶部或者底部会有黑边，需要调整程序最开始的**ALIGNED**

  
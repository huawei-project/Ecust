# Reproduce Mobilefacenet

## Prepare data
### 检测
1. 下载CASIA数据集`CASIA_WebFace`，解压到`data/CASIA-WebFace`
2. 标签文件`data/CASIA_labels.txt`
3. 运行`prepare_data/detect.py`，检测结果保存在`data/CASIA_detect.txt`
4. 运行`prepare_data/crop.py`，剪裁结果保存在`data/CASIA-WebFace-Unaligned`，对齐后数据保存在`data/CASIA-WebFace-Aligned`


### 剪裁与对齐
注意，利用关键点计算变换矩阵$M_{2\times3}$，再调用函数`cv2.warpAffine`对齐

1. 变换矩阵$M$的求解
    例如现有$n$个关键点
    $$
    xy = \left[\begin{matrix}
        x_1 & y_1 \\
        x_2 & y_2 \\
        ... & ... \\
        x_n & y_n \\
    \end{matrix}\right]
    $$

    希望对齐后的坐标点为
    $$
    \hat{xy} = \left[\begin{matrix}
        \hat{x_1} & \hat{y_1} \\
        \hat{x_2} & \hat{y_2} \\
        ...  & ...  \\
        \hat{x_n} & \hat{y_n} \\
    \end{matrix}\right]
    $$

    构造矩阵
    $$
    X_{2n\times4} = \left[\begin{matrix}
        \vec{x} &  \vec{y} & \vec{1} & \vec{0} \\
        \vec{y} & -\vec{x} & \vec{0} & \vec{1}
    \end{matrix}\right]
    $$

    $$
    b_{2n} = \left[\begin{matrix}
        \hat{x_1} & \hat{x_2} & \cdots & \hat{x_n} &
        \hat{y_1} & \hat{y_2} & \cdots & \hat{y_n}
    \end{matrix}\right]^T
    $$

    其中
    $$
    \vec{x} = \left[\begin{matrix}
        x_1 & x_2 & \cdots & x_n
    \end{matrix}\right]^T
    $$

    $$
    \vec{y} = \left[\begin{matrix}
        y_1 & y_2 & \cdots & y_n
    \end{matrix}\right]^T
    $$

    $$
    \vec{1} = \left[\begin{matrix}
        1 & 1 & \cdots & 1
    \end{matrix}\right]^T
    $$

    $$
    \vec{0} = \left[\begin{matrix}
        0 & 0 & \cdots & 0
    \end{matrix}\right]^T
    $$

    求解下式解向量$r_{4\times1}$
    $$
    X \cdot r = b
    $$

    注意增广矩阵的秩
    $$ \text{rank}(X) < rank([X | b]) $$

    上式无解，可使用伪逆求解
    $$
    r = (X^T X + \lambda I)^{-1} X^T b
    $$

    构造矩阵
    $$
    R = \left[\begin{matrix}
        r_1 & -r_2 & 0 \\
        r_2 &  r_1 & 0 \\
        r_3 & -r_4 & 1 \\
    \end{matrix}\right]
    $$

    则变换矩阵$M$可由下式求解
    $$
    \left[\begin{matrix}
        M^T & \begin{matrix}
            0 \\ 0 \\ 1
        \end{matrix}
    \end{matrix}\right] = R^{-1}
    $$

2. 坐标变换
    $$
    M = \left[\begin{matrix}
        m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23}
    \end{matrix}\right]
    $$

    对于坐标$(x, y)$，其变换后的坐标$(\hat{x}, \hat{y})$为
    $$
    \left[\begin{matrix}
        \hat{x} \\ \hat{y} \\
    \end{matrix}\right]
    = M \left[\begin{matrix}
        x \\ y \\ 1
    \end{matrix}\right]
    $$
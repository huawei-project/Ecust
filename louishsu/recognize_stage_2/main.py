# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-13 13:12:18
@LastEditTime: 2019-08-13 13:17:08
@Update: 
'''
from train import train
from test  import test
from config import configer

def main():
    train(configer)
    test(configer)

if __name__ == "__main__":
    main()
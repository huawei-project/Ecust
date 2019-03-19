from train import train
from test import test

def main():
    mode = input("please input 'train' or 'test': ")
    if mode == 'train':
        train()
    elif mode == 'test':
        test()

if __name__ == "__main__":
    main()
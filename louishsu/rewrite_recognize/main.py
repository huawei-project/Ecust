from train import train
from test  import test
from config import configer

def main():
    train(configer)
    test(configer)

if __name__ == "__main__":
    main()
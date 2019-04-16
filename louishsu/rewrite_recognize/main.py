from train import train
from test  import test
from gen_excel import gen_out_excel
from config import configer

def main():
    train(configer)
    test(configer)
    gen_out_excel(configer)

if __name__ == "__main__":
    main()
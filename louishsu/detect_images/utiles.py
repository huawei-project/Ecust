import time

getTime     = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
getVol      = lambda subidx: (subidx - 1) // 10 + 1
getWavelen  = lambda path: int(path.split('.')[0].split('_')[-1])
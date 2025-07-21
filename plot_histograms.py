import glob
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob('output/hist_*.csv'))
for f in files:
    df = pd.read_csv(f, names=['intensity','count'])
    plt.figure()
    plt.bar(df['intensity'], df['count'])
    plt.title(f)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(f.replace('.csv','.png'))
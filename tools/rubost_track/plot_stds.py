import matplotlib.pyplot as plt
import numpy as np

def main():
    stds=np.loadtxt('/home/rislab/Workspace/pysot/rb_result/stds/stds.txt', delimiter=',')
    stds=stds[2:250,:]
    stds[:,1] = stds[:,1]/12.5
    plt.plot(stds[:,0].astype(int),stds[:,1])
    plt.ylabel(r'$\tau$')
    plt.xlabel('Frame #')
    fig = plt.gcf()
    fig.set_size_inches(6.7, 3.8)

    plt.savefig('/home/rislab/Workspace/pysot/rb_result/stds/stds'+'.png', dpi=300,
                            bbox_inches='tight')

    # plt.show()

if __name__ == '__main__':
    main()
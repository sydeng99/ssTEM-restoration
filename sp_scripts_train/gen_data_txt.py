import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help='path to training data')
parser.add_argument('-n', '--num', type=int, default=4000)
args = parser.parse_args()

txt=open(args.folder+'/sp_train_data.txt', 'w')
for i in range(1, int(args.num)+1):
    txt.write(str(i).zfill(4)+'_1.png' + ' ' +
                str(i).zfill(4)+'_2.png' + ' ' +
                str(i).zfill(4)+'_2_degra.png' + ' ' +
                str(i).zfill(4)+'_3.png' + ' ' +
                str(i).zfill(4)+'_3_degra.png' + ' ' +
                str(i).zfill(4)+'_4.png' + ' ' +
                
                str(i).zfill(4)+'_2_degra_maska.png' + ' ' +
                str(i).zfill(4)+'_3_degra_maska.png' + ' ' +

                str(i).zfill(4)+'_2_degra_maskb.png' + ' ' +
                str(i).zfill(4)+'_3_degra_maskb.png' + ' ' +

                str(i).zfill(4)+'_2_interp.png' + ' ' +
                str(i).zfill(4)+'_3_interp.png' + ' ' +
                '\n')
txt.close()
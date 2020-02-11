import os
import pickle
import argparse
import pandas as pd
import dynet as dy
from tqdm import tqdm

from cnn_text_classification_project.utils import build_dataset_1 , associate_parameters , binary_pred , forwards

parser = argparse.ArgumentParser (description='Convolutional Neural Networks for Sentence Classification in DyNet')
parser.add_argument ('--gpu' , type=int , default=-1 , help='GPU ID to use. For cpu, set -1 [default: -1]')
parser.add_argument ('--model_file' , type=str , default='../data/models/cnn_text_classification/model_e10' ,
                      help='Model to use for prediction [default: ./model]')
parser.add_argument ('--input_file' , type=str ,
                      default='D:/Users/wissam/Documents/These/these/Datasets/Paraphrases_sentiment_datasets/news_headlines_paraphrases_sentiment.csv' ,
                      help='Input file path [default: ./data/annotation_S2]')
parser.add_argument ('--out_file' , type=str , default='../data/results/sentiment_dataset_cnn.csv' ,
                      help='Output file path [default: ./pred_yannotation_S2.txt]')
parser.add_argument ('--w2i_file' , type=str , default='../data/models/cnn_text_classification/w2i.dump' ,
                      help='Word2Index file path [default: ../data/models/cnn_text_classification/w2i.dump]')
parser.add_argument ('--i2w_file' , type=str , default='../data/models/cnn_text_classification/i2w.dump' ,
                      help='Index2Word file path [default: ../data/models/cnn_text_classification/i2w.dump]')
parser.add_argument ('--alloc_mem' , type=int , default=1024 ,
                      help='Amount of memory to allocate [mb] [default: 1024]')

args = parser.parse_args ()

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str (args.gpu)

    MODEL_FILE = args.model_file
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.out_file
    W2I_FILE = args.w2i_file
    I2W_FILE = args.i2w_file
    ALLOC_MEM = args.alloc_mem
    df_input = pd.read_csv (INPUT_FILE , sep=";" , encoding="ISO-8859-1")
    df_input.columns = ["Id" , "Review" , "Golden"]
    # DyNet setting
    dyparams = dy.DynetParams ()
    dyparams.set_mem (ALLOC_MEM)
    dyparams.init ()
    print (MODEL_FILE)
    # Load model
    model = dy.Model ()
    pretrained_model = dy.load (MODEL_FILE , model)




    if len (pretrained_model) == 3:
        V1 , layers = pretrained_model[0] , pretrained_model[1:]
        MULTICHANNEL = False
    else:
        V1 , V2 , layers = pretrained_model[0] , pretrained_model[1] , pretrained_model[2:]
        MULTICHANNEL = True

    EMB_DIM = V1.shape ()[0]
    WIN_SIZES = layers[0].win_sizes

    # Load test data
    with open (W2I_FILE , 'rb') as f_w2i , open (I2W_FILE , 'rb') as f_i2w:
        w2i = pickle.load (f_w2i)
        i2w = pickle.load (f_i2w)

    max_win = max (WIN_SIZES)
    test_X = build_dataset_1 (INPUT_FILE , w2i=w2i , unksym='unk')
    # test_X , Golden= build_dataset_df(INPUT_FILE)
    print (test_X)
    test_X = [[0] * max_win + instance_x + [0] * max_win for instance_x in test_X]

    # Pred
    pred_y = []
    for instance_x in tqdm (test_X):
        # Create a new computation graph
        dy.renew_cg ()
        associate_parameters (layers)

        sen_len = len (instance_x)

        if MULTICHANNEL:
            x_embs1 = dy.concatenate ([dy.lookup (V1 , x_t , update=False) for x_t in instance_x] , d=1)
            x_embs2 = dy.concatenate ([dy.lookup (V2 , x_t , update=False) for x_t in instance_x] , d=1)
            x_embs1 = dy.transpose (x_embs1)
            x_embs2 = dy.transpose (x_embs2)
            x_embs = dy.concatenate ([x_embs1 , x_embs2] , d=2)
        else:
            x_embs = dy.concatenate ([dy.lookup (V1 , x_t , update=False) for x_t in instance_x] , d=1)
            x_embs = dy.transpose (x_embs)
            # x_embs = dy.reshape(x_embs, (sen_len, EMB_DIM, 1))

        y = forwards (layers , x_embs)
        pred_y.append (str (float ( (y.value ()))))
    pred_y = pred_y[1:]
    P = pd.Series (pred_y).rename ("Pred_sentiment")
    print (P , df_input["Review"])
    df = pd.DataFrame ()
    df["Id"] = df_input["Id"]
    df["Review"] = df_input["Review"]
    df["Golden"] = df_input["Golden"]
    df["Pred_sentiment"] = P
    """
    df= pd.DataFrame({"Id":df_input["Id"] },
                     {"Review": df_input["Golden"] },
                     {"Pred_sentiment": P })
                     #{"Pred_sentiment": P} )"""
    print(df)
    df.to_csv (OUTPUT_FILE , sep=";" , index=False)

    # with open(OUTPUT_FILE, 'w') as f:
    # f.write('\n'.join(pred_y))


if __name__ == '__main__':
    main (args)

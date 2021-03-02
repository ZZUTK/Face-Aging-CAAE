import tensorflow as tf
from FaceAging import FaceAging
from os import environ
import argparse

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='None', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
FLAGS = parser.parse_args()


def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = FaceAging(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset  # name of the dataset in the folder ./data
        )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            if not FLAGS.use_trained_model:
                print('\n\tPre-train the network')
                model.train(
                    num_epochs=10,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model,
                    weigts=(0, 0, 0)
                )
                print('\n\tPre-train is done! The training will start.')
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=FLAGS.use_init_model
            )
        else:
            print('\n\tTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()


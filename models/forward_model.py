from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow import keras
import numpy as np
import baseline.config as config
import cv2

# Set memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logic_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs")
        print(len(logic_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



class ForwardModel():
    """
    This class uses a fully connected network trained with the
    actions given in input.

    Parameters
    ----------     

        latent_dim : int
            number of dimensions of the latent space

    Attributes & Methods
    ----------
        abstractor : function
            VAEncoder
        
        latent_dim:
            dimension of the latent space
        
        units_layer_1:
        
        action_size:
        
        forward:
            Fully connected Forward model (input: (s+a) --> ŝ')

        train:
            function to train forward model

       
    """
    def __init__(self, abstractor, units_layer_1 = 256, action_size =4,
                retrain=False):

        # TO-DO : ADD PRE-TRAINED OPTIONS #
        if False:
            print("!!!")
        # if config.abst['pre_trained_vae'] and not retrain:
        #     # load a pre-trained auto-encoder
        #     self.encoder = keras.models.load_model('trained_encoder')
        #     self.decoder = keras.models.load_model('trained_decoder')
        
        else:
            
            self.abstractor = abstractor 
            self.latent_dim = self.abstractor.latent_dim
            self.units_layer_1 = units_layer_1
            self.action_size = action_size

            input_shape = (self.latent_dim + self.action_size, )
            
            
            # forward model
            # build forward model
            inputs = Input(shape=input_shape, name='forward_model_input')  
            x = Dense(self.units_layer_1, activation='relu')(inputs)
            outputs = Dense(self.latent_dim, activation='relu')(x)

            # instantiate encoder model
            self.forward = Model(inputs, outputs, name='forward')
            self.forward.summary()

            self.forward.compile(loss ='mse', optimizer='adam')
            self.forward.summary()

    def train(self, actions):

        pre_images = [actions[i][0] for i in range(len(actions))]
        actions_list = [actions[i][1] for i in range(len(actions))]
        post_images = [actions[i][2] for i in range(len(actions))]

        #flaten action array
        actions_list = [np.reshape(action,[-1,self.action_size]) for action in actions_list]

        #get the right shape: (self.action_size, )
        actions_list = [action[0] for action in actions_list]


 
        pre_bin_images = self.abstractor.get_binary_images(pre_images)
        post_bin_images = self.abstractor.get_binary_images(post_images)

        pre_abs = [self.abstractor.get_abstraction_from_binary_image(image)[-1] for image in pre_bin_images]
        post_abs = [self.abstractor.get_abstraction_from_binary_image(image)[-1] for image in post_bin_images]

        #get the right shape: (self.latent_dim, )
        pre_abs = [abstraction[0] for abstraction in pre_abs]
        post_abs = [abstraction[0] for abstraction in post_abs]

        # IL FAUT CONSTRUIRE L'INPUT  ET RÉFLÉCHIR À NORMALISER LES INPUTS ET OUTPUTS
        #   --> PRINTER un élément de pre abs et voiir comment créer la liste concaténée avec celles des actions. (zip ? )


        #Concatenate pre_abs and actions_list to build forward model's input
        forward_inputs = [np.concatenate(action_abstract, axis = 0) for action_abstract in zip(actions_list, pre_abs)]

        x_train = forward_inputs[:int(np.floor(len(forward_inputs) * 0.80))]
        x_test = forward_inputs[int(np.ceil(len(forward_inputs) * 0.80)):]

        y_train = post_abs[:int(np.floor(len(post_abs) * 0.80))]
        y_test = post_abs[int(np.ceil(len(post_abs) * 0.80)):]

        x_train = np.reshape(x_train, [-1, self.latent_dim + self.action_size])
        x_test = np.reshape(x_test, [-1, self.latent_dim + self.action_size])

        y_train = np.reshape(y_train, [-1, self.latent_dim])
        y_test = np.reshape(y_test, [-1, self.latent_dim])

        batch_size = 128
        epochs = 30

        self.forward.fit(x_train,y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, y_test))
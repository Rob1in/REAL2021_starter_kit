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



class CuriousExplorer():
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
    def __init__(self, action_space, abstractor,
                forward, num_hidden_units = 48,
                retrain=False):

        # TO-DO : ADD PRE-TRAINED OPTIONS #
        if False:
            print("!!!")
        # if config.abst['pre_trained_vae'] and not retrain:
        #     # load a pre-trained auto-encoder
        #     self.encoder = keras.models.load_model('trained_encoder')
        #     self.decoder = keras.models.load_model('trained_decoder')
        
        else:
            self.action_space = action_space

            self.abstractor = abstractor
            
            self.forward = forward 
            self.latent_dim = self.abstractor.latent_dim
            self.action_size = self.forward.action_size

            self.num_hidden_units = num_hidden_units
            self.explorer = LSTM_A2C(num_actions = self.action_size, num_hidden_units = self.num_hidden_units)
            self.curious_reward = [0]

            self.count_time_step_episode = 0 #set initial state back to 0 when 100 actions have been  outputed

            self.allCuriousRewards = [] 

    def selectNextAction(self, actions):
        #take first random action
        print(f"____ PROPOSE ACTION AT TIME STEP {len(actions)}")
        print(len(actions))
        inputs = self.get_inputs(actions)
        mus, variances, value = self.explorer(inputs, initial = False)
        print(f"mus is {mus}")
        print(type(mus))
        print(f"variances is {variances}")
        print(f"value is {value}")
        action = [np.random.normal(mu, var, 1) for mu, var in zip(mus[0], variances[0])]
        print(type(action))
        print(action)
        action = np.clip(action, -0.25, 0.25)
        action = np.reshape(action, (2,2))
        print(type(action))
        print(action.shape)
        print(action)

        return action, 'curious'

    def get_inputs(self, actions):

        """
        returns:
         array (None, 12) by concatening current abstraction (7,) + last action (4,) + last_curious_reward (1,)
        """

        current_state_abstraction = self.get_state_abstraction(actions, time = 'current')
        print(f"current_state_abstraction's shape is : {current_state_abstraction.shape}")
        last_action = self.get_last_action(actions)
        print(f"last_action's shape is : {last_action.shape}")
        last_reward = self.get_curious_reward(actions)
        #get right shape ( () --> (1,) )
        last_reward = np.reshape(last_reward, [1,-1])[0]
        print(f"last_reward's shape is : {last_reward.shape}")
        explorer_input = np.concatenate((current_state_abstraction, last_action, last_reward), axis = 0)
        #get the right shape for prediction ( (12, )  --> (1. 1, 12))
        explorer_input = np.expand_dims(explorer_input, axis = 0)
        explorer_input = np.expand_dims(explorer_input, axis = 0)
        
        return explorer_input

    
    def get_state_abstraction(self, actions, time):
        """
        pre_image_t is post_image_t-1
        
        returns : 
        <class 'numpy.ndarray'>
        (7,)

        """
        if time == 'current':
            post_images = [actions[i][2] for i in range(len(actions))]
            last_bin_image = self.abstractor.get_binary_last_image(post_images)
            #get the right shape: (self.latent_dim, )
            last_abstraction = self.abstractor.get_abstraction_from_binary_image(last_bin_image)[-1][0]
            return last_abstraction

        if time == 'last_initial':
            pre_images = [actions[i][0] for i in range(len(actions))]
            last_bin_image = self.abstractor.get_binary_last_image(pre_images)
            #get the right shape: (self.latent_dim, )
            last_abstraction = self.abstractor.get_abstraction_from_binary_image(last_bin_image)[-1][0]
            return last_abstraction

    def get_last_action(self, actions):
        """
        
        returns : 
        <class 'numpy.ndarray'>
        (self.action_size,)

        """
        last_action = actions[-1][1]

        #flaten action array
        last_action = np.reshape(last_action,[-1,self.action_size])

        #get the right shape: (self.action_size, )
        last_action = last_action[0]

        return last_action

    def get_curious_reward(self, actions, store = True):

        '''
        calculate curious reward with Pathak et al. ICM
        store it and return it
        '''
        last_action = self.get_last_action(actions)
        last_pre_abstraction = self.get_state_abstraction(actions, time = 'last_initial')
        last_post_abstraction = self.get_state_abstraction(actions, time = 'current')

        forward_input = np.concatenate((last_action, last_pre_abstraction), axis = 0)
        forward_input = np.reshape(forward_input, [-1, self.latent_dim + self.action_size])
        predicted_current_state_abstraction = self.forward.forward(forward_input)
        predicted_current_state_abstraction = predicted_current_state_abstraction[0]   

        curiosity_reward = np.linalg.norm(last_post_abstraction - predicted_current_state_abstraction)
        if store:
            self.allCuriousRewards += curiosity_reward

        return curiosity_reward



    @tf.function
    def train_step(
        model: tf.keras.Model, 
        optimizer: tf.keras.optimizers.Optimizer, 
        nTrials: tf.Tensor,
        gamma: float) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, episode_entropy, values, rewards, _, _ = run_episode(model, nTrials) #don't care about number of each actions and probability environment

            # Calculate expected returns
            #returns = tf.expand_dims(helper.get_n_step_return(rewards, values, gamma),1)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values = [
            tf.expand_dims(x, 1) for x in [action_probs, values]] 

            # Calculating loss values to update our network
            loss, actor_loss, critic_loss, entropy_reg = compute_loss(
            action_probs, 
            episode_entropy, 
            values, 
            rewards, 
            nTrials, 
            gamma,
            beta_v,
            beta_e)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))



class LSTM_A2C(tf.keras.Model):

    def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
        super().__init__()
        
        self.initialCell = tf.Variable(
                initial_value=tf.zeros((1,num_hidden_units)),
                trainable=True,
                name="initialCell"
                )
        self.initialHidden = tf.Variable(
                initial_value=tf.zeros((1,num_hidden_units)),
                trainable=True,
                name="initialHidden"
                )
        self.LSTM = tf.keras.layers.LSTM(
                num_hidden_units, 
                stateful=True,
                batch_input_shape=(1,1,12),
                name="lstm"
                )
        self.mu = tf.keras.layers.Dense(num_actions, name="mus_actor")
        self.var = tf.keras.layers.Dense(num_actions, activation = 'softplus', name="var_actor")
        self.value = tf.keras.layers.Dense(1, name="critic")

    def call(self, inputs: tf.Tensor, initial: bool):
        if initial:
            x = self.LSTM(
                    inputs, 
                    initial_state=[self.initialCell, self.initialHidden]
                    )
        else:
            x = self.LSTM(inputs)
        #return x
        return self.mu(x), self.var(x), self.value(x) #on retourne mu_t, var_t value, cell_state (h_t, c_t)




'''
--> la différence avec le tuto youtube c'est qu'on a plusieurs variables réelles pour définir l'action
    donc soit on fait --> ça c'est reglé, c'est la somme

--> il faut un self.grads qui vaut 0 au début d"un épisode puis à chaque time step : self.grads += tape.gradient(loss, explorer.trainable_variables)

--> idée d'avoir une reward 2D et de calculer la norme
'''

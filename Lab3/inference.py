import numpy as np
import graphics
import rover

def initialize_forward(observations, 
                       all_possible_hidden_states,
                       observation_model, 
                       prior_distribution,
                       forward_messages): 
    x_0 = observations[0]
    for z_0 in all_possible_hidden_states:
        if x_0 != None:
            location_given_z_0 = observation_model(z_0)[x_0]
        else:
            location_given_z_0 = 1

        prior_z_0 = prior_distribution[z_0]
        if (location_given_z_0 * prior_z_0) != 0:
            forward_messages[0][z_0] = location_given_z_0 * prior_z_0

    forward_messages[0].renormalize()

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # encoding everything as a Distribution
    for n in range(num_time_steps):
        forward_messages[n] = rover.Distribution()
        backward_messages[n] = rover.Distribution()
        marginals[n] = rover.Distribution()

    # forward intialization
    initialize_forward(observations, 
                       all_possible_hidden_states,
                       observation_model, 
                       prior_distribution,
                       forward_messages)

    # forward messages
    for k in range(1, num_time_steps):
        x_k = observations[k]

        for z_k in all_possible_hidden_states:
            forward_sum = 0
            if x_k != None:
                location_given_z_k = observation_model(z_k)[x_k]
            else:
                location_given_z_k = 1

            for prev_z in forward_messages[k-1]:
                forward_sum += forward_messages[k-1][prev_z] * transition_model(prev_z)[z_k]

            forward_message = location_given_z_k * forward_sum
            if forward_message != 0:
                forward_messages[k][z_k] = forward_message

        forward_messages[k].renormalize()
                   
    # backward initialization
    for z_k in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][z_k] = 1
    backward_messages[num_time_steps - 1].renormalize()
    
    # backward messages
    for k in range(1, num_time_steps):
        for z_k in all_possible_hidden_states:
            backward_sum = 0
            for next_z in backward_messages[num_time_steps-k]:
                next_x = observations[num_time_steps-k]
                if next_x != None:
                    location_given_next_z = observation_model(next_z)[next_x]
                else:
                    location_given_next_z = 1

                backward_sum += backward_messages[num_time_steps-k][next_z] * location_given_next_z * transition_model(z_k)[next_z]
            if backward_sum != 0:
                backward_messages[num_time_steps - k - 1][z_k] = backward_sum

        backward_messages[num_time_steps - k - 1].renormalize()

    # marginals
    for k in range (num_time_steps):
        marginal_sum = 0
        for z_k in all_possible_hidden_states:
            marginal = forward_messages[k][z_k] * backward_messages[k][z_k]
            if marginal != 0:
                marginals[k][z_k] = marginal
                marginal_sum += marginal

        for z_k in marginals[k].keys():
            marginals[k][z_k] = marginals[k][z_k] / marginal_sum
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    prev_zz = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps 

    for n in range(num_time_steps):
        forward_messages[n] = rover.Distribution()

    # forward intialization
    initialize_forward(observations, 
                       all_possible_hidden_states,
                       observation_model, 
                       prior_distribution,
                       forward_messages)

    for k in range(1, num_time_steps):
        x_k = observations[k]
        prev_zz[k] = dict()

        for z_k in all_possible_hidden_states:
            forward_max = -np.Infinity
            if x_k != None:
                location_given_z_k = observation_model(z_k)[x_k]
            else:
                location_given_z_k = 1

            for prev_z in forward_messages[k-1]:
                if transition_model(prev_z)[z_k] != 0:
                    new_max = max(forward_max, np.log(transition_model(prev_z)[z_k]) + forward_messages[k-1][prev_z])
                    if (new_max !=forward_max):
                        forward_max = new_max
                        prev_zz[k][z_k] = prev_z

            if location_given_z_k != 0:
                forward_messages[k][z_k] = np.log(location_given_z_k) + forward_max

    final_max = -np.Infinity
    for z_k in forward_messages[num_time_steps-1]:
        new_max = max(final_max, forward_messages[num_time_steps-1][z_k])
        if (new_max != final_max):
            final_max = new_max
            estimated_hidden_states[num_time_steps-1] = z_k
    
    for k in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps-1-k] = prev_zz[num_time_steps-k][estimated_hidden_states[num_time_steps-k]]
    
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = [False, True]
    for missing_observation in missing_observations:
        if missing_observation:
            filename = 'test_missing.txt'
        else:
            filename = 'test.txt'
                
        # load data    
        hidden_states, observations = rover.load_data(filename)

        if missing_observation:
            num_time_steps = 31
        else:
            num_time_steps = len(hidden_states)

        all_possible_hidden_states   = rover.get_all_hidden_states()
        all_possible_observed_states = rover.get_all_observed_states()
        prior_distribution           = rover.initial_distribution()
        
        print('Running forward-backward on ', filename)
        marginals = forward_backward(all_possible_hidden_states,
                                    all_possible_observed_states,
                                    prior_distribution,
                                    rover.transition_model,
                                    rover.observation_model,
                                    observations)
        print('\n')


    
        timestep = num_time_steps - 1
        print("Most likely parts of marginal at time %d:" % (timestep))
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
        print('\n')

    num_time_steps = len(hidden_states)
    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # get forward backward estimations
    fb_estimates = [None] * num_time_steps
    for i in range(num_time_steps):
        fb_max = -np.Infinity
        for k in marginals[i]:
            fb_max = max(fb_max, marginals[i][k])
            if marginals[i][k] == fb_max:
                fb_estimates[i] = k

    # calculate error
    v_correct = 0
    fb_correct = 0
    for i in range (num_time_steps):
        if fb_estimates[i] == hidden_states[i]:
            fb_correct += 1
        if estimated_states[i] == hidden_states[i]:
            v_correct += 1
        
    v_error = 1 - (v_correct/100)
    fb_error = 1 - (fb_correct/100)
    print ('The error probability of {z~} is ', v_error)
    print ('The error probability of {z^} is ', fb_error)

    # find inconsistnecy
    for i in range (num_time_steps):
        if fb_estimates[i][2] == 'stay' and fb_estimates[i+1][2] == 'stay' and fb_estimates[i] != fb_estimates[i+1]:
            print ('Inconsistency at i=', i, fb_estimates[i], fb_estimates[i+1])
            break

    # for i in range (num_time_steps-1):
    #     if fb_estimates[i][0] == fb_estimates[i+1][0] and fb_estimates[i][1] == fb_estimates[i+1][1] and fb_estimates[i+1][2] != 'stay':
    #         print ('Inconsistency at i=', i, fb_estimates[i], fb_estimates[i+1])
    #         break
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    # if enable_graphics:
    #     app = graphics.playback_positions(hidden_states,
    #                                       observations,
    #                                       estimated_states,
    #                                       marginals)
    #     app.mainloop()
        

"""
    This is an implementation of NSGA-II by Haris Ali Khan that has been
    modified to evolve deep neural networks to play Sonic the Hedgehog.
    Much of the Sonic code comes from the OpenAI Retro Contest. Although
    evolution instantiates the networks, they also learn using a PyTorch
    version of PPO.
"""

# Importing required modules
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import functools
from operator import mul

# Importing necessary PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import PPO code
import helpers.ppo as ppo
import helpers.utils as utils
from helpers.arguments import get_args
from helpers.envs import make_vec_envs
from helpers.model import Policy
from helpers.storage import RolloutStorage
from evaluation import evaluate

# Read PNG info
import struct

# For log file
from datetime import datetime

args = get_args()

def index_of(an_element, a_list):
    """ Function to find index of an element in a list """
    for i in range(0, len(a_list)):
        if a_list[i] == an_element:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list

def fast_non_dominated_sort(values1, values2):
    """ Function to carry out NSGA-II's fast non dominated sort """
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def calculate_novelty(population_behaviors, archive_behaviors):
    """ Calculate novelty scores by comparing behavior characterizations from
        current population to the archive of all behaviors. 
    """
    ns = []
    for i in range(0, len(population_behaviors)):
        total_dist = 0
        for j in range(0, len(archive_behaviors)):
            i_behavior = population_behaviors[i]
            j_behavior = archive_behaviors[j]
            dist = euclidean_dist(i_behavior, j_behavior)
            total_dist += dist

        avg_dist = total_dist / len(archive_behaviors)
        ns.append(avg_dist)
    return ns

def euclidean_dist(i_behavior, j_behavior):
    """ Distance between two behavior characterizations """
    index = total = 0
    # Position lists could be different lengths
    minimum = min(len(i_behavior), len(j_behavior))
    # Difference between common positions
    while index < minimum:
        total += (i_behavior[index] - j_behavior[index]) ** 2
        index += 1
    # If i_behavior is longer
    while index < len(i_behavior):
        total += i_behavior[index] ** 2
        index += 1
    # If j_behavior is longer
    while index < len(j_behavior):
        total += j_behavior[index] ** 2
        index += 1
    # NOTE: Could there be an issue where both vectors being compared are shorter than the longest
    #       vector in the whole population? What would be the consequences?
    total = math.sqrt(total)
    return total

def crowding_distance(values1, values2, front):
    """ Function to calculate crowding distance """
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def crossover(a, b):
    """ Function to carry out the crossover """
    r = random.random()
    if r > 0.5: # Crossover by adding the vectors and averaging
        return mutation((a+b)/2)
    else: # No crossover ... just mutate first of the two
        return mutation(a)

def mutation(solution):
    """ Function to carry out the mutation operator """
    max_range = args.mutation_range
    # Bit vector multiplied by the range
    mutation_scale = np.random.randint(2, size=len(solution)) * max_range
    solution = np.random.normal(solution, mutation_scale).astype(np.float32)
    return solution

def learn(env, agent):
    """ One network learns from evolved starting point. """
    global device

    net = agent.actor_critic
    num_steps = args.num_steps
    rollouts = RolloutStorage(num_steps, args.num_processes, envs.observation_space.shape,
                              envs.action_space, net.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    done = False
    num_updates = args.num_updates
    num_updates_with_no_reward = 0
    for j in range(num_updates):
        # decrease learning rate linearly
        # simply changes learning rate of optimizer
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr)

        accumulated_reward = 0
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = net.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # To watch while learning
            if args.watch_learning:
                envs.render()
            if device.type == 'cuda': # For some reason, CUDA actions are nested in an extra layer
                action = action[0]
            obs, reward, done, infos = envs.step(action)
            accumulated_reward += reward[0][0].item()

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        # Print total returns each time
        print(accumulated_reward, end=",")
        if accumulated_reward == 0:
            num_updates_with_no_reward += 1
        else:
            num_updates_with_no_reward = 0

        with torch.no_grad():
            next_value = net.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # Learning update (ignore return values)
        agent.update(rollouts)
        rollouts.after_update()

        if num_updates_with_no_reward > args.no_reward_tolerance:
            print("Agent not learning. Stop. ", end="")
            # If agent keeps getting no reward, then don't waste time on it
            break

    # Carriage return after all of the learning scores
    print("")

def random_genome(n):
    """ Random initial genome (network init should be favored) """
    # n is the number of weights
    return np.random.uniform(-1, 1, n).astype(np.float32)

def evaluate_population(solutions, agent, generation, pop_type):
    """ Evaluate every member of the included population, which is a collection
        of weight vectors for the neural networks.
    """
    global device
    fitness_scores = []
    behavior_characterizations = []
    save_path = os.path.join(logging_location, "gen{}".format(generation))

    if generation % args.save_interval == 0:
        try:
            os.makedirs(save_path)
        except OSError:
            pass

    for i in range(pop_size):
        print("Evaluating genome #{}:".format(i), end=" ")  # No newline: Fitness will print here

        # Creates solutions[i] Tensor and converts it to type Float before passing on to set_weights
        weights = torch.from_numpy(solutions[i])
        weights = weights.to(device)
        set_weights(agent.actor_critic, weights)

        if args.evol_mode in {'baldwin', 'lamarck'}:        
            # Make the agent optimize the starting weights. Weights of agent are changed via side-effects
            print("Learning.", end=" ")
            learn(envs, agent)
            
            if args.evol_mode == 'lamarck':
                solutions[i] = extract_weights(agent.actor_critic)

        if generation % args.save_interval == 0 and pop_type == "parents":
            torch.save([
                agent.actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "model-{}.pt".format(i)).replace("\\", "/"))  # Windows thing

        # Do evaluation of agent without learning to get fitness and behavior characterization
        print("Evaluating.", end=" ")
        fitness, behavior_char = evaluate(agent.actor_critic, envs, device, generation, args)
        print("({},{})".format(fitness, behavior_char))
        fitness_scores.append(fitness)
        behavior_characterizations.append(behavior_char)
            
    return (fitness_scores, behavior_characterizations)

def set_weights(net, weights):
    """ Set weights of the network to those from a linear genome. """

    i = 0
    # Get lengths of all the layers
    lengths = []
    sizes = []
    for layer in list(net.parameters()):
        length = torch.numel(layer)
        lengths.append(length)
        size = tuple(layer.size())
        sizes.append(size)

    # Split into several weight Tensors corresponding to the number of elements in every layer.
    split_vector = torch.split(weights, lengths, 0)

    # With every layer, replace its respective weight Tensor and set its data to that Tensor.
    for layer in list(net.parameters()):
        if i >= len(lengths) or i >= len(sizes):
            print("Index out of bounds")
            quit()
        if functools.reduce(mul, sizes[i], 1) != lengths[i]:
            print("Size error at Layer {}: {}".format(i, layer))
            quit()

        reshaped_weights = torch.reshape(split_vector[i], sizes[i])

        layer.data = reshaped_weights
        i += 1

def extract_weights(net):
    """Function to extract learned network weights from model as a linear vector/genome. 
       Mainly needed for Lamarckian evolution.
    """
    cnn_weights = []
    for layer in list(net.parameters()):
        cloned_layer = layer.clone()
        cloned_layer = cloned_layer.reshape(-1)
        layer_array = cloned_layer.cpu().data.numpy()
        cnn_weights = np.concatenate((cnn_weights, layer_array)).astype(np.float32)
    
    return cnn_weights

def log_line(str):
    """ Log given string to globally defined log file """
    f = open(log_file_name, 'a')  # Append a line
    f.write(str)
    f.close()

def log_behavior_archive(generation, behavior_archive):
    """ Save all behavior vectors to file """
    global logging_location
    f = open(os.path.join('{}/gen{}'.format(logging_location, generation),
             "archive.gen{}.txt".format(generation)), 'w')
    for i in range(len(behavior_archive)):
        bc_list = behavior_archive[i]
        for j in range(len(bc_list)):
            f.write("{}\t".format(bc_list[j]))

        f.write("\n")

    f.close()
    # Make a gnuplot file that plots end points on a map of the level
    global map_width
    global map_height
    global map_file
    plotfile = open(os.path.join('{}/gen{}'.format(logging_location, generation),
                    "endpoints.gen{}.plot".format(generation)), 'w')
    plotfile.write("set xrange [0:{}]\n".format(map_width))
    plotfile.write("set yrange [0:{}]\n".format(map_height))
    plotfile.write("plot \"../../../{}\" binary filetype=png w rgbimage notitle, \\\n".format(map_file))
    plotfile.write("     \"archive.gen{}.txt\" u 1:({} - $2) ls 7 notitle\n".format(generation,map_height))
    plotfile.write("pause -1\n")
    plotfile.close()

def log_scores_and_behaviors(population_type, generation, fitness_scores, novelty_scores, behavior_characterizations):
    """ Create log file with details from a single generation """
    global logging_location
    f = open(os.path.join('{}/gen{}'.format(logging_location, generation),
             "{}.gen{}.txt".format(population_type, generation)), 'w')
    f.write("#Fitness\tNovelty\tFinal x\tFinal y\n")
    for i in range(len(fitness_scores)):
        if args.final_pt:
            bc_list = behavior_characterizations[i]
        else:
            bc_list = behavior_characterizations[i][-2:]
        f.write("{}\t{}\t{}\t{}\n".format(fitness_scores[i],novelty_scores[i],bc_list[0],bc_list[1]))

    f.close()

if __name__ == '__main__':
    # Main program starts here

    # Competition version of environment make:
    # Mus pip install gym==0.12.1 in order for this to work
    # env = make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")

    # For some reason, this seems necessary for a wrapper that allows early environment termination
    log_dir = os.path.expanduser('/tmp/gym/') # Nothing is really stored here
    utils.cleanup_log_dir(log_dir)

    # Set up plotting of end points on PNG map of level
    global map_height
    global map_width
    global map_file

    map_file = "maps/{}.{}.png".format(args.env_name, args.env_state)
    # From : https://stackoverflow.com/questions/8032642/how-to-obtain-image-size-using-standard-python-class-without-using-external-lib
    with open(map_file, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            print("Level map image file not present!")
            quit(1)
        check = struct.unpack('>i', head[4:8])[0]
        if check != 0x0d0a1a0a:
            print("Level map image file could not be read!")
            quit(1)
        width, height = struct.unpack('>ii', head[16:24])
        map_height = height
        map_width = width

    # Actual logs
    global log_file_name, logging_location
    now = datetime.now()  # current date and time
    #new_log = now.strftime("%Y-%m-%d-{}-{}".format(args.env_state, args.evol_mode))
    # Drop dates from logs
    new_log = now.strftime("{}-{}".format(args.env_state, args.evol_mode))
    if args.save_dir != "": logging_location = os.path.join(args.save_dir, new_log)
    try:
        os.makedirs(logging_location)
    except OSError:
        pass
    log_file_name = os.path.join(logging_location, "{}-log.txt".format(new_log)).replace("\\", "/")  # Windows, again
    log_line("#Gen\tMinFitness\tAvgFitness\tMaxFitness\tMinNovelty\tAvgNovelty\tMaxNovelty\n")

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.env_state, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, allow_early_resets=True)

    pop_size = args.pop_size
    global num_weights
    
    if args.resume_gen > -1:
        # Figure out where the files are saved
        gen_dir = os.path.join(logging_location, "gen{}".format(args.resume_gen)).replace("\\", "/")
        solutions = []
        # Load each population member
        for i in range(0, pop_size):
            model_path = "{}/model-{}.pt".format(gen_dir, i)
            print("Load {}".format(model_path))
            # Load network model
            if torch.cuda.is_available():
                actor_critic, _ = torch.load(model_path)
            else:
                actor_critic, _ = torch.load(model_path, map_location='cpu')
            # Extract the weights
            solutions.append(extract_weights(actor_critic))

        num_weights = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    else:
        if args.init_from_network:
            solutions = []
            for i in range(0, pop_size):
                # Policy is created - in our case, since obs_shape is 3, it becomes a CNN
                # Each initialization results in a new set of random weights
                actor_critic = Policy(
                    envs.observation_space.shape,
                    envs.action_space,
                    base_kwargs={'recurrent': args.recurrent_policy, 'is_genesis': True})
                actor_critic.to(device)
                # Take those random weights and create a genome
                solutions.append(extract_weights(actor_critic))

            num_weights = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
        else:
            # Policy is created - in our case, since obs_shape is 3, it becomes a CNN
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy, 'is_genesis': True})
            actor_critic.to(device)
        
            num_weights = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
            # Only need to know the number of weights in order to create completely random weights
            solutions = [random_genome(num_weights) for i in range(0, pop_size)]

    # Agent is initialized
    agent = ppo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=1e-8, # This epsilon is not for exploration. It is for numerical stability of the Adam optimizer. This is the default value.
        max_grad_norm=args.max_grad_norm)
            
    behavior_archive = []
    if args.resume_gen > -1:
        gen_no = args.resume_gen
        # Load the previous behavior archive
        with open(os.path.join('{}/gen{}'.format(logging_location, gen_no),
             "archive.gen{}.txt".format(gen_no)), 'r') as f:
            num_entries = 0
            # Twice the pop_size number of entries for each completed generation (parents and children)
            # but only parents from the final generation. 
            expected_entries = (gen_no * 2 * pop_size) + pop_size
            previous_entries = (gen_no * 2 * pop_size) # Just from previous generations
            # Need to store most recent parents in behavior characterizations for recalculation of novelty
            behavior_characterizations = []
            for line in f:
                values = [int(x) for x in line.split()]
                behavior_archive.append(values)
                num_entries += 1
                if num_entries > expected_entries:
                    # Don't load saved child entries from last generation, since new children will be generated on the resume
                    break
                if num_entries > previous_entries:
                    behavior_characterizations.append(values)
                    
    else:
        gen_no = 0

    while gen_no < args.num_gens:
        if gen_no == args.resume_gen:
            fitness_scores = [] # Rewards
            novelty_scores = [] # Novelty
            # If loading the previously saved models, don't re-evaluate parents. Load the previous scores
            with open(os.path.join('{}/gen{}'.format(logging_location, gen_no),
                    "parents.gen{}.txt".format(gen_no)), 'r') as f:
                # Discard line of headers
                f.readline()
                # Remaining lines have data
                for line in f:
                    values = [float(x) for x in line.split()]
                    fitness_scores.append(values[0])
                    novelty_scores.append(values[1])
        else:
            print("Start generation {}".format(gen_no))
            (fitness_scores, behavior_characterizations) = evaluate_population(solutions, agent, gen_no, "parents")
            # Compare all of the behavior characterizations to get the diversity/novelty scores.
            behavior_archive = behavior_archive + behavior_characterizations
            novelty_scores = calculate_novelty(behavior_characterizations, behavior_archive)

            if gen_no % args.save_interval == 0:
                log_scores_and_behaviors("parents", gen_no, fitness_scores, novelty_scores, behavior_characterizations)
        
        log_line("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(gen_no,
                        np.min(fitness_scores), np.mean(fitness_scores), np.max(fitness_scores),
                        np.min(novelty_scores), np.mean(novelty_scores), np.max(novelty_scores)))
        
        print("Max, Average, Min Fitness are {}, {} and {}".format(np.max(fitness_scores), np.mean(fitness_scores), np.min(fitness_scores)))
        print("Max, Average, Min Novelty are {}, {} and {}".format(np.max(novelty_scores), np.mean(novelty_scores), np.min(novelty_scores)))
        non_dominated_sorted_solution = fast_non_dominated_sort(fitness_scores[:], novelty_scores[:])
        print("The best front for Generation number ", gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print("Fitness:", fitness_scores[valuez])
            print("Novelty:", novelty_scores[valuez])
            print("------------------")
         
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(fitness_scores[:], novelty_scores[:], non_dominated_sorted_solution[i][:]))

        # The lambda children
        solution2 = []
        # Generating offspring
        while len(solution2) != pop_size:
            a1 = random.randint(0, pop_size-1)
            b1 = random.randint(0, pop_size-1)
            solution2.append(crossover(solutions[a1], solutions[b1]))
        
        print("Evaluate children of generation {}".format(gen_no))
        (fitness_scores2, behavior_characterizations2) = evaluate_population(solution2, agent, gen_no, "children")
        # The novelty scores for the parents need to be recomputed given the new members of the archive before selection on
        # the combined parent/child population occurs.
        combined_behaviors = behavior_characterizations+behavior_characterizations2 # parents and children
        behavior_archive = behavior_archive + behavior_characterizations2 # Add children to archive (parents already present)
        novelty_scores_combined = calculate_novelty(combined_behaviors, behavior_archive)

        # Combine parent and child populations into one before elitist selection
        function1_values2 = fitness_scores + fitness_scores2
        function2_values2 = novelty_scores_combined

        log_behavior_archive(gen_no, behavior_archive)
        log_scores_and_behaviors("combined", gen_no, function1_values2, function2_values2, combined_behaviors)
        
        # Selection to generate the next parent population
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == pop_size:
                    break
            if len(new_solution) == pop_size:
                break
        # Combine the parent and child solutions so the best can be selected for the next parent population
        solution2 = solutions + solution2
        solutions = [solution2[i] for i in new_solution]
        gen_no += 1

    # Let's plot the final front now
    function1 = fitness_scores
    function2 = novelty_scores
    plt.xlabel('Fitness', fontsize=15)
    plt.ylabel('Novelty', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()

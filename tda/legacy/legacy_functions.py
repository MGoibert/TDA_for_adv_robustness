# TODO: Deprecate this
def compute_persistent_dgm(model, test_set, loss_func,
                           numero_ex=0, adversarial=False, epsilon=.25,
                           threshold=0, noise=0, num_classes=None):
    """
    Compute the persistent diagram
    """

    t0 = time.time()

    # Input
    sample = test_set[numero_ex]

    x, y = process_sample(
        sample,
        adversarial,
        noise,
        epsilon,
        model,
        num_classes
    )

    pred = model(x).argmax(dim=-1).item()
    x = x.view(-1, 28 * 28)

    edge_dict = compute_all_edge_values(model, x)

    diagram = compute_dgm_from_edges(
        edge_dict=edge_dict,
        threshold=threshold
    )

    d.plot.plot_diagram(diagram, show=True)

    t1 = time.time()
    print("Time: %s, true label = %s, pred = %s, adv = %s" % (np.round(t1 - t0, decimals=2),
                                                              test_set[numero_ex][1], pred, adversarial))

    return diagram, test_set[numero_ex][1], adversarial, pred


# Derive valid indices for a specific class
def get_class_indices(label_wanted, number=2, start_i=0, test_set=test_set):
    accept = False
    valid_numero = []
    i = start_i
    if number == "all":
        for i in range(len(test_set)):
            numero_ex = i
            y = test_set[i][1]
            if y == label_wanted:
                valid_numero.append(numero_ex)
    else:
        while accept == False:
            numero_ex = i
            y = test_set[i][1]
            if y == label_wanted:
                valid_numero.append(numero_ex)
            if len(valid_numero) == number:
                accept = True
            i = i + 1
    return valid_numero


# Compute intra-class distances distrib
def compute_intra_distances(dgms_dict):
    # Building the dictionary containing the distance distributions
    distrib_dist = {}

    # Step 1: intra-class distances distributions
    for k, key in enumerate(dgms_dict.keys()):
        dist_temp = []
        print("key =", key)
        for i, ind1 in enumerate(dgms_dict[key].keys()):
            print("ind =", ind1)
            for j in range(i + 1, len(dgms_dict[key].keys())):
                ind2 = list(dgms_dict[key].keys())[j]
                dist_temp.append(
                    d.wasserstein_distance(dgms_dict[key][ind1][0],
                                           dgms_dict[key][ind2][0], q=2))
        distrib_dist["dist_" + str(k) + "_" + str(k)] = dist_temp
    return distrib_dist


# Compute adv. or noisy inputs vs clean inputs distances distrib
def compute_distances(dgms_dict, dgms_dict_perturbed, adversarial=True, noisy=False):
    add = "_adv" * adversarial + "_noise" * noisy
    distrib_dist = {}
    print("Computing distance distribution for", add, "inputs")

    for k, key in enumerate(dgms_dict.keys()):
        dist_temp = []
        print("key =", key)
        for i, ind in enumerate(dgms_dict_perturbed[key].keys()):
            print("ind" + add + " =", ind)
            for j, ind_clean in enumerate(dgms_dict[key].keys()):
                dist_temp.append(
                    d.wasserstein_distance(dgms_dict_perturbed[key][ind][0],
                                           dgms_dict[key][ind_clean][0], q=2))
        distrib_dist["dist" + add + "_" + str(k)] = dist_temp
    return distrib_dist


# Compute inter-class distribution for clean inputs.
# Careful, very long to compute
def compute_inter_distances(dgms_dict):
    distrib_dist = {}
    for class1, _ in enumerate(dgms_dict.keys()):
        key1 = "dgms_" + str(class1)
        print("key1 =", key1)
        for class2 in range(class1 + 1, len(dgms_dict.keys())):
            key2 = "dgms_" + str(class2)
            print("key2 =", key2)
            dist_temp = []
            for i, ind1 in enumerate(dgms_dict[key1].keys()):
                for j, ind2 in enumerate(dgms_dict[key2].keys()):
                    print("inds =", ind1, ind2)
                    dist_temp.append(
                        d.wasserstein_distance(dgms_dict[key1][ind1][0],
                                               dgms_dict[key2][ind2][0], q=2))
        distrib_dist["dist_" + str(class1) + "_" + str(class2)] = dist_temp

    return distrib_dist


def produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=False, epsilon=0.25, noise=0, num_classes=None):
    dgms_dict = {}
    dict_temp = {}
    for i in inds_all_class.keys():
        for index in inds_all_class[i]:
            dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
                                                      numero_ex=index, threshold=threshold, adversarial=adversarial,
                                                      epsilon=epsilon, noise=noise, num_classes=num_classes)

    for i in inds_all_class.keys():
        temp = {}
        for index in dict_temp.keys():
            if dict_temp[index][3] == i:
                temp[index] = dict_temp[index]
        dgms_dict["dgms_" + str(i)] = temp

    return dgms_dict


# Save results
def save_result(result, threshold, epsilon, noise):
    import os
    param = "threshold_%s_eps_%s_noise_%s/" % (threshold, epsilon, noise)
    path = "/Users/m.goibert/Documents/Criteo/Project_2-Persistent_Homology/TDA_for_adv_robustness/dict_files/" + param

    if not os.path.exists(path):
        os.makedirs(path)

    import pickle
    import _pickle as cPickle

    # Clean input
    t0 = time.time()
    with open(path + 'dgms_dict.pickle', 'wb') as fp:
        cPickle.dump(result[1], fp, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print("dgms_dict saved ! Time =", t1 - t0)

    with open(path + 'distrib_dist.pickle', 'wb') as fp:
        cPickle.dump(result[2], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + 'inds_all_class.pickle', 'wb') as fp:
        cPickle.dump(result[0], fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Adv input
    with open(path + 'dgms_dict_adv.pickle', 'wb') as fp:
        cPickle.dump(result[4], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_adv saved !")

    with open(path + 'distrib_dist_adv.pickle', 'wb') as fp:
        cPickle.dump(result[5], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + 'inds_all_class_adv.pickle', 'wb') as fp:
        cPickle.dump(result[3], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + 'dgms_dict_adv_incorrect.pickle', 'wb') as fp:
        cPickle.dump(result[6], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_adv_incorrect saved !")

    with open(path + 'distrib_dist_adv_incorrect.pickle', 'wb') as fp:
        cPickle.dump(result[7], fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Noisy input
    with open(path + 'dgms_dict_noise.pickle', 'wb') as fp:
        cPickle.dump(result[9], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_noise saved !")

    with open(path + 'inds_all_class_noise.pickle', 'wb') as fp:
        cPickle.dump(result[8], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + 'distrib_dist_noise.pickle', 'wb') as fp:
        cPickle.dump(result[10], fp, protocol=pickle.HIGHEST_PROTOCOL)




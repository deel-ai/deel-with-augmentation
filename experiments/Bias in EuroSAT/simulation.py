from base_model import CNN
from evaluate import evaluate_2
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import torch

def simulation_simple(X_train, S_train, y_train, new_images, nb_new_images, device, type=['blue_highway', 'blue_river']):
    global_accuracy = []
    highway_accuracy = []
    blue_highway_accuracy = []
    river_accuracy = []
    blue_river_accuracy = []
    disparate_impact_highway = []
    disparate_impact_river = []
    average_abs_odds_difference_highway = []
    average_abs_odds_difference_river = []
    percentage_error_minority = []
    percentage_error_majority = []
    error_rate_ratio = []
    for i in range(len(nb_new_images)):
        if type == 'blue_highway':
            new_X_train = np.concatenate((X_train, new_images[0][:nb_new_images[i]].cpu().detach().numpy()), axis=0)
            new_S_train = np.concatenate((S_train, torch.zeros(nb_new_images[i]).cpu().detach().numpy()), axis=0)
            new_y_train = np.concatenate((y_train, torch.zeros(nb_new_images[i]).cpu().detach().numpy()), axis=0)
        elif type == 'blue_river':
            new_X_train = np.concatenate((X_train, new_images[0][:nb_new_images[i]].cpu().detach().numpy()), axis=0)
            new_S_train = np.concatenate((S_train, torch.zeros(nb_new_images[i]).cpu().detach().numpy()), axis=0)
            new_y_train = np.concatenate((y_train, torch.ones(nb_new_images[i]).cpu().detach().numpy()), axis=0)

        cnn_model = CNN(output_neurons=1, device=device)
        losses = cnn_model.fit(X_train=new_X_train, y_train=new_y_train, num_epochs=15, batch_size=16, learning_rate=2e-5)

        glo_acc, hw_acc, bl_hw_acc, rv_acc, bl_rv_acc, dis_hw, dis_rv, aaod_hw, aaod_rv, err_min, err_maj, err = evaluate_2(
            X_train=new_X_train,
            y_train=new_y_train,
            S_train=new_S_train,
            model=cnn_model
        )
        global_accuracy.append(glo_acc)
        highway_accuracy.append(hw_acc)
        blue_highway_accuracy.append(bl_hw_acc)
        river_accuracy.append(rv_acc)
        blue_river_accuracy.append(bl_rv_acc)
        disparate_impact_highway.append(dis_hw)
        disparate_impact_river.append(dis_rv)
        average_abs_odds_difference_highway.append(aaod_hw)
        average_abs_odds_difference_river.append(aaod_rv)
        percentage_error_minority.append(err_min)
        percentage_error_majority.append(err_maj)
        error_rate_ratio.append(err)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()   

    # Define Plot Data
    data=[["Global accuracy", global_accuracy[0], global_accuracy[1], global_accuracy[2], global_accuracy[3], global_accuracy[4], global_accuracy[5]],
        ["Highway accuracy", highway_accuracy[0], highway_accuracy[1], highway_accuracy[2], highway_accuracy[3], highway_accuracy[4], highway_accuracy[5]],
        ["Blue highway accuracy", blue_highway_accuracy[0], blue_highway_accuracy[1], blue_highway_accuracy[2], blue_highway_accuracy[3], blue_highway_accuracy[4], blue_highway_accuracy[5]],
        ["River accuracy", river_accuracy[0], river_accuracy[1], river_accuracy[2], river_accuracy[3], river_accuracy[4], river_accuracy[5]],
        ["Blue river accuracy", blue_river_accuracy[0], blue_river_accuracy[1], blue_river_accuracy[2], blue_river_accuracy[3], blue_river_accuracy[4], blue_river_accuracy[5]]
        ]
    # Plot multiple columns bar chart
    df=pd.DataFrame(data,columns=["Evaluations", "Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                                "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])])
    df.plot(x="Evaluations", y=["Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(15,8))
    plt.title("Accuracy when increasing the number of images added to the original dataset")
    # Show
    plt.show() 

    data_1=[["DI highway", disparate_impact_highway[0][0], disparate_impact_highway[1][0], disparate_impact_highway[2][0], disparate_impact_highway[3][0], disparate_impact_highway[4][0], disparate_impact_highway[5][0]],
        ["DI river", disparate_impact_river[0][0], disparate_impact_river[1][0], disparate_impact_river[2][0], disparate_impact_river[3][0], disparate_impact_river[4][0], disparate_impact_river[5][0]]
        ]
    # Plot multiple columns bar chart
    df_1=pd.DataFrame(data_1,columns=["Evaluations","Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                                    "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])])
    df_1.plot(x="Evaluations", y=["Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                  "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(15,8))
    plt.title("Disparate impact")
    plt.show()

    data_2=[["AaoD highway", average_abs_odds_difference_highway[0], average_abs_odds_difference_highway[1], average_abs_odds_difference_highway[2], average_abs_odds_difference_highway[3], average_abs_odds_difference_highway[4], average_abs_odds_difference_highway[5]],
        ["AaoD river", average_abs_odds_difference_river[0], average_abs_odds_difference_river[1], average_abs_odds_difference_river[2], average_abs_odds_difference_river[3], average_abs_odds_difference_river[4],average_abs_odds_difference_river[5]]
        ]
    # Plot multiple columns bar chart
    df_2=pd.DataFrame(data_2,columns=["Evaluations","Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                                    "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])])
    df_2.plot(x="Evaluations", y=["Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                  "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(15,8))
    plt.title("Average absolute difference")
    plt.show()

    data_3=[["Err blue", percentage_error_minority[0], percentage_error_minority[1], percentage_error_minority[2], percentage_error_minority[3], percentage_error_minority[4], percentage_error_minority[5]],
        ["Err normal", percentage_error_majority[0], percentage_error_majority[1], percentage_error_majority[2], percentage_error_majority[3], percentage_error_majority[4], percentage_error_majority[5]]
        ]
    # Plot multiple columns bar chart
    df_3=pd.DataFrame(data_3,columns=["Evaluations","Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                                    "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])])
    df_3.plot(x="Evaluations", y=["Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                  "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(9,5))
    plt.title("Percentage error of blue images and normal images")
    plt.show()

    data_4=[["ErR", error_rate_ratio[0], error_rate_ratio[1], error_rate_ratio[2], error_rate_ratio[3], error_rate_ratio[4], error_rate_ratio[5]]]
    # Plot multiple columns bar chart
    df_4=pd.DataFrame(data_4,columns=["Evaluations","Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                                    "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])])
    df_4.plot(x="Evaluations", y=["Original dataset", "nb {} = {}".format(type, nb_new_images[1]), "nb {} = {}".format(type, nb_new_images[2]),
                                  "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(9,5))
    plt.title("Error rate ratio")
    # Show
    plt.suptitle("Evaluations when increasing the number of images added to the original dataset")
    plt.show()

def simulation_double(X_train, S_train, y_train, new_images_1, nb_new_images_1, new_images_2, nb_new_images_2, device, type_1=['blue_highway', 'blue_river'], type_2=['blue_highway', 'blue_river']):
    global_accuracy = []
    highway_accuracy = []
    blue_highway_accuracy = []
    river_accuracy = []
    blue_river_accuracy = []
    disparate_impact_highway = []
    disparate_impact_river = []
    average_abs_odds_difference_highway = []
    average_abs_odds_difference_river = []
    percentage_error_minority = []
    percentage_error_majority = []
    error_rate_ratio = []
    for i in range(len(nb_new_images_1)):
        if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
            new_X_train = np.concatenate((X_train, new_images_1[0][:nb_new_images_1[i]].cpu().detach().numpy(), new_images_2[0][:nb_new_images_2[i]].cpu().detach().numpy()), axis=0)
            new_S_train = np.concatenate((S_train, torch.zeros(nb_new_images_1[i]).cpu().detach().numpy(), torch.zeros(nb_new_images_2[i]).cpu().detach().numpy()), axis=0)
            new_y_train = np.concatenate((y_train, torch.zeros(nb_new_images_1[i]).cpu().detach().numpy(), torch.ones(nb_new_images_2[i]).cpu().detach().numpy()), axis=0)
        elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
            new_X_train = np.concatenate((X_train, new_images_1[0][:nb_new_images_1[i]].cpu().detach().numpy(), new_images_2[0][:nb_new_images_2[i]].cpu().detach().numpy()), axis=0)
            new_S_train = np.concatenate((S_train, torch.zeros(nb_new_images_1[i]).cpu().detach().numpy(), torch.zeros(nb_new_images_2[i]).cpu().detach().numpy()), axis=0)
            new_y_train = np.concatenate((y_train, torch.ones(nb_new_images_1[i]).cpu().detach().numpy(), torch.zeros(nb_new_images_2[i]).cpu().detach().numpy()), axis=0)

        cnn_model = CNN(output_neurons=1, device=device)
        losses = cnn_model.fit(X_train=new_X_train, y_train=new_y_train, num_epochs=10, batch_size=200, learning_rate=2e-5)

        glo_acc, hw_acc, bl_hw_acc, rv_acc, bl_rv_acc, dis_hw, dis_rv, aaod_hw, aaod_rv, err_min, err_maj, err = evaluate_2(
            X_train=new_X_train,
            y_train=new_y_train,
            S_train=new_S_train,
            model=cnn_model
        )
        global_accuracy.append(glo_acc)
        highway_accuracy.append(hw_acc)
        blue_highway_accuracy.append(bl_hw_acc)
        river_accuracy.append(rv_acc)
        blue_river_accuracy.append(bl_rv_acc)
        disparate_impact_highway.append(dis_hw)
        disparate_impact_river.append(dis_rv)
        average_abs_odds_difference_highway.append(aaod_hw)
        average_abs_odds_difference_river.append(aaod_rv)
        percentage_error_minority.append(err_min)
        percentage_error_majority.append(err_maj)
        error_rate_ratio.append(err)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()   

    # Define Plot Data
    data=[["Global accuracy", global_accuracy[0], global_accuracy[1], global_accuracy[2], global_accuracy[3], global_accuracy[4], global_accuracy[5]],
        ["Highway accuracy", highway_accuracy[0], highway_accuracy[1], highway_accuracy[2], highway_accuracy[3], highway_accuracy[4], highway_accuracy[5]],
        ["Blue highway accuracy", blue_highway_accuracy[0], blue_highway_accuracy[1], blue_highway_accuracy[2], blue_highway_accuracy[3], blue_highway_accuracy[4], blue_highway_accuracy[5]],
        ["River accuracy", river_accuracy[0], river_accuracy[1], river_accuracy[2], river_accuracy[3], river_accuracy[4], river_accuracy[5]],
        ["Blue river accuracy", blue_river_accuracy[0], blue_river_accuracy[1], blue_river_accuracy[2], blue_river_accuracy[3], blue_river_accuracy[4], blue_river_accuracy[5]]
        ]
    # Plot multiple columns bar chart
    df=pd.DataFrame(data,columns=["Evaluations", "Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df.plot(x="Evaluations", y=["Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(15,8))
    plt.title("Accuracy when added pairs of image numbers into the original dataset")
    # Show
    plt.show()

    data_1=[["DI highway", disparate_impact_highway[0][0], disparate_impact_highway[1][0], disparate_impact_highway[2][0], disparate_impact_highway[3][0], disparate_impact_highway[4][0], disparate_impact_highway[5][0]],
        ["DI river", disparate_impact_river[0][0], disparate_impact_river[1][0], disparate_impact_river[2][0], disparate_impact_river[3][0], disparate_impact_river[4][0], disparate_impact_river[5][0]]
        ]
    # Plot multiple columns bar chart
    df_1=pd.DataFrame(data_1,columns=["Evaluations", "Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df_1.plot(x="Evaluations", y=["Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(15,8))
    plt.title("Disparate impact")
    plt.show()

    data_2=[["AaoD highway", average_abs_odds_difference_highway[0], average_abs_odds_difference_highway[1], average_abs_odds_difference_highway[2], average_abs_odds_difference_highway[3], average_abs_odds_difference_highway[4], average_abs_odds_difference_highway[5]],
        ["AaoD river", average_abs_odds_difference_river[0], average_abs_odds_difference_river[1], average_abs_odds_difference_river[2], average_abs_odds_difference_river[3], average_abs_odds_difference_river[4], average_abs_odds_difference_river[5]]
        ]
    # Plot multiple columns bar chart
    df_2=pd.DataFrame(data_2,columns=["Evaluations","Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df_2.plot(x="Evaluations", y=["Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(15,8))
    plt.title("Average absolute difference")
    plt.show()

    data_3=[["Err blue", percentage_error_minority[0], percentage_error_minority[1], percentage_error_minority[2], percentage_error_minority[3], percentage_error_minority[4], percentage_error_minority[5]],
        ["Err normal", percentage_error_majority[0], percentage_error_majority[1], percentage_error_majority[2], percentage_error_majority[3], percentage_error_majority[4], percentage_error_majority[5]]
        ]
    # Plot multiple columns bar chart
    df_3=pd.DataFrame(data_3,columns=["Evaluations","Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df_3.plot(x="Evaluations", y=["Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(9,5))
    plt.title("Percentage error of blue images and normal images")
    plt.show()

    data_4=[["ErR", error_rate_ratio[0], error_rate_ratio[1], error_rate_ratio[2], error_rate_ratio[3], error_rate_ratio[4], error_rate_ratio[5]]]
    # Plot multiple columns bar chart
    df_4=pd.DataFrame(data_4,columns=["Evaluations","Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df_4.plot(x="Evaluations", y=["Original dataset","nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(9,5))
    plt.title("Error rate ratio")
    # Show
    plt.suptitle("Evaluations when added pairs of image numbers into the original dataset")
    plt.show()
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters:
    -----------
    patience
        How long to wait after last time validation loss improved
    verbose
        If True, prints a message for each validation loss improvement (Default: False)
    delta
        Minimum change in the monitored quantity to qualify as an improvement (Default: 0)
    path
        Path for the checkpoint to be saved to (Default: 'checkpoint.pt')
    trace_func
        trace print function (Default: print)
    """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ===>>>')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def average_seeds(path, nb_seeds):
    accuracy = 0
    error_rate = 0
    for i in range(nb_seeds):
        accuracy_seed = np.load(f"{path}/accuracy_seed_{i+1}.npy")
        error_rate_seed = np.load(f"{path}/error_rate_seed_{i+1}.npy")
        accuracy += accuracy_seed
        error_rate += error_rate_seed
    accuracy = accuracy/nb_seeds
    error_rate = error_rate/nb_seeds
    return accuracy, error_rate

def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    for tar, bia in zip(targets, biases):
        confusion_matrix_org[bia.long(), tar.long()] += 1
    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    return confusion_matrix 

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def clip_max_ratio(score, ratio):
    upper_bd = score.min() * ratio
    return np.clip(score, None, upper_bd)


def plot_wrong_pred(wrong_pred):
    # Plot the Land classification wrongly predicted
    labels =['Highway', 'River']
    plt.figure(figsize = (11,11))
    for idx, pred in enumerate(wrong_pred[:4]):
        img = np.array(pred[1].tolist())
        plt.subplot(2,2,idx+1)
        plt.title(f'Correct: {labels[pred[2]]} -- Predicted: {labels[pred[0]]}')
        #plt.xlabel(f'Confidence score : {pred[3]}')
        img = np.dstack((img[0],img[1],img[2]))
        plt.imshow(img)

def plot_bar_err_rate(err_rate):
    plt.figure(figsize = (7,7))
    bars = ('err_bl_hw', 'err_bl_rv', 'err_bl', 'err_nor', 'err_nor_hw', 'err_nor_rv', 'err_hw', 'err_rv')
    # Choose the position of each barplots on the x-axis
    x_pos = [1,2,4,5,7,8,10,11]
    err_rate = np.array(err_rate)
    colors = cm.viridis(err_rate / float(max(err_rate)))
    # Create bars
    plot = plt.scatter(err_rate, err_rate, c = err_rate, cmap = 'viridis')
    plt.cla()
    plt.colorbar(plot)
    plt.bar(x_pos, err_rate, color=colors)
    # Create names on the x-axis
    plt.xticks(x_pos, bars, rotation=45)
    # zip joins x and y coordinates in pairs
    for x,y in zip(x_pos, err_rate):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,5), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    plt.ylabel('Error rate')
    plt.title('Error rate of image groups')
    # Show graphic
    plt.show()

def plot_loss_acc_train(train_losses, val_losses, train_acc, val_acc):
    # Ploting of loss and accuracy on the training.
    plt.figure(figsize = (12,10))
    plt.subplot(2,2,1)
    plt.plot(train_losses, linewidth = 2, color = "r", label='Training Loss')
    plt.plot(val_losses, linewidth = 2, color = "g", label='Validation Loss')
    # Find position of lowest validation loss
    minposs = val_losses.index(min(val_losses))+1 
    plt.axvline(minposs, linestyle='--', color='b', label='Early Stopping')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(train_acc, color = "r", label = 'Training Accuracy')
    plt.plot(val_acc, color = "g", label = 'Validation Accuracy')
    plt.axvline(minposs, linestyle='--', color='b', label='Early Stopping')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

def plot_bar_acc_simu_simple(
    global_accuracy,
    highway_accuracy,
    blue_highway_accuracy,
    river_accuracy,
    blue_river_accuracy,
    nb_new_images,
    type=['blue_highway', 'blue_river']
):
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
                                "nb {} = {}".format(type, nb_new_images[3]), "nb {} = {}".format(type, nb_new_images[4]), "nb {} = {}".format(type, nb_new_images[5])], kind="bar", figsize=(12,6))
    plt.title('Accuracy when increasing the number of '+str(type)+' images added to the original dataset')
    # Show
    plt.show()

def plot_err_rate_simu_simple(error_rate_blue_images, nb_new_images, type=['blue_highway', 'blue_river']):
    plt.figure(figsize=(12,6))
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.title('Error rate of blue images when adding '+str(type)+' images')
    i = 0
    for er in error_rate_blue_images:
        plt.plot(er, label='num_added: %s'%nb_new_images[i])
        i += 1
    plt.legend()
    plt.show()

def plot_bar_acc_simu_double(
    global_accuracy,
    highway_accuracy,
    blue_highway_accuracy,
    river_accuracy,
    blue_river_accuracy,
    type_1=['blue_highway', 'blue_river'], type_2=['blue_highway', 'blue_river']
):
    # Define Plot Data
    data=[["Global accuracy", global_accuracy[0], global_accuracy[1], global_accuracy[2], global_accuracy[3], global_accuracy[4], global_accuracy[5]],
        ["Highway accuracy", highway_accuracy[0], highway_accuracy[1], highway_accuracy[2], highway_accuracy[3], highway_accuracy[4], highway_accuracy[5]],
        ["Blue highway accuracy", blue_highway_accuracy[0], blue_highway_accuracy[1], blue_highway_accuracy[2], blue_highway_accuracy[3], blue_highway_accuracy[4], blue_highway_accuracy[5]],
        ["River accuracy", river_accuracy[0], river_accuracy[1], river_accuracy[2], river_accuracy[3], river_accuracy[4], river_accuracy[5]],
        ["Blue river accuracy", blue_river_accuracy[0], blue_river_accuracy[1], blue_river_accuracy[2], blue_river_accuracy[3], blue_river_accuracy[4], blue_river_accuracy[5]]
        ]
    # Plot multiple columns bar chart
    df=pd.DataFrame(data,columns=["Evaluations", "Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"])
    df.plot(x="Evaluations", y=["Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"], kind="bar", figsize=(12,6))
    plt.title('Accuracy when added pairs of '+str(type_1)+' and '+str(type_2)+' images numbers into the original dataset')
    # Show
    plt.show()

def plot_err_rate_simu_double(error_rate_blue_images):
    plt.figure(figsize=(12,6))
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.title('Error rate of blue images when adding '+str(type_1)+' and '+str(type_2)+' images')
    labels=["Original dataset", "nb_H = nb_R", "2 nb_H = nb_R", "3 nb_H = nb_R", "nb_H = 2 nb_R", "nb_H = 3 nb_R"]
    i = 0
    for er in error_rate_blue_images:
        plt.plot(er, label=labels[i])
        i += 1
    plt.legend()
    plt.show()

def plot_err_rate_test(error_rate_blue_images, type_new_images_1, nb_new_images_1, type_new_images_2):
    e_bl_hw = []
    e_bl_rv = []
    e_bl = []
    for i in range(len(error_rate_blue_images)):
        e_bl_hw.append(error_rate_blue_images[i][0])
        e_bl_rv.append(error_rate_blue_images[i][1])
        e_bl.append(error_rate_blue_images[i][2])

    list_number = [*range(0, nb_new_images_1+50, 50)]

    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 8))
    fig.suptitle('Error rate of blue images in test dataset when adding '+str(nb_new_images_1)+' '+str(type_new_images_1)+\
              ' and changing the number of added '+str(type_new_images_2)+' images')
    plt.xlabel('Number of '+str(type_new_images_2)+' images added')

    cmap=cm.get_cmap('viridis')
    normalizer=Normalize(0,10)
    im=cm.ScalarMappable(norm=normalizer)

    axs[0].plot(e_bl_hw[0], 'r*', label='Original dataset')
    axs[0].scatter(x=list_number[1:], y=e_bl_hw[1:], c=e_bl_hw[1:], cmap=cmap, norm=normalizer)
    axs[0].set_ylabel('Err rate bl_hw')
    for x,y in zip(list_number, e_bl_hw):
        label = "{:.2f}".format(y)
        axs[0].annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(e_bl_rv[0], 'r*', label='Original dataset')
    axs[1].scatter(x=list_number[1:], y=e_bl_rv[1:], c=e_bl_rv[1:], cmap=cmap, norm=normalizer)
    axs[1].set_ylabel('Err rate bl_rv')
    for x,y in zip(list_number, e_bl_rv):
        label = "{:.2f}".format(y)
        axs[1].annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(e_bl[0], 'r*', label='Original dataset')
    axs[2].scatter(x=list_number[1:], y=e_bl[1:], c=e_bl[1:], cmap=cmap, norm=normalizer)
    axs[2].set_ylabel('Err rate bl')
    for x,y in zip(list_number, e_bl):
        label = "{:.2f}".format(y)
        axs[2].annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    axs[2].legend()
    axs[2].grid()

    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def plot_acc_test(accuracy, type_new_images_1, nb_new_images_1, type_new_images_2):
    list_number = [*range(0, nb_new_images_1+50, 50)]
    # Create a sample dataframe with an text index
    index = []
    index.append("Original dataset")
    for i in range(len(list_number[1:])):
        index.append("Num_added:"+str(list_number[i+1]))
    plotdata = pd.DataFrame(
        {"Accuracy": accuracy}, 
        index=index)
    # Plot a bar chart
    fig = plotdata.plot(kind="bar", legend=None, figsize=(12,6))
    plt.xticks(rotation=90)
    fig.bar_label(fig.containers[0], label_type='center')
    plt.title('Accuracy in test dataset when adding '+str(nb_new_images_1)+' '+str(type_new_images_1)+\
              ' and changing the number of added '+str(type_new_images_2)+' images')
    plt.ylabel("Accuracy")
    plt.show()

def plot_err_rate_test_double(error_rate_blue_images, type_new_images_1, nb_new_images_1, type_new_images_2, nb_new_images_2):
    e_bl_hw = []
    e_bl_rv = []
    e_bl = []
    for i in range(len(error_rate_blue_images)):
        e_bl_hw.append(error_rate_blue_images[i][0])
        e_bl_rv.append(error_rate_blue_images[i][1])
        e_bl.append(error_rate_blue_images[i][2])
    cmap=cm.get_cmap('viridis')
    normalizer=Normalize(0,10)
    im=cm.ScalarMappable(norm=normalizer)

    fig = plt.figure(figsize=(17, 5))
    fig.suptitle('Error rate of blue images in test dataset when adding '+str(type_new_images_1)+\
              ' and '+str(type_new_images_2)+' images')

    ax0 = fig.add_subplot(131, projection="3d")
    ax0.scatter(nb_new_images_1, nb_new_images_2, e_bl_hw, c=e_bl_hw, cmap=cmap, norm=normalizer)
    ax0.set_xlabel('Nb '+str(type_new_images_1)+' added')
    ax0.set_ylabel('Nb '+str(type_new_images_2)+' added')
    ax0.set_zlabel('Err rate bl_hw')

    ax1 = fig.add_subplot(132, projection="3d")
    ax1.scatter(nb_new_images_1, nb_new_images_2, e_bl_rv, c=e_bl_rv, cmap=cmap, norm=normalizer)
    ax1.set_xlabel('Nb '+str(type_new_images_1)+' added')
    ax1.set_ylabel('Nb '+str(type_new_images_2)+' added')
    ax1.set_zlabel('Err rate bl_rv')

    ax2 = fig.add_subplot(133, projection="3d")
    ax2.scatter(nb_new_images_1, nb_new_images_2, e_bl, c=e_bl, cmap=cmap, norm=normalizer)
    ax2.set_xlabel('Nb '+str(type_new_images_1)+' added')
    ax2.set_ylabel('Nb '+str(type_new_images_2)+' added')
    ax2.set_zlabel('Err rate all bl')

    fig.colorbar(im, ax=[ax0,ax1,ax2])
    plt.show()

def plot_heatmap_err_rate(values, nbs_hw, nbs_rv):
    err_rate_bl_hw = []
    err_rate_bl_rv = []
    err_rate_bl = []

    err_rate_nor = []
    err_rate_nor_hw = []
    err_rate_nor_rv = []

    for i in range(len(values)):
        new_err_bl_hw = []
        new_err_bl_rv = []
        new_err_bl = []

        new_err_nor = []
        new_err_nor_hw = []
        new_err_nor_rv = []
        for j in range(len(values[i])):
            err1 = values[i][j][0]
            err2 = values[i][j][1]
            err3 = values[i][j][2]

            err4 = values[i][j][3]
            err5 = values[i][j][4]
            err6 = values[i][j][5]
            
            new_err_bl_hw.append(err1)
            new_err_bl_rv.append(err2)
            new_err_bl.append(err3)

            new_err_nor.append(err4)
            new_err_nor_hw.append(err5)
            new_err_nor_rv.append(err6)

        err_rate_bl_hw.append(new_err_bl_hw)
        err_rate_bl_rv.append(new_err_bl_rv)
        err_rate_bl.append(new_err_bl)

        err_rate_nor.append(new_err_nor)
        err_rate_nor_hw.append(new_err_nor_hw)
        err_rate_nor_rv.append(new_err_nor_rv)

    annot_kws={'fontsize':8, 
            'fontstyle':'italic',  
            'color':"k",
            'alpha':1.0, 
            'rotation':"horizontal",
            'verticalalignment':'center',
            'backgroundcolor':'w'}

    cbar_kws = {"orientation":"vertical", 
                "shrink":1,
                'extend':'both',
                #'format': '%.0f%%',
                'extendfrac':0.1,
                #"ticks":np.arange(0,100),
                "drawedges":True}

    kwargs = {'cmap':'jet', 'robust':True, 'annot':True,
            'xticklabels':nbs_hw, 'yticklabels':nbs_rv,
            'vmin':values.min(), 'vmax':values.max(),
            'annot_kws':annot_kws, 'cbar_kws':cbar_kws,
            'linewidth':2, 'linecolor':'w', 'fmt':'.2f'}

    fig, ((ax1,ax2,ax3) , (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize = (45,30))
    cbar_ax = fig.add_axes([0.95, .2, .01, .6])

    g1 = sns.heatmap(err_rate_bl_hw,**kwargs, ax=ax1, cbar_ax=cbar_ax)
    g1.invert_yaxis()
    g1.set_title("Blue highway")
    g2 = sns.heatmap(err_rate_bl_rv,**kwargs, ax=ax2, cbar=False)
    g2.invert_yaxis()
    g2.set_title("Blue river")
    g3 = sns.heatmap(err_rate_bl,**kwargs, ax=ax3, cbar=False)
    g3.invert_yaxis()
    g3.set_title("All blue")

    g4 = sns.heatmap(err_rate_nor_hw,**kwargs, ax=ax4, cbar=False)
    g4.invert_yaxis()
    g4.set_title("Normal highway")
    g5 = sns.heatmap(err_rate_nor_rv,**kwargs, ax=ax5, cbar=False)
    g5.invert_yaxis()
    g5.set_title("Normal river")
    g6 = sns.heatmap(err_rate_nor,**kwargs, ax=ax6, cbar=False)
    g6.invert_yaxis()
    g6.set_title("All normal")

    fig.suptitle("Error rate in test set when adding blue highway and blue river images", fontsize=25)
    fig.supylabel("Number of blue highway added", fontsize=25)
    fig.supxlabel("Number of blue river added", fontsize=25)
    plt.show()

def plot_heatmap_accuracy(values, nbs_hw, nbs_rv):
    annot_kws={'fontsize':8, 
            'fontstyle':'italic',  
            'color':"k",
            'alpha':1.0, 
            'rotation':"horizontal",
            'verticalalignment':'center',
            'backgroundcolor':'w'}

    cbar_kws = {"orientation":"vertical", 
                "shrink":1,
                'extend':'both', 
                'extendfrac':0.1,
                "drawedges":True}

    kwargs = {'cmap':'jet', 'robust':True, 'annot':True,
            'xticklabels':nbs_hw, 'yticklabels':nbs_rv,
            'vmin':values.min(), 'vmax':values.max(),
            'annot_kws':annot_kws, 'cbar_kws':cbar_kws,
            'linewidth':2, 'linecolor':'w', 'fmt':'.2f'}

    plt.figure(figsize = (15,9))
    ax = sns.heatmap(values,**kwargs)
    ax.invert_yaxis()
    plt.title("Accuracy in test set when adding blue highway and blue river images", fontsize=15)
    plt.ylabel("Number of blue highway added", fontsize=15)
    plt.xlabel("Number of blue river added", fontsize=15)
    plt.show()

def get_acc_err_rate(path, type, nb_seeds):
    accuracy = []
    err_bl_hw, err_bl_rv, err_bl, err_nor, err_nor_hw, err_nor_rv, err_hw, err_rv = [], [], [], [], [], [], [], []
    for i in range(nb_seeds):
        accuracy_seed = np.load(f"{path}/{type}/accuracy_seed_{i+1}.npy")
        error_rate_seed = np.load(f"{path}/{type}/error_rate_seed_{i+1}.npy")

        accuracy.append(accuracy_seed[0])
        err_bl_hw.append(error_rate_seed[0][0])
        err_bl_rv.append(error_rate_seed[0][1])
        err_bl.append(error_rate_seed[0][2])
        err_nor.append(error_rate_seed[0][3])
        err_nor_hw.append(error_rate_seed[0][4])
        err_nor_rv.append(error_rate_seed[0][5])
        err_hw.append(error_rate_seed[0][6])
        err_rv.append(error_rate_seed[0][7])

    df_err_rate = pd.DataFrame({
        "Err_bl_hw": err_bl_hw,
        "Err_bl_rv": err_bl_rv,
        "Err_bl": err_bl,
        "Err_nor": err_nor,
        "Err_nor_hw": err_nor_hw,
        "Err_nor_rv": err_nor_rv,
        "Err_hw": err_hw,
        "Err_rv": err_rv
    })
    return accuracy, df_err_rate

def get_dataframe(path, model_name, aug_name, nb_hw, nb_rv, nb_seeds):
    Config_1 = tuple((0,0,'CE'))
    Config_2 = tuple((nb_hw, nb_rv,'CE'))
    Config_3 = tuple((0,0,'BC'))
    Config_4 = tuple((0,0,'BB'))
    Config_5 = tuple((0,0,'BC_BB'))

    Config_6 = tuple((64,14,'BC'))
    Config_7 = tuple((158,108,'BC'))
    Config_8 = tuple((250,200,'BC'))
    Config_9 = tuple((nb_hw, nb_rv,'BC'))

    Config_10 = tuple((64,14,'BB'))
    Config_11 = tuple((158,108,'BB'))
    Config_12 = tuple((250,200,'BB'))
    Config_13 = tuple((nb_hw, nb_rv,'BB'))

    Config_14 = tuple((64,14,'BC_BB'))
    Config_15 = tuple((158,108,'BC_BB'))
    Config_16 = tuple((250,200,'BC_BB'))
    Config_17 = tuple((nb_hw, nb_rv,'BC_BB'))

    config_name = [Config_1, Config_2, Config_3, Config_4, Config_5, Config_6, Config_7, Config_8, Config_9, Config_10, Config_11, Config_12, Config_13, Config_14, Config_15, Config_16, Config_17]
    method_name = ['Baseline', 'Aug', 'BC', 'BB', 'BC_BB', 'Aug_BC_1', 'Aug_BC_2', 'Aug_BC_3', 'Aug_BC_4', 'Aug_BB_1', 'Aug_BB_2', 'Aug_BB_3', 'Aug_BB_4',\
                   'Aug_BC_BB_1', 'Aug_BC_BB_2', 'Aug_BC_BB_3', 'Aug_BC_BB_4']

    folder_name = ['Baseline', f'{aug_name}/Aug', 'BC', 'BB', 'BC_BB', f'{aug_name}/Aug_BC', f'{aug_name}/Aug_BC', f'{aug_name}/Aug_BC',
                   f'{aug_name}/Aug_BC', f'{aug_name}/Aug_BB', f'{aug_name}/Aug_BB', f'{aug_name}/Aug_BB', f'{aug_name}/Aug_BB',
                   f'{aug_name}/Aug_BC_BB', f'{aug_name}/Aug_BC_BB', f'{aug_name}/Aug_BC_BB', f'{aug_name}/Aug_BC_BB']

    row_eles = []
    metrics = ['Err_bl_hw', 'Err_bl_rv', 'Err_bl', 'Err_nor', 'Err_nor_hw', 'Err_nor_rv', 'Err_hw', 'Err_rv']

    for k in range(len(config_name)):
        for i in range(nb_seeds):
            for j in range(len(metrics)):
                error_rate_seed = np.load(f"{path}/{folder_name[k]}/error_rate_seed_{i+1}.npy")
                con_name = f"Config_{k+1}"

                if (con_name == 'Config_6') or (con_name == 'Config_10') or (con_name == 'Config_14'):
                    metric_value = error_rate_seed[0][j]

                elif (con_name == 'Config_7') or (con_name == 'Config_11') or (con_name == 'Config_15'):
                    metric_value = error_rate_seed[1][j]

                elif (con_name == 'Config_8') or (con_name == 'Config_12') or (con_name == 'Config_16'):
                    metric_value = error_rate_seed[2][j]

                elif (con_name == 'Config_9') or (con_name == 'Config_13') or (con_name == 'Config_17'):
                    metric_value = error_rate_seed[3][j]

                else:
                    metric_value = error_rate_seed[0][j]

                new_row = [model_name, aug_name, i+1, con_name, metrics[j], metric_value, config_name[k], method_name[k]]
                row_eles.append(new_row)

    df = pd.DataFrame(row_eles, columns=['Model name', 'Augmentation method', 'Seed', 'Config', 'Metric name', 'Metric value', 'Mapping', 'Method'])
    return df 

def get_dataframe_2(path, model_name, aug_name, nb_hw, nb_rv, nb_seeds):
    Config_1 = tuple((0,0,'CE'))
    Config_2 = tuple((nb_hw, nb_rv,'CE'))

    config_name = [Config_1, Config_2]
    method_name = ['Baseline', 'Aug']

    folder_name = ['Baseline', f'{aug_name}/Aug']

    row_eles = []
    metrics = ['Err_bl_hw', 'Err_bl_rv', 'Err_bl', 'Err_nor', 'Err_nor_hw', 'Err_nor_rv', 'Err_hw', 'Err_rv']

    for k in range(len(config_name)):
        for i in range(nb_seeds):
            for j in range(len(metrics)):
                error_rate_seed = np.load(f"{path}/{folder_name[k]}/error_rate_seed_{i+1}.npy")
                con_name = f"Config_{k+1}"

                metric_value = error_rate_seed[0][j]

                new_row = [model_name, aug_name, i+1, con_name, metrics[j], metric_value, config_name[k], method_name[k]]
                row_eles.append(new_row)

    df = pd.DataFrame(row_eles, columns=['Model name', 'Augmentation method', 'Seed', 'Config', 'Metric name', 'Metric value', 'Mapping', 'Method'])
    return df

def get_dataframe_3(path, model_name, aug_name, nb_hw, nb_rv, nb_seeds):
    Config_1 = tuple((0,0,'CE'))
    Config_2 = tuple((nb_hw, nb_rv,'CE'))

    config_name = [Config_1, Config_2]
    method_name = ['Baseline', 'Aug']

    folder_name = ['Baseline', f'{aug_name}']

    row_eles = []
    metrics = ['Err_bl_hw', 'Err_bl_rv', 'Err_bl', 'Err_nor', 'Err_nor_hw', 'Err_nor_rv', 'Err_hw', 'Err_rv']

    for k in range(len(config_name)):
        for i in range(nb_seeds):
            for j in range(len(metrics)):
                error_rate_seed = np.load(f"{path}/{folder_name[k]}/error_rate_seed_{i+1}.npy")
                con_name = f"Config_{k+1}"

                metric_value = error_rate_seed[0][j]

                new_row = [model_name, aug_name, i+1, con_name, metrics[j], metric_value, config_name[k], method_name[k]]
                row_eles.append(new_row)

    df = pd.DataFrame(row_eles, columns=['Model name', 'Augmentation method', 'Seed', 'Config', 'Metric name', 'Metric value', 'Mapping', 'Method'])
    return df

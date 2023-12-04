import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_res_celeba_global(df, mode_redis, save_path = None):
    male_black = df[df['Metric name']=="err_male_black"]
    male_blond = df[df['Metric name']=="err_male_blond"]
    male_brown = df[df['Metric name']=="err_male_brown"]
    male_gray = df[df['Metric name']=="err_male_gray"]

    female_black = df[df['Metric name']=="err_female_black"]
    female_blond = df[df['Metric name']=="err_female_blond"]
    female_brown = df[df['Metric name']=="err_female_brown"]
    female_gray = df[df['Metric name']=="err_female_gray"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Global Male", "Global Female"), vertical_spacing = 0.15)

    x_axis = ['Over/5', 'Over/10', 'Over/15', 'Over/20',
            'Under/5', 'Under/10', 'Under/15', 'Under/20',
            'Classic/5', 'Classic/10', 'Classic/15', 'Classic/20',
            'CUT_face/5', 'CUT_face/10', 'CUT_face/15', 'CUT_face/20',
            'CUT_hair/5', 'CUT_hair/10', 'CUT_hair/15', 'CUT_hair/20',
            'My_CyGAN/5', 'My_CyGAN/10', 'My_CyGAN/15', 'My_CyGAN/20',
            'CyGAN_face/5', 'CyGAN_face/10', 'CyGAN_face/15', 'CyGAN_face/20',
            'CyGAN_hair/5', 'CyGAN_hair/10', 'CyGAN_hair/15', 'CyGAN_hair/20']

    # Subplot 1
    fig.add_trace(
        go.Bar(
            legendgroup="group",
            name="Redistribution",
            legendgrouptitle_text="Black Hair Male",
            x=x_axis,
            y=male_black[male_black['Method']=='Aug']['Metric value'],
            marker={'color': 'darkgreen'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_black[male_black['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkgreen",
                row=1, col=1,
                annotation_text="Baseline Black Hair Male", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="darkgreen")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Blond Hair Male",
            x=x_axis,
            y=male_blond[male_blond['Method']=='Aug']['Metric value'],
            marker={'color': 'lightskyblue'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_blond[male_blond['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="lightskyblue",
                row=1, col=1,
                annotation_text="Baseline Blond Hair Male", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="lightskyblue")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Brown Hair Male",
            x=x_axis,
            y=male_brown[male_brown['Method']=='Aug']['Metric value'],
            marker={'color': 'darkorange'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_brown[male_brown['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkorange",
                row=1, col=1,
                annotation_text="Baseline Brown Hair Male", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="darkorange")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Gray Hair Male",
            x=x_axis,
            y=male_gray[male_gray['Method']=='Aug']['Metric value'],
            marker={'color': 'red'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_gray[male_gray['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="red",
                row=1, col=1,
                annotation_text="Baseline Gray Hair Male", 
                annotation_position="bottom left",
                annotation_font_size=10,
                annotation_font_color="red")

    # Subplot 2
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Black Hair Female",
            x=x_axis,
            y=female_black[female_black['Method']=='Aug']['Metric value'],
            marker={'color': 'darkviolet'}
        ),
        row=2,
        col=1
    )
    fig.add_hline(y=female_black[female_black['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkviolet",
                row=2, col=1,
                annotation_text="Baseline Black Hair Female", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="darkviolet")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Blond Hair Female",
            x=x_axis,
            y=female_blond[female_blond['Method']=='Aug']['Metric value'],
            marker={'color': 'black'}
        ),
        row=2,
        col=1
    )
    fig.add_hline(y=female_blond[female_blond['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="black",
                row=2, col=1,
                annotation_text="Baseline Blond Hair Female", 
                annotation_position="bottom right",
                annotation_font_size=10,
                annotation_font_color="black")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Brown Hair Female",
            x=x_axis,
            y=female_brown[female_brown['Method']=='Aug']['Metric value'],
            marker={'color': 'blue'}
        ),
        row=2,
        col=1
    )
    fig.add_hline(y=female_brown[female_brown['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="blue",
                row=2, col=1,
                annotation_text="Baseline Brown Hair Female", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="blue")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Gray Hair Female",
            x=x_axis,
            y=female_gray[female_gray['Method']=='Aug']['Metric value'],
            marker={'color': 'burlywood'}
        ),
        row=2,
        col=1
    )
    fig.add_hline(y=female_gray[female_gray['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="burlywood",
                row=2, col=1,
                annotation_text="Baseline Gray Hair Female", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="burlywood")

    fig.update_layout(
        height=700, width=2000,
        title=f'Comparison the error rate of images classes after redistribution {mode_redis} images', # title of plot
        barmode="group",
        bargap=0.5, # gap between bars of adjacent location coordinates
        bargroupgap=0.02, # gap between bars of the same location coordinates
        margin=dict(l=15, r=15, t=70, b=20),
        #paper_bgcolor="LightSteelBlue",
    )

    fig.show()
    if save_path:
        fig.write_image(save_path)

def analyse_string(string):
    new_string = []
    list_string = string.split('_')
    for string in list_string:
        string = string.capitalize()
        new_string.append(string)
    new_string = ' '.join(new_string)
    return new_string

def plot_res_celeba_minority(df, minority_group, mode_redis, save_path = None):
    value = df[df['Metric name']==f"err_{minority_group}"]

    title = analyse_string(minority_group)
    fig = make_subplots(rows=1, cols=1)

    x_axis = ['Over/5', 'Over/10', 'Over/15', 'Over/20',
            'Under/5', 'Under/10', 'Under/15', 'Under/20',
            'Classic/5', 'Classic/10', 'Classic/15', 'Classic/20',
            'CUT_face/5', 'CUT_face/10', 'CUT_face/15', 'CUT_face/20',
            'CUT_hair/5', 'CUT_hair/10', 'CUT_hair/15', 'CUT_hair/20',
            'My_CyGAN/5', 'My_CyGAN/10', 'My_CyGAN/15', 'My_CyGAN/20',
            'CyGAN_face/5', 'CyGAN_face/10', 'CyGAN_face/15', 'CyGAN_face/20',
            'CyGAN_hair/5', 'CyGAN_hair/10', 'CyGAN_hair/15', 'CyGAN_hair/20']

    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text=f"Only {title}",
            x=x_axis,
            y=value[value['Method']=='Aug']['Metric value'],
            marker={'color': 'lightskyblue'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=value[value['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="lightskyblue",
                row=1, col=1,
                annotation_text="Baseline", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="lightskyblue")

    fig.update_layout(
        height=400, width=900,
        title=f'The error rate of images classes after redistribution {mode_redis} images', # title of plot
        barmode="group",
        bargap=0.5, # gap between bars of adjacent location coordinates
        bargroupgap=0.02, # gap between bars of the same location coordinates
        margin=dict(l=15, r=15, t=70, b=20),
        paper_bgcolor="LightSteelBlue",
    )

    fig.show()
    if save_path:
        fig.write_image(save_path)

def plot_best_res_celeba(df, mode_redis, best_method, save_path = None):
    male_black = df[df['Metric name']=="err_male_black"]
    male_blond = df[df['Metric name']=="err_male_blond"]
    male_brown = df[df['Metric name']=="err_male_brown"]
    male_gray = df[df['Metric name']=="err_male_gray"]

    female_black = df[df['Metric name']=="err_female_black"]
    female_blond = df[df['Metric name']=="err_female_blond"]
    female_brown = df[df['Metric name']=="err_female_brown"]
    female_gray = df[df['Metric name']=="err_female_gray"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Male", "Female"))

    x_axis = ['Over/5', 'Over/10', 'Over/15', 'Over/20',
            'Under/5', 'Under/10', 'Under/15', 'Under/20',
            'Classic/5', 'Classic/10', 'Classic/15', 'Classic/20',
            'CUT_face/5', 'CUT_face/10', 'CUT_face/15', 'CUT_face/20',
            'CUT_hair/5', 'CUT_hair/10', 'CUT_hair/15', 'CUT_hair/20',
            'My_CyGAN/5', 'My_CyGAN/10', 'My_CyGAN/15', 'My_CyGAN/20',
            'CyGAN_face/5', 'CyGAN_face/10', 'CyGAN_face/15', 'CyGAN_face/20',
            'CyGAN_hair/5', 'CyGAN_hair/10', 'CyGAN_hair/15', 'CyGAN_hair/20']

    for position, item in enumerate(x_axis):
        if item == best_method:
            id = position

    # Subplot 1
    fig.add_trace(
        go.Bar(
            legendgroup="group",
            name="Redistribution",
            legendgrouptitle_text="Black Hair Male",
            x=[best_method],
            y=[male_black[male_black['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'darkgreen'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_black[male_black['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkgreen",
                row=1, col=1,
                annotation_text="Baseline Black Hair Male", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="darkgreen")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Blond Hair Male",
            x=[best_method],
            y=[male_blond[male_blond['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'lightskyblue'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_blond[male_blond['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="lightskyblue",
                row=1, col=1,
                annotation_text="Baseline Blond Hair Male", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="lightskyblue")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Brown Hair Male",
            x=[best_method],
            y=[male_brown[male_brown['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'darkorange'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_brown[male_brown['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkorange",
                row=1, col=1,
                annotation_text="Baseline Brown Hair Male", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="darkorange")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Gray Hair Male",
            x=[best_method],
            y=[male_gray[male_gray['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'red'}
        ),
        row=1,
        col=1
    )
    fig.add_hline(y=male_gray[male_gray['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="red",
                row=1, col=1,
                annotation_text="Baseline Gray Hair Male", 
                annotation_position="bottom left",
                annotation_font_size=10,
                annotation_font_color="red")

    # Subplot 2
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Black Hair Female",
            x=[best_method],
            y=[female_black[female_black['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'darkviolet'}
        ),
        row=1,
        col=2
    )
    fig.add_hline(y=female_black[female_black['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="darkviolet",
                row=1, col=2,
                annotation_text="Baseline Black Hair Female", 
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="darkviolet")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Blond Hair Female",
            x=[best_method],
            y=[female_blond[female_blond['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'black'}
        ),
        row=1,
        col=2
    )
    fig.add_hline(y=female_blond[female_blond['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="black",
                row=1, col=2,
                annotation_text="Baseline Blond Hair Female", 
                annotation_position="bottom right",
                annotation_font_size=10,
                annotation_font_color="black")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Brown Hair Female",
            x=[best_method],
            y=[female_brown[female_brown['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'blue'}
        ),
        row=1,
        col=2
    )
    fig.add_hline(y=female_brown[female_brown['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="blue",
                row=1, col=2,
                annotation_text="Baseline Brown Hair Female", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="blue")
    fig.add_trace(
        go.Bar(
            name="Redistribution",
            legendgrouptitle_text="Gray Hair Female",
            x=[best_method],
            y=[female_gray[female_gray['Method']=='Aug']['Metric value'].iloc[id]],
            marker={'color': 'burlywood'}
        ),
        row=1,
        col=2
    )
    fig.add_hline(y=female_gray[female_gray['Method']=='Baseline']['Metric value'].iloc[0],
                line_dash="dot", line_color="burlywood",
                row=1, col=2,
                annotation_text="Baseline Gray Hair Female", 
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="burlywood")

    fig.update_layout(
        height=500, width=1200,
        title=f'Comparison the error rate of images classes after redistribution {mode_redis} images with best DA method', # title of plot
        barmode="group",
        bargap=0.5, # gap between bars of adjacent location coordinates
        bargroupgap=0.02, # gap between bars of the same location coordinates
        margin=dict(l=15, r=15, t=70, b=20)
    )

    fig.show()
    if save_path:
        fig.write_image(save_path)

def plot_bar_err_rate(err_rate):
    plt.figure(figsize = (7,7))
    bars = ('err_female_gray', 'err_female_blond', 'err_female', 'err_male', 'err_male_gray', 'err_male_blond', 'err_gray', 'err_blond')
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
    plt.xticks(x_pos, bars, rotation=75)
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

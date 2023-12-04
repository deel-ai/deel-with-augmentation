import torch
from cnn_model import ResNet18
from torch.utils.data import TensorDataset, DataLoader

def simulation_simple(num_epochs, valid_loader, X_train, S_train, y_train, new_images, nb_new_images, device, save_folder_path, type=['blue_highway', 'blue_river']):
    global_accuracy = []
    highway_accuracy = []
    blue_highway_accuracy = []
    river_accuracy = []
    blue_river_accuracy = []
    for i in range(len(nb_new_images)):
        if type == 'blue_highway':
            new_X_train = torch.cat((X_train, new_images[torch.randint(len(new_images), (nb_new_images[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images[i])), 0)
            new_y_train = torch.cat((y_train, torch.zeros(nb_new_images[i]).long()), 0)
        elif type == 'blue_river':
            new_X_train = torch.cat((X_train, new_images[torch.randint(len(new_images), (nb_new_images[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images[i])), 0)
            new_y_train = torch.cat((y_train, torch.ones(nb_new_images[i]).long()), 0)
        
        train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
        train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

        new_path = save_folder_path + '/' + 'simu_sim' + '_' + str(type) + '_' + str(nb_new_images[i]) + '.pt'
        cnn_model = ResNet18(device=device)
        _, _, _, _ = cnn_model.train(train_loader=train_dataloader, valid_loader=valid_loader, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
        glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc = cnn_model.evaluate(new_X_train, new_y_train, new_S_train)

        global_accuracy.append(glo_acc)
        highway_accuracy.append(hw_acc)
        blue_highway_accuracy.append(bhw_acc)
        river_accuracy.append(rv_acc)
        blue_river_accuracy.append(brv_acc)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()   
    return global_accuracy, highway_accuracy, blue_highway_accuracy, river_accuracy, blue_river_accuracy

def simulation_double(num_epochs, valid_loader, X_train, S_train, y_train, new_images_1, nb_new_images_1, new_images_2, nb_new_images_2, device, save_folder_path, type_1=['blue_highway', 'blue_river'], type_2=['blue_highway', 'blue_river']):
    global_accuracy = []
    highway_accuracy = []
    blue_highway_accuracy = []
    river_accuracy = []
    blue_river_accuracy = []
    for i in range(len(nb_new_images_1)):
        if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[i])), 0)
            new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1[i]).long(), torch.ones(nb_new_images_2[i]).long()), 0)
        elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[i])), 0)
            new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1[i]).long(), torch.zeros(nb_new_images_2[i]).long()), 0)

        train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
        train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

        new_path = save_folder_path + '/' + 'simu_dou' + '_' + str(type_1) + '_' + str(nb_new_images_1[i]) + '_' + str(type_2) + '_' + str(nb_new_images_2[i]) + '.pt'
        cnn_model = ResNet18(device=device)
        _, _, _, _ = cnn_model.train(train_loader=train_dataloader, valid_loader=valid_loader, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
        glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc = cnn_model.evaluate(new_X_train, new_y_train, new_S_train)

        global_accuracy.append(glo_acc)
        highway_accuracy.append(hw_acc)
        blue_highway_accuracy.append(bhw_acc)
        river_accuracy.append(rv_acc)
        blue_river_accuracy.append(brv_acc)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()  
    return global_accuracy, highway_accuracy, blue_highway_accuracy, river_accuracy, blue_river_accuracy

def evaluate_simple_in_test_set(test_set, num_epochs, X_train, S_train, y_train,
                        new_images_1, new_images_2, nb_new_images_1,
                        device, save_folder_path,
                        type_new_images_1=['blue_highway', 'blue_river'],
                        type_new_images_2=['blue_highway', 'blue_river']):
    accuracy = []
    error_rate = []

    dataset = TensorDataset(X_train, y_train, S_train)
    dataloader = DataLoader(dataset, batch_size=56, shuffle=True)
    new_path = save_folder_path + '/' + 'eva_sim' + '_' + str(type_new_images_1) + '_' + 'origin' + '.pt'
    model = ResNet18(device=device)
    _, _, _, _ = model.train(train_loader=dataloader, valid_loader=test_set, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
    _, acc, _, ratio_wrong = model.test(test_loader=test_set, test_gobal=True)
    accuracy.append(acc)
    error_rate.append(ratio_wrong)
    torch.cuda.empty_cache()

    list_number = [*range(50, nb_new_images_1+50, 50)]

    for i in range(len(list_number)):
        if type_new_images_1 == 'blue_highway':
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1,))], new_images_2[torch.randint(len(new_images_2), (list_number[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1), torch.zeros(list_number[i])), 0)
            new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1).long(), torch.ones(list_number[i]).long()), 0)
        elif type_new_images_1 == 'blue_river':
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1,))], new_images_2[torch.randint(len(new_images_2), (list_number[i],))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1), torch.zeros(list_number[i])), 0)
            new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1).long(), torch.zeros(list_number[i]).long()), 0)
        
        train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
        train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

        new_path = save_folder_path + '/' + 'eva_sim' + '_' + str(type_new_images_1) + '_' + str(nb_new_images_1) + '_' + str(type_new_images_2) + '_' + str(list_number[i]) + '.pt'
        cnn_model = ResNet18(device=device)
        _, _, _, _ = cnn_model.train(train_loader=train_dataloader, valid_loader=test_set, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
        _, new_acc, _, new_ratio_wrong = cnn_model.test(test_loader=test_set, test_gobal=True)

        accuracy.append(new_acc)
        error_rate.append(new_ratio_wrong)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()
    return accuracy, error_rate

def evaluate_double_in_test_set(test_set, num_epochs, X_train, S_train, y_train,
                        new_images_1, nb_new_images_1, 
                        new_images_2, nb_new_images_2,
                        device, save_folder_path,
                        type_1=['blue_highway', 'blue_river'],
                        type_2=['blue_highway', 'blue_river']):
    accuracy = []
    error_rate = []

    for i in range(len(nb_new_images_1)):
        new_acc = []
        new_err_rate = []
        for j in range(len(nb_new_images_2)):
            if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
                new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[j],))]), 0)
                new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[j])), 0)
                new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1[i]).long(), torch.ones(nb_new_images_2[j]).long()), 0)
            elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
                new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[j],))]), 0)
                new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[j])), 0)
                new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1[i]).long(), torch.zeros(nb_new_images_2[j]).long()), 0)

            train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
            train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

            new_path = save_folder_path + '/' + 'eva_dou' + '_' + str(type_1) + '_' + str(nb_new_images_1[i]) + '_' + str(type_2) + '_' + str(nb_new_images_2[j]) + '.pt'
            cnn_model = ResNet18(device=device)
            _, _, _, _ = cnn_model.train(train_loader=train_dataloader, valid_loader=test_set, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
            _, acc, _, ratio_wrong = cnn_model.test(test_loader=test_set, test_gobal=True)

            new_acc.append(acc)
            new_err_rate.append(ratio_wrong)
            print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
            torch.cuda.empty_cache()
            
        accuracy.append(new_acc)
        error_rate.append(new_err_rate)
    return accuracy, error_rate

def load_eva_model(model: ResNet18, path):
    model = model.load_state_dict(torch.load(save_path))
    _, acc, _, ratio_wrong = model.test(test_loader=test_dataloader, test_gobal=True)

    new_acc.append(acc)
    new_err_rate.append(ratio_wrong)
    print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
    torch.cuda.empty_cache()
